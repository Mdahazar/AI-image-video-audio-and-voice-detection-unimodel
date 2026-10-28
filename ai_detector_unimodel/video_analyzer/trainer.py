import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
import hashlib
import shutil

class VideoDetectorTrainer:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.model = None
        self.scaler = StandardScaler()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.trained_videos_file = 'models/trained_videos.json'
        self.datasetstrain_path = 'datasetstrain'  # Same folder as images
        
    def update_progress(self, current, stage, message, accuracy=0.0, epoch=0, total_epochs=0):
        if self.progress_callback:
            self.progress_callback(current, stage, message, accuracy, epoch, total_epochs)
    
    def get_video_hash(self, video_path):
        """Generate a unique hash for the video file"""
        try:
            file_stats = os.stat(video_path)
            hash_input = f"{video_path}_{file_stats.st_size}_{file_stats.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return None
    
    def load_trained_videos(self):
        """Load the list of already trained videos"""
        try:
            if os.path.exists(self.trained_videos_file):
                with open(self.trained_videos_file, 'r') as f:
                    return set(json.load(f))
            return set()
        except:
            return set()
    
    def save_trained_videos(self, trained_videos):
        """Save the list of trained videos"""
        try:
            os.makedirs('models', exist_ok=True)
            with open(self.trained_videos_file, 'w') as f:
                json.dump(list(trained_videos), f)
        except Exception as e:
            print(f"Error saving trained videos list: {e}")
    
    def get_new_videos_count(self):
        """Get count of new videos that haven't been trained yet"""
        trained_videos = self.load_trained_videos()
        new_real_count = 0
        new_ai_count = 0
        
        # Check real videos in datasetstrain/real_video
        real_path = os.path.join(self.datasetstrain_path, 'real_video')
        if os.path.exists(real_path):
            for video_file in os.listdir(real_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                    video_path = os.path.join(real_path, video_file)
                    video_hash = self.get_video_hash(video_path)
                    if video_hash and video_hash not in trained_videos:
                        new_real_count += 1
        
        # Check AI videos in datasetstrain/ai_video
        ai_path = os.path.join(self.datasetstrain_path, 'ai_video')
        if os.path.exists(ai_path):
            for video_file in os.listdir(ai_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                    video_path = os.path.join(ai_path, video_file)
                    video_hash = self.get_video_hash(video_path)
                    if video_hash and video_hash not in trained_videos:
                        new_ai_count += 1
        
        return new_real_count, new_ai_count
    
    def prepare_dataset_for_training(self, use_only_new=True):
        """Prepare dataset for training using ONLY datasetstrain folder"""
        trained_videos = self.load_trained_videos()
        
        X = []
        y = []
        video_count = 0
        all_trained_videos = set(trained_videos)
        
        # Process real videos from datasetstrain/real_video ONLY
        real_path = os.path.join(self.datasetstrain_path, 'real_video')
        if os.path.exists(real_path):
            for video_file in os.listdir(real_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                    video_path = os.path.join(real_path, video_file)
                    video_hash = self.get_video_hash(video_path)
                    
                    # Only use videos that haven't been trained yet (or all if not use_only_new)
                    if not use_only_new or (video_hash and video_hash not in trained_videos):
                        self.update_progress(15, 'processing', f'Processing real video: {video_file}')
                        features, _ = self.extract_video_features(video_path)
                        if features and len(features) > 0:
                            X.append(features)
                            y.append(0)  # 0 for real
                            video_count += 1
                            
                            if video_hash:
                                all_trained_videos.add(video_hash)
                        else:
                            print(f"⚠️ Could not extract features from {video_file}")
        
        # Process AI videos from datasetstrain/ai_video ONLY
        ai_path = os.path.join(self.datasetstrain_path, 'ai_video')
        if os.path.exists(ai_path):
            for video_file in os.listdir(ai_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                    video_path = os.path.join(ai_path, video_file)
                    video_hash = self.get_video_hash(video_path)
                    
                    # Only use videos that haven't been trained yet (or all if not use_only_new)
                    if not use_only_new or (video_hash and video_hash not in trained_videos):
                        self.update_progress(25, 'processing', f'Processing AI video: {video_file}')
                        features, _ = self.extract_video_features(video_path)
                        if features and len(features) > 0:
                            X.append(features)
                            y.append(1)  # 1 for AI
                            video_count += 1
                            
                            if video_hash:
                                all_trained_videos.add(video_hash)
                        else:
                            print(f"⚠️ Could not extract features from {video_file}")
        
        print(f"✅ Prepared {len(X)} videos for training ({np.sum(y == 0)} real, {np.sum(y == 1)} AI)")
        return np.array(X), np.array(y), video_count, all_trained_videos
    
    def extract_video_features(self, video_path, max_frames=30):
        """Extract features from video using MediaPipe"""
        features = []
        video_info = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return None, {}
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                print(f"❌ No frames in video: {video_path}")
                cap.release()
                return None, {}
            
            # Sample frames evenly
            frame_interval = max(1, total_frames // max_frames)
            frame_count = 0
            features_extracted = 0
            
            while features_extracted < max_frames and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frame_features = self.extract_frame_features(frame)
                    if frame_features is not None:
                        features.extend(frame_features)
                        features_extracted += 1
                
                frame_count += 1
            
            cap.release()
            
            if len(features) == 0:
                print(f"❌ No features extracted from video: {video_path}")
                return None, {}
            
            # Pad features to consistent length
            target_length = 150  # 30 frames * 5 features
            if len(features) < target_length:
                features.extend([0] * (target_length - len(features)))
            else:
                features = features[:target_length]
            
            return features, video_info
            
        except Exception as e:
            print(f"❌ Error processing video {video_path}: {e}")
            return None, {}
    
    def extract_frame_features(self, frame):
        """Extract features from a single frame using MediaPipe"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            features = []
            
            # Basic frame statistics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features.extend([
                np.mean(gray), np.std(gray), np.median(gray)
            ])
            
            # MediaPipe Face Mesh features
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                features.append(1)  # Face detected
                features.append(len(face_results.multi_face_landmarks))  # Number of faces
                
                # Extract key facial landmarks
                for face_landmarks in face_results.multi_face_landmarks[:1]:
                    key_points = []
                    # Key facial points: left eye, right eye, nose, mouth
                    key_indices = [33, 263, 1, 61]  # Example indices
                    for idx in key_indices:
                        if idx < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[idx]
                            key_points.extend([landmark.x, landmark.y])
                    
                    features.extend(key_points[:8])  # Use first 8 coordinates
            else:
                features.extend([0, 0] + [0] * 8)  # No face detected
            
            # Texture and edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
            features.append(edge_density)
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(laplacian_var)
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting frame features: {e}")
            return None
    
    def create_model(self, input_shape):
        """Create a neural network model for video detection"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def move_trained_videos_to_dataset(self):
        """Move videos from datasetstrain to datasets folder after successful training"""
        try:
            # Move real videos
            real_train_path = os.path.join(self.datasetstrain_path, 'real_video')
            real_dataset_path = os.path.join('datasets', 'real_video')
            
            if os.path.exists(real_train_path):
                moved_count = 0
                for video_file in os.listdir(real_train_path):
                    if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                        src_path = os.path.join(real_train_path, video_file)
                        dst_path = os.path.join(real_dataset_path, video_file)
                        
                        # Ensure destination directory exists
                        os.makedirs(real_dataset_path, exist_ok=True)
                        
                        # Move file
                        shutil.move(src_path, dst_path)
                        moved_count += 1
                        print(f"✅ Moved {video_file} to datasets/real_video")
                
                print(f"✅ Moved {moved_count} real videos to datasets")
            
            # Move AI videos
            ai_train_path = os.path.join(self.datasetstrain_path, 'ai_video')
            ai_dataset_path = os.path.join('datasets', 'ai_video')
            
            if os.path.exists(ai_train_path):
                moved_count = 0
                for video_file in os.listdir(ai_train_path):
                    if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                        src_path = os.path.join(ai_train_path, video_file)
                        dst_path = os.path.join(ai_dataset_path, video_file)
                        
                        # Ensure destination directory exists
                        os.makedirs(ai_dataset_path, exist_ok=True)
                        
                        # Move file
                        shutil.move(src_path, dst_path)
                        moved_count += 1
                        print(f"✅ Moved {video_file} to datasets/ai_video")
                
                print(f"✅ Moved {moved_count} AI videos to datasets")
            
            print("✅ All trained videos moved from datasetstrain to datasets folder")
            
        except Exception as e:
            print(f"❌ Error moving trained videos: {e}")
    
    def train_with_progress(self, use_only_new=True):
        """Train the model with progress updates - using ONLY datasetstrain folder"""
        try:
            # Create datasetstrain video directories
            os.makedirs(os.path.join(self.datasetstrain_path, 'real_video'), exist_ok=True)
            os.makedirs(os.path.join(self.datasetstrain_path, 'ai_video'), exist_ok=True)
            
            # Check if model exists for incremental training
            model_exists = os.path.exists('models/video_detector_model.h5')
            
            if use_only_new and model_exists:
                # Check if there are new videos to train in datasetstrain
                new_real_count, new_ai_count = self.get_new_videos_count()
                total_new = new_real_count + new_ai_count
                
                if total_new == 0:
                    self.update_progress(100, 'completed', 'No new videos to train! Model is up to date.', accuracy=1.0)
                    return None, 1.0
                
                self.update_progress(5, 'checking', f'Found {new_real_count} new real and {new_ai_count} new AI videos in datasetstrain')
            
            # Prepare dataset (get features from datasetstrain ONLY)
            self.update_progress(10, 'loading', 'Extracting features from new videos...')
            X, y, video_count, trained_videos = self.prepare_dataset_for_training(use_only_new)
            
            if video_count == 0:
                self.update_progress(100, 'completed', 'No videos found in datasetstrain for training!', accuracy=0.0)
                return None, 0.0
            
            # Check if we have videos in both categories
            real_count = np.sum(y == 0)
            ai_count = np.sum(y == 1)
            
            if real_count == 0 or ai_count == 0:
                error_msg = f"Need videos in both categories! Found {real_count} real and {ai_count} AI videos in datasetstrain."
                self.update_progress(0, 'error', error_msg)
                raise ValueError(error_msg)
            
            training_type = "incremental" if (use_only_new and model_exists) else "full"
            self.update_progress(30, 'preprocessing', f'{training_type.capitalize()} training with {video_count} NEW videos from datasetstrain...')
            
            # For very small datasets, handle differently
            if video_count < 4:
                print("⚠️ Small dataset detected, using simple training approach")
                # Use all data for training
                X_scaled = self.scaler.fit_transform(X)
                
                if use_only_new and model_exists:
                    # Load existing model for incremental training
                    self.model = keras.models.load_model('models/video_detector_model.h5')
                    self.scaler = joblib.load('models/video_scaler.pkl')
                    X_scaled = self.scaler.transform(X)
                    current_epochs = 5
                else:
                    # Create new model
                    self.model = self.create_model(X.shape[1])
                    current_epochs = 10
                
                # Train without validation split for small datasets
                history = self.model.fit(
                    X_scaled, y,
                    epochs=current_epochs,
                    batch_size=min(4, len(X)),
                    verbose=0
                )
                
                accuracy = history.history['accuracy'][-1] if 'accuracy' in history.history else 0.8
                
            else:
                # Standard training for larger datasets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                if use_only_new and model_exists:
                    # Load existing model and scaler
                    self.model = keras.models.load_model('models/video_detector_model.h5')
                    self.scaler = joblib.load('models/video_scaler.pkl')
                    X_train_scaled = self.scaler.transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    current_epochs = 8
                else:
                    # Scale features and create new model
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    self.model = self.create_model(X_train.shape[1])
                    current_epochs = 15
                
                # Custom callback for progress updates
                class ProgressCallback(keras.callbacks.Callback):
                    def __init__(self, trainer, total_epochs):
                        self.trainer = trainer
                        self.total_epochs = total_epochs
                    
                    def on_epoch_end(self, epoch, logs=None):
                        progress = 50 + ((epoch + 1) / self.total_epochs) * 40
                        self.trainer.update_progress(
                            progress, 'training',
                            f'Training epoch {epoch+1}/{self.total_epochs}',
                            logs.get('accuracy', 0) if logs else 0,
                            epoch + 1,
                            self.total_epochs
                        )
                
                self.update_progress(50, 'training', 'Starting model training with NEW videos only...')
                
                history = self.model.fit(
                    X_train_scaled, y_train,
                    epochs=current_epochs,
                    batch_size=8,
                    validation_data=(X_test_scaled, y_test),
                    verbose=0,
                    callbacks=[ProgressCallback(self, current_epochs)]
                )
                
                # Evaluate model
                self.update_progress(95, 'validation', 'Validating model...')
                y_pred = (self.model.predict(X_test_scaled) > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred)
            
            # Save model and update trained videos list
            os.makedirs('models', exist_ok=True)
            self.model.save('models/video_detector_model.h5')
            joblib.dump(self.scaler, 'models/video_scaler.pkl')
            self.save_trained_videos(trained_videos)
            
            # Move trained videos from datasetstrain to datasets folder
            self.move_trained_videos_to_dataset()
            
            training_type = "incremental" if (use_only_new and model_exists) else "full"
            self.update_progress(100, 'completed', f'{training_type.capitalize()} training completed! Accuracy: {accuracy:.4f}', accuracy=accuracy)
            
            return history, accuracy
            
        except Exception as e:
            self.update_progress(0, 'error', f'Training failed: {str(e)}')
            raise

def train_video_model():
    """Main training function"""
    trainer = VideoDetectorTrainer()
    return trainer.train_with_progress()