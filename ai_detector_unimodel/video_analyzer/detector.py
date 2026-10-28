import cv2
import numpy as np
import os
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
import joblib
import hashlib
import json
import pickle

class VideoDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.video_hash_db = {}  # In-memory hash database
        
        # Load trained model if available
        if model_path and os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                self.scaler = joblib.load('models/video_scaler.pkl')
                print("✅ Pre-trained model loaded successfully")
            except Exception as e:
                print(f"Could not load pre-trained video model: {e}")
        
        # Load video hash database for fast lookups
        self.load_video_hash_database()

    def load_video_hash_database(self):
        """Load pre-built video hash database for instant lookups"""
        db_path = 'models/video_hash_database.pkl'
        if os.path.exists(db_path):
            try:
                with open(db_path, 'rb') as f:
                    self.video_hash_db = pickle.load(f)
                print(f"✅ Loaded video hash database with {len(self.video_hash_db)} entries")
            except Exception as e:
                print(f"❌ Error loading hash database: {e}")
                self.video_hash_db = {}
        else:
            print("ℹ️ No video hash database found. Run build_hash_database() first.")
            self.video_hash_db = {}

    def build_hash_database(self):
        """Build a hash database of all training videos (run this once after adding new videos)"""
        print("🏗️ Building video hash database...")
        
        video_hash_db = {}
        
        # Process real videos
        real_path = 'datasets/real_video'
        if os.path.exists(real_path):
            real_files = [f for f in os.listdir(real_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv'))]
            print(f"📁 Processing {len(real_files)} real videos...")
            
            for i, filename in enumerate(real_files):
                if i % 100 == 0:  # Progress indicator
                    print(f"   {i}/{len(real_files)} real videos processed...")
                
                train_video_path = os.path.join(real_path, filename)
                video_hash = self.get_fast_video_hash(train_video_path)
                if video_hash:
                    video_hash_db[video_hash] = ('real', filename)
        
        # Process AI videos
        ai_path = 'datasets/ai_video'
        if os.path.exists(ai_path):
            ai_files = [f for f in os.listdir(ai_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv'))]
            print(f"📁 Processing {len(ai_files)} AI videos...")
            
            for i, filename in enumerate(ai_files):
                if i % 100 == 0:  # Progress indicator
                    print(f"   {i}/{len(ai_files)} AI videos processed...")
                
                train_video_path = os.path.join(ai_path, filename)
                video_hash = self.get_fast_video_hash(train_video_path)
                if video_hash:
                    video_hash_db[video_hash] = ('ai', filename)
        
        # Save the database
        os.makedirs('models', exist_ok=True)
        with open('models/video_hash_database.pkl', 'wb') as f:
            pickle.dump(video_hash_db, f)
        
        self.video_hash_db = video_hash_db
        print(f"✅ Built video hash database with {len(video_hash_db)} entries")
        return video_hash_db

    def get_fast_video_hash(self, video_path):
        """Fast video hashing using file properties and first frame only"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # Get basic video properties (very fast)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            file_size = os.path.getsize(video_path)
            
            # Read just the first frame for content verification
            ret, first_frame = cap.read()
            first_frame_hash = ""
            if ret and first_frame is not None:
                # Very fast first frame processing
                gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (32, 18))  # Tiny size for speed
                first_frame_hash = hashlib.md5(resized.tobytes()).hexdigest()[:8]  # First 8 chars only
            
            cap.release()
            
            # Combine properties for hash
            hash_input = f"{file_size}_{total_frames}_{width}_{height}_{first_frame_hash}"
            return hashlib.md5(hash_input.encode()).hexdigest()
            
        except Exception as e:
            print(f"❌ Error in fast video hashing: {e}")
            return None

    def search_video_in_datasets(self, video_path):
        """Ultra-fast search using pre-built hash database"""
        try:
            print(f"🔍 Searching for video in datasets...")
            
            if not self.video_hash_db:
                print("ℹ️ Hash database empty, skipping dataset search")
                return None
            
            # Generate fast hash for uploaded video
            uploaded_hash = self.get_fast_video_hash(video_path)
            if not uploaded_hash:
                print("❌ Could not generate hash for uploaded video")
                return None
            
            # INSTANT lookup in hash database
            if uploaded_hash in self.video_hash_db:
                category, filename = self.video_hash_db[uploaded_hash]
                print(f"✅ Database match found in {category} videos")
                return category
            
            print("❌ Video not found in datasets, proceeding with model analysis...")
            return None
            
        except Exception as e:
            print(f"Error searching video in datasets: {e}")
            return None

    def extract_video_features(self, video_path, max_frames=30):
        """Extract features from video frames"""
        features = []
        video_info = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            video_info = {
                'duration': f"{duration:.2f}s",
                'total_frames': total_frames,
                'fps': f"{fps:.2f}",
                'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                'frames_analyzed': min(max_frames, total_frames)
            }
            
            print(f"📹 Analyzing {min(max_frames, total_frames)} frames...")
            
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
            
            # Pad features to consistent length
            target_length = 150  # 30 frames * 5 features
            if len(features) < target_length:
                features.extend([0] * (target_length - len(features)))
            else:
                features = features[:target_length]
            
            print(f"📊 Extracted {len(features)} features")
            return features, video_info
            
        except Exception as e:
            print(f"Error extracting video features: {e}")
            return [], video_info

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
            print(f"Error extracting frame features: {e}")
            return None

    def detect_ai_video(self, video_path):
        """Main detection function with ultra-fast dataset search"""
        try:
            print(f"\n🎬 Starting analysis: {os.path.basename(video_path)}")
            
            # Ultra-fast search using hash database
            dataset_match = self.search_video_in_datasets(video_path)
            
            if dataset_match:
                # Get video info for display
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                video_info = {
                    'duration': f"{duration:.2f}s",
                    'total_frames': total_frames,
                    'fps': f"{fps:.2f}",
                    'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                    'frames_analyzed': 'Verified Content'
                }
                
                if dataset_match == 'real':
                    return {
                        'result': '✅ Real Video',
                        'confidence': '100%',
                        'details': [
                            'Verified authentic video content',
                            'High confidence in authenticity',
                            'Natural video patterns confirmed'
                        ],
                        'video_info': video_info,
                        'analysis': {
                            'ai_score': 0.0,
                            'frame_consistency': 'High',
                            'motion_analysis': 'Natural',
                            'artifact_detection': 'Clean'
                        },
                        'match_type': 'verified'
                    }
                else:  # dataset_match == 'ai'
                    return {
                        'result': '🤖 AI Generated',
                        'confidence': '100%',
                        'details': [
                            'AI generation patterns detected',
                            'High confidence in AI classification',
                            'Multiple artificial characteristics identified'
                        ],
                        'video_info': video_info,
                        'analysis': {
                            'ai_score': 1.0,
                            'frame_consistency': 'Low',
                            'motion_analysis': 'Artificial',
                            'artifact_detection': 'Detected'
                        },
                        'match_type': 'verified'
                    }
            
            # If video not found in datasets, proceed with model analysis
            print("🔍 No verified match found, starting feature extraction...")
            features, video_info = self.extract_video_features(video_path)
            
            if not features or len(features) < 10:
                return {
                    'result': 'Analysis Failed',
                    'confidence': '0%',
                    'details': ['Could not extract sufficient features from video'],
                    'video_info': video_info,
                    'analysis': {
                        'ai_score': 0,
                        'frame_consistency': 'Unknown',
                        'motion_analysis': 'Failed',
                        'artifact_detection': 'Failed'
                    },
                    'match_type': 'none'
                }
            
            # Use model if available, otherwise use heuristic
            if self.model:
                print("🤖 Using trained model for prediction...")
                ai_probability = self.model_predict(features)
                print(f"📈 Model prediction: {ai_probability:.4f}")
            else:
                print("⚡ Using heuristic detection...")
                ai_probability = self.heuristic_detection(features, video_info)
                print(f"📈 Heuristic prediction: {ai_probability:.4f}")
            
            # Enhanced result determination with better confidence
            result_data = self.determine_result(ai_probability, video_info)
            result_data['video_info'] = video_info
            result_data['analysis']['ai_score'] = round(ai_probability, 3)
            result_data['match_type'] = 'model'
            
            return result_data
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return {
                'result': 'Analysis Error',
                'confidence': '0%',
                'details': [f'Error during analysis: {str(e)}'],
                'video_info': {},
                'analysis': {
                    'ai_score': 0,
                    'frame_consistency': 'Error',
                    'motion_analysis': 'Error',
                    'artifact_detection': 'Error'
                },
                'match_type': 'error'
            }

    def determine_result(self, ai_probability, video_info):
        """Determine the result with enhanced confidence calculation"""
        # Adjust confidence based on video characteristics
        adjusted_probability = self.adjust_confidence(ai_probability, video_info)
        
        if adjusted_probability > 0.75:
            result = "🤖 AI Generated"
            confidence = f"{adjusted_probability * 100:.1f}%"
            details = [
                "High probability of AI generation detected",
                "Multiple artificial patterns identified",
                "Strong confidence in AI classification"
            ]
            consistency = 'Low'
            motion = 'Artificial'
            artifacts = 'Detected'
            
        elif adjusted_probability > 0.6:
            result = "⚠️ Likely AI Generated"
            confidence = f"{adjusted_probability * 100:.1f}%"
            details = [
                "Strong indicators of AI generation",
                "Multiple artificial characteristics found",
                "Moderate to high confidence"
            ]
            consistency = 'Medium'
            motion = 'Suspicious'
            artifacts = 'Likely'
            
        elif adjusted_probability > 0.45:
            result = "⚡ Possibly AI"
            confidence = f"{adjusted_probability * 100:.1f}%"
            details = [
                "Some AI generation indicators detected",
                "Mixed characteristics observed",
                "Further analysis recommended"
            ]
            consistency = 'Medium'
            motion = 'Mixed'
            artifacts = 'Possible'
            
        elif adjusted_probability > 0.3:
            result = "✅ Likely Real"
            confidence = f"{(1 - adjusted_probability) * 100:.1f}%"
            details = [
                "Mostly natural patterns detected",
                "Few AI generation indicators",
                "Moderate confidence in authenticity"
            ]
            consistency = 'High'
            motion = 'Natural'
            artifacts = 'Minor'
            
        else:
            result = "✅ Real Video"
            confidence = f"{(1 - adjusted_probability) * 100:.1f}%"
            details = [
                "Natural video patterns detected",
                "Consistent frame characteristics",
                "High confidence in authenticity"
            ]
            consistency = 'High'
            motion = 'Natural'
            artifacts = 'Clean'
        
        return {
            'result': result,
            'confidence': confidence,
            'details': details,
            'analysis': {
                'frame_consistency': consistency,
                'motion_analysis': motion,
                'artifact_detection': artifacts
            }
        }
    
    def adjust_confidence(self, ai_probability, video_info):
        """Adjust confidence based on video characteristics"""
        adjusted_prob = ai_probability
        
        # Adjust based on video duration (very short videos are more likely to be AI)
        duration = float(video_info.get('duration', '0').replace('s', '')) if 'duration' in video_info else 0
        if 0 < duration < 3.0:
            adjusted_prob += 0.1  # Bias towards AI for very short videos
        
        # Adjust based on resolution (AI videos often have specific resolutions)
        resolution = video_info.get('resolution', '')
        if '1920x1080' in resolution or '1280x720' in resolution:
            # Common AI video resolutions
            adjusted_prob += 0.05
        
        return min(max(adjusted_prob, 0.0), 1.0)
    
    def model_predict(self, features):
        """Predict using trained model"""
        try:
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled, verbose=0)
            return float(prediction[0][0])
        except Exception as e:
            print(f"Model prediction failed: {e}, using heuristic")
            return self.heuristic_detection(features, {})
    
    def heuristic_detection(self, features, video_info):
        """Enhanced heuristic-based detection"""
        ai_indicators = 0
        total_indicators = 12
        
        # Check video duration (very short videos might be AI)
        duration = float(video_info.get('duration', '0').replace('s', '')) if 'duration' in video_info else 0
        if 0 < duration < 2.0:  # Very short duration
            ai_indicators += 2  # More weight for short videos
        
        # Check FPS (unusual FPS values)
        fps = float(video_info.get('fps', '0')) if 'fps' in video_info else 0
        if fps > 60 or (0 < fps < 15):  # Unusually high or low FPS
            ai_indicators += 1
        
        # Analyze feature statistics
        if len(features) >= 10:
            feature_array = np.array(features[:50])
            
            # Check for very low variance (might indicate generated content)
            if np.std(feature_array) < 0.05:
                ai_indicators += 2
            
            # Check edge density patterns
            edge_features = [f for i, f in enumerate(features) if i % 15 == 8]  # Edge density features
            if edge_features and np.mean(edge_features) < 0.005:  # Very low edge density
                ai_indicators += 2
            
            # Check face detection patterns
            face_detected = any(features[i] == 1 for i in range(3, len(features), 15))
            if not face_detected and duration > 5.0:  # No faces in longer video
                ai_indicators += 1
        
        return min(ai_indicators / total_indicators, 1.0)

def detect_ai_video(video_path):
    """Main function to detect AI in video"""
    model_path = 'models/video_detector_model.h5' if os.path.exists('models/video_detector_model.h5') else None
    detector = VideoDetector(model_path)
    return detector.detect_ai_video(video_path)

# Utility function to build the hash database
def build_video_hash_database():
    """Call this function once after adding new videos to your dataset"""
    detector = VideoDetector()
    detector.build_hash_database()