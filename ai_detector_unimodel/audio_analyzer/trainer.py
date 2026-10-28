import os
import numpy as np
import librosa
import json
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shutil

class AudioDetectorTrainer:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.model = None
        self.scaler = StandardScaler()
        self.trained_audio_file = 'models/trained_audio.json'
        
    def update_progress(self, current, stage, message, accuracy=0.0, epoch=0, total_epochs=0):
        if self.progress_callback:
            self.progress_callback(current, stage, message, accuracy, epoch, total_epochs)
    
    def get_audio_hash(self, audio_path):
        """Generate a unique hash for the audio file"""
        try:
            file_stats = os.stat(audio_path)
            hash_input = f"{audio_path}_{file_stats.st_size}_{file_stats.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return None
    
    def load_trained_audio(self):
        """Load the list of already trained audio files"""
        try:
            if os.path.exists(self.trained_audio_file):
                with open(self.trained_audio_file, 'r') as f:
                    return set(json.load(f))
            return set()
        except:
            return set()
    
    def save_trained_audio(self, trained_audio):
        """Save the list of trained audio files"""
        try:
            os.makedirs('models', exist_ok=True)
            with open(self.trained_audio_file, 'w') as f:
                json.dump(list(trained_audio), f)
        except Exception as e:
            print(f"Error saving trained audio list: {e}")
    
    def get_new_audio_count(self):
        """Get count of new audio files that haven't been trained yet"""
        trained_audio = self.load_trained_audio()
        new_real_count = 0
        new_ai_count = 0
        
        # Check real audio from datasetstrain (new files)
        real_path = 'datasetstrain/real_audio'
        if os.path.exists(real_path):
            for audio_file in os.listdir(real_path):
                if audio_file.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
                    audio_path = os.path.join(real_path, audio_file)
                    audio_hash = self.get_audio_hash(audio_path)
                    if audio_hash and audio_hash not in trained_audio:
                        new_real_count += 1
        
        # Check AI audio from datasetstrain (new files)
        ai_path = 'datasetstrain/ai_audio'
        if os.path.exists(ai_path):
            for audio_file in os.listdir(ai_path):
                if audio_file.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
                    audio_path = os.path.join(ai_path, audio_file)
                    audio_hash = self.get_audio_hash(audio_path)
                    if audio_hash and audio_hash not in trained_audio:
                        new_ai_count += 1
        
        return new_real_count, new_ai_count
    
    def load_audio_dataset(self, use_only_new=True):
        """Load and preprocess audio dataset - only new audio if requested"""
        X = []
        y = []
        audio_count = 0
        all_trained_audio = set()
        trained_audio = self.load_trained_audio()
        
        if use_only_new:
            # Load only new audio files from datasetstrain
            new_real_audio = []
            new_ai_audio = []
            
            # Check real audio from datasetstrain
            real_path = 'datasetstrain/real_audio'
            if os.path.exists(real_path):
                for audio_file in os.listdir(real_path):
                    if audio_file.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
                        audio_path = os.path.join(real_path, audio_file)
                        audio_hash = self.get_audio_hash(audio_path)
                        if audio_hash and audio_hash not in trained_audio:
                            new_real_audio.append(audio_file)
            
            # Check AI audio from datasetstrain
            ai_path = 'datasetstrain/ai_audio'
            if os.path.exists(ai_path):
                for audio_file in os.listdir(ai_path):
                    if audio_file.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
                        audio_path = os.path.join(ai_path, audio_file)
                        audio_hash = self.get_audio_hash(audio_path)
                        if audio_hash and audio_hash not in trained_audio:
                            new_ai_audio.append(audio_file)
            
            self.update_progress(10, 'loading', f'Found {len(new_real_audio)} new real and {len(new_ai_audio)} new AI audio files')
            
            # Load new real audio from datasetstrain
            for i, audio_file in enumerate(new_real_audio):
                audio_path = os.path.join('datasetstrain/real_audio', audio_file)
                features, _ = self.extract_audio_features(audio_path)
                if features and len(features) > 0:
                    X.append(features)
                    y.append(0)  # 0 for real
                    audio_count += 1
                    # Add to trained audio set
                    audio_hash = self.get_audio_hash(audio_path)
                    if audio_hash:
                        all_trained_audio.add(audio_hash)
                
                progress = 10 + (i / len(new_real_audio)) * 20 if new_real_audio else 10
                self.update_progress(progress, 'loading', f'Processing new real audio... ({i+1}/{len(new_real_audio)})')
            
            # Load new AI audio from datasetstrain
            for i, audio_file in enumerate(new_ai_audio):
                audio_path = os.path.join('datasetstrain/ai_audio', audio_file)
                features, _ = self.extract_audio_features(audio_path)
                if features and len(features) > 0:
                    X.append(features)
                    y.append(1)  # 1 for AI
                    audio_count += 1
                    # Add to trained audio set
                    audio_hash = self.get_audio_hash(audio_path)
                    if audio_hash:
                        all_trained_audio.add(audio_hash)
                
                progress = 30 + (i / len(new_ai_audio)) * 20 if new_ai_audio else 30
                self.update_progress(progress, 'loading', f'Processing new AI audio... ({i+1}/{len(new_ai_audio)})')
                
        else:
            # Load all audio files from datasets folder (full retrain)
            self.update_progress(10, 'loading', 'Loading all audio files from datasets...')
            real_path = 'datasets/real_audio'
            ai_path = 'datasets/ai_audio'
            
            real_audio = [f for f in os.listdir(real_path) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac'))] if os.path.exists(real_path) else []
            ai_audio = [f for f in os.listdir(ai_path) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac'))] if os.path.exists(ai_path) else []
            
            # Load real audio from datasets
            for i, audio_file in enumerate(real_audio):
                audio_path = os.path.join(real_path, audio_file)
                features, _ = self.extract_audio_features(audio_path)
                if features and len(features) > 0:
                    X.append(features)
                    y.append(0)
                    audio_count += 1
                    # Add to trained audio set
                    audio_hash = self.get_audio_hash(audio_path)
                    if audio_hash:
                        all_trained_audio.add(audio_hash)
                
                progress = 10 + (i / len(real_audio)) * 20 if real_audio else 10
                self.update_progress(progress, 'loading', f'Processing real audio... ({i+1}/{len(real_audio)})')
            
            # Load AI audio from datasets
            for i, audio_file in enumerate(ai_audio):
                audio_path = os.path.join(ai_path, audio_file)
                features, _ = self.extract_audio_features(audio_path)
                if features and len(features) > 0:
                    X.append(features)
                    y.append(1)
                    audio_count += 1
                    # Add to trained audio set
                    audio_hash = self.get_audio_hash(audio_path)
                    if audio_hash:
                        all_trained_audio.add(audio_hash)
                
                progress = 30 + (i / len(ai_audio)) * 20 if ai_audio else 30
                self.update_progress(progress, 'loading', f'Processing AI audio... ({i+1}/{len(ai_audio)})')
        
        return np.array(X), np.array(y), audio_count, all_trained_audio
    
    def extract_audio_features(self, audio_path):
        """Extract features from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=22050)  # Resample to 22.05 kHz
            duration = librosa.get_duration(y=y, sr=sr)
            
            features = []
            
            # Basic audio features
            features.extend([
                np.mean(y), np.std(y), np.max(y), np.min(y)  # Amplitude statistics
            ])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids)
            ])
            
            # MFCC features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            features.extend(mfcc_means[:8])  # Use first 8 MFCC coefficients
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.extend([np.mean(rolloff), np.std(rolloff)])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean[:6])  # Use first 6 chroma features
            
            # Root Mean Square Energy
            rms = librosa.feature.rms(y=y)
            features.extend([np.mean(rms), np.std(rms)])
            
            audio_info = {
                'duration': f"{duration:.2f}s",
                'format': os.path.splitext(audio_path)[1].upper().replace('.', ''),
                'sample_rate': f"{sr} Hz",
                'channels': 'Mono' if len(y.shape) == 1 else f"Stereo ({y.shape[0]} channels)"
            }
            
            return features, audio_info
            
        except Exception as e:
            print(f"Error extracting audio features from {audio_path}: {e}")
            return None, {}
    
    def create_model(self, input_shape):
        """Create a neural network model for audio detection"""
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
    
    def train_with_progress(self, use_only_new=True):
        """Train the model with progress updates - only new audio if requested"""
        try:
            # Check if model exists for incremental training
            model_exists = os.path.exists('models/audio_detector_model.h5')
            
            if use_only_new and model_exists:
                # Check if there are new audio files to train
                new_real_count, new_ai_count = self.get_new_audio_count()
                total_new = new_real_count + new_ai_count
                
                if total_new == 0:
                    self.update_progress(100, 'completed', 'No new audio files to train! Model is up to date.', accuracy=1.0)
                    return None, 1.0
                
                self.update_progress(5, 'checking', f'Found {new_real_count} new real and {new_ai_count} new AI audio files')
            
            # Load dataset
            self.update_progress(10, 'loading', 'Loading audio dataset...')
            X, y, audio_count, trained_audio = self.load_audio_dataset(use_only_new=use_only_new)
            
            if audio_count < 5:
                if use_only_new and model_exists:
                    self.update_progress(100, 'completed', 'No new audio files to train! Model is up to date.', accuracy=1.0)
                    return None, 1.0
                else:
                    raise Exception(f"Insufficient data. Only {audio_count} audio files found. Need at least 5 files per category.")
            
            training_type = "incremental" if (use_only_new and model_exists) else "full"
            self.update_progress(50, 'preprocessing', f'{training_type.capitalize()} training with {audio_count} audio files...')
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features
            if use_only_new and model_exists:
                # Load existing scaler for incremental training
                self.scaler = joblib.load('models/audio_scaler.pkl')
                X_train = self.scaler.transform(X_train)
                X_test = self.scaler.transform(X_test)
            else:
                # Fit new scaler for full training
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
            
            self.update_progress(60, 'training', 'Creating/setting up model...')
            
            # Create or load model
            if use_only_new and model_exists:
                # Load existing model for incremental training
                self.model = keras.models.load_model('models/audio_detector_model.h5')
                print("🔄 Loaded existing model for incremental training")
                current_epochs = 8  # Fewer epochs for incremental training
            else:
                # Create new model for full training
                self.model = self.create_model(X_train.shape[1])
                current_epochs = 15
            
            # Custom callback for progress updates
            class ProgressCallback(keras.callbacks.Callback):
                def __init__(self, trainer, total_epochs):
                    self.trainer = trainer
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    progress = 60 + ((epoch + 1) / self.total_epochs) * 35
                    self.trainer.update_progress(
                        progress, 'training',
                        f'Training epoch {epoch+1}/{self.total_epochs}',
                        logs.get('accuracy', 0) if logs else 0,
                        epoch + 1,
                        self.total_epochs
                    )
            
            self.update_progress(65, 'training', 'Starting model training...')
            
            history = self.model.fit(
                X_train, y_train,
                epochs=current_epochs,
                batch_size=8,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[ProgressCallback(self, current_epochs)]
            )
            
            # Evaluate model
            self.update_progress(95, 'validation', 'Validating model...')
            y_pred = (self.model.predict(X_test) > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model and update trained audio list
            os.makedirs('models', exist_ok=True)
            self.model.save('models/audio_detector_model.h5')
            joblib.dump(self.scaler, 'models/audio_scaler.pkl')
            self.save_trained_audio(trained_audio)
            
            self.update_progress(100, 'completed', f'{training_type.capitalize()} training completed! Accuracy: {accuracy:.4f}', accuracy=accuracy)
            
            return history, accuracy
            
        except Exception as e:
            self.update_progress(0, 'error', f'Training failed: {str(e)}')
            raise

def train_audio_model():
    """Main training function"""
    trainer = AudioDetectorTrainer()
    return trainer.train_with_progress()