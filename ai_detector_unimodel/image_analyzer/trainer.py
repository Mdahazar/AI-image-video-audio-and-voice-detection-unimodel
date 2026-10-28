import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
import hashlib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import shutil

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, progress_callback, total_epochs):
        super().__init__()
        self.progress_callback = progress_callback
        self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        progress = 60 + (epoch / self.total_epochs) * 35
        self.progress_callback(
            progress,
            'training',
            f'Training epoch {epoch + 1}/{self.total_epochs}',
            epoch=epoch + 1,
            total_epochs=self.total_epochs
        )
    
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy', 0) if logs else 0
        val_accuracy = logs.get('val_accuracy', 0) if logs else 0
        progress = 60 + ((epoch + 1) / self.total_epochs) * 35
        self.progress_callback(
            progress,
            'training',
            f'Epoch {epoch + 1}/{self.total_epochs} - Accuracy: {accuracy:.3f}, Val Accuracy: {val_accuracy:.3f}',
            accuracy=accuracy,
            epoch=epoch + 1,
            total_epochs=self.total_epochs
        )

class CustomDataGenerator(keras.utils.Sequence):
    """Custom data generator that works with file paths instead of directories"""
    def __init__(self, image_paths, labels, datagen, batch_size=16, img_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.datagen = datagen
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        valid_batch_labels = []
        
        for i, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize(self.img_size)
                img_array = np.array(img) / 255.0
                batch_images.append(img_array)
                valid_batch_labels.append(batch_labels[i])
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Skip corrupted images
                continue
        
        if len(batch_images) == 0:
            # Return a minimal batch if all images failed
            batch_images = [np.zeros((*self.img_size, 3))]
            valid_batch_labels = [0]
        
        return np.array(batch_images), np.array(valid_batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

class ImageDetectorTrainer:
    def __init__(self, dataset_path='datasets', model_save_path='models/ai_image_detector.h5', progress_callback=None):
        self.dataset_path = dataset_path
        self.datasetstrain_path = 'datasetstrain'  # New folder for training images
        self.model_save_path = model_save_path
        self.class_indices_path = 'models/class_indices.json'
        self.trained_images_file = 'models/trained_images.json'
        self.img_size = (224, 224)
        self.batch_size = 16
        self.epochs = 20
        self.model = None
        self.progress_callback = progress_callback
        
        # Create models and datasetstrain directories
        os.makedirs('models', exist_ok=True)
        os.makedirs(self.datasetstrain_path, exist_ok=True)
        os.makedirs(os.path.join(self.datasetstrain_path, 'real'), exist_ok=True)
        os.makedirs(os.path.join(self.datasetstrain_path, 'ai'), exist_ok=True)
    
    def update_progress(self, current, stage, message, accuracy=0.0, epoch=0, total_epochs=0):
        if self.progress_callback:
            self.progress_callback(current, stage, message, accuracy, epoch, total_epochs)
    
    def get_image_hash(self, image_path):
        """Generate a unique hash for the image file"""
        try:
            file_stats = os.stat(image_path)
            hash_input = f"{image_path}_{file_stats.st_size}_{file_stats.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return None
    
    def load_trained_images(self):
        """Load the list of already trained images"""
        try:
            if os.path.exists(self.trained_images_file):
                with open(self.trained_images_file, 'r') as f:
                    return set(json.load(f))
            return set()
        except:
            return set()
    
    def save_trained_images(self, trained_images):
        """Save the list of trained images"""
        try:
            os.makedirs('models', exist_ok=True)
            with open(self.trained_images_file, 'w') as f:
                json.dump(list(trained_images), f)
        except Exception as e:
            print(f"Error saving trained images list: {e}")
    
    def get_new_images_count(self):
        """Get count of new images that haven't been trained yet"""
        trained_images = self.load_trained_images()
        new_real_count = 0
        new_ai_count = 0
        
        # Check real images in datasetstrain
        real_path = os.path.join(self.datasetstrain_path, 'real')
        if os.path.exists(real_path):
            for img_file in os.listdir(real_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(real_path, img_file)
                    img_hash = self.get_image_hash(img_path)
                    if img_hash and img_hash not in trained_images:
                        new_real_count += 1
        
        # Check AI images in datasetstrain
        ai_path = os.path.join(self.datasetstrain_path, 'ai')
        if os.path.exists(ai_path):
            for img_file in os.listdir(ai_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(ai_path, img_file)
                    img_hash = self.get_image_hash(img_path)
                    if img_hash and img_hash not in trained_images:
                        new_ai_count += 1
        
        return new_real_count, new_ai_count
    
    def prepare_dataset_for_training(self, use_only_new=True):
        """Prepare dataset for training using datasetstrain folder"""
        trained_images = self.load_trained_images()
        
        # Track which images to use for training
        training_images = {'real': [], 'ai': []}
        all_trained_images = set(trained_images)
        total_images = 0
        
        # Process real images from datasetstrain
        real_path = os.path.join(self.datasetstrain_path, 'real')
        if os.path.exists(real_path):
            for img_file in os.listdir(real_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(real_path, img_file)
                    img_hash = self.get_image_hash(img_path)
                    
                    if not use_only_new or (img_hash and img_hash not in trained_images):
                        training_images['real'].append(img_path)
                        total_images += 1
                        
                        if img_hash:
                            all_trained_images.add(img_hash)
        
        # Process AI images from datasetstrain
        ai_path = os.path.join(self.datasetstrain_path, 'ai')
        if os.path.exists(ai_path):
            for img_file in os.listdir(ai_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(ai_path, img_file)
                    img_hash = self.get_image_hash(img_path)
                    
                    if not use_only_new or (img_hash and img_hash not in trained_images):
                        training_images['ai'].append(img_path)
                        total_images += 1
                        
                        if img_hash:
                            all_trained_images.add(img_hash)
        
        return training_images, total_images, all_trained_images
    
    def create_model(self):
        """Create CNN model for image classification - USING PRE-TRAINED MODEL"""
        # Use pre-trained MobileNetV2 for better feature extraction
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False  # Freeze base model layers
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def clean_dataset_files(self, training_images):
        """Remove corrupted images from the training set"""
        self.update_progress(5, 'cleaning', 'Checking for corrupted images...')
        
        corrupted_files = []
        
        # Check real images
        for img_path in training_images['real'][:]:  # Use slice copy to avoid modification during iteration
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    img.convert('RGB')
            except Exception as e:
                print(f"❌ Corrupted image found: {img_path} - {e}")
                corrupted_files.append(img_path)
                training_images['real'].remove(img_path)
        
        # Check AI images
        for img_path in training_images['ai'][:]:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    img.convert('RGB')
            except Exception as e:
                print(f"❌ Corrupted image found: {img_path} - {e}")
                corrupted_files.append(img_path)
                training_images['ai'].remove(img_path)
        
        if corrupted_files:
            print(f"✅ Removed {len(corrupted_files)} corrupted images from training set")
        else:
            print("✅ No corrupted images found")
            
        return len(corrupted_files)
    
    def create_data_generators(self, training_images):
        """Create data generators using file paths instead of directory structure"""
        self.update_progress(30, 'preprocessing', 'Creating data generators...')
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255, 
            validation_split=0.2,
        )
        
        # Prepare data for generators
        all_image_paths = []
        all_labels = []
        
        # Real images (label 1)
        for path in training_images['real']:
            all_image_paths.append(path)
            all_labels.append(1)  # Real images are class 1
        
        # AI images (label 0)
        for path in training_images['ai']:
            all_image_paths.append(path)
            all_labels.append(0)  # AI images are class 0
        
        if len(all_image_paths) == 0:
            raise ValueError("No valid images found for training!")
        
        # Split into train and validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        print(f"✅ Training samples: {len(train_paths)}")
        print(f"✅ Validation samples: {len(val_paths)}")
        
        # Create generators
        train_generator = CustomDataGenerator(train_paths, train_labels, train_datagen, 
                                            batch_size=self.batch_size, img_size=self.img_size)
        val_generator = CustomDataGenerator(val_paths, val_labels, val_datagen, 
                                          batch_size=self.batch_size, img_size=self.img_size, shuffle=False)
        
        # Save class indices (important for prediction)
        class_indices = {'ai': 0, 'real': 1}
        with open(self.class_indices_path, 'w') as f:
            json.dump(class_indices, f)
        
        print(f"✅ Class indices: {class_indices}")
        
        return train_generator, val_generator, class_indices
    
    def move_trained_images_to_dataset(self):
        """Move images from datasetstrain to datasets folder after successful training"""
        try:
            # Move real images
            real_train_path = os.path.join(self.datasetstrain_path, 'real')
            real_dataset_path = os.path.join(self.dataset_path, 'real')
            
            if os.path.exists(real_train_path):
                for img_file in os.listdir(real_train_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        src_path = os.path.join(real_train_path, img_file)
                        dst_path = os.path.join(real_dataset_path, img_file)
                        
                        # Ensure destination directory exists
                        os.makedirs(real_dataset_path, exist_ok=True)
                        
                        # Move file
                        shutil.move(src_path, dst_path)
                        print(f"✅ Moved {img_file} to datasets/real")
            
            # Move AI images
            ai_train_path = os.path.join(self.datasetstrain_path, 'ai')
            ai_dataset_path = os.path.join(self.dataset_path, 'ai')
            
            if os.path.exists(ai_train_path):
                for img_file in os.listdir(ai_train_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        src_path = os.path.join(ai_train_path, img_file)
                        dst_path = os.path.join(ai_dataset_path, img_file)
                        
                        # Ensure destination directory exists
                        os.makedirs(ai_dataset_path, exist_ok=True)
                        
                        # Move file
                        shutil.move(src_path, dst_path)
                        print(f"✅ Moved {img_file} to datasets/ai")
            
            print("✅ All trained images moved from datasetstrain to datasets folder")
            
        except Exception as e:
            print(f"❌ Error moving trained images: {e}")
    
    def train_with_progress(self, use_only_new=True):
        """Train the model with progress tracking - using datasetstrain folder"""
        self.update_progress(0, 'initializing', 'Starting model training...')
        
        try:
            # Check if model exists for incremental training
            model_exists = os.path.exists(self.model_save_path)
            
            if use_only_new and model_exists:
                # Check if there are new images to train in datasetstrain
                new_real_count, new_ai_count = self.get_new_images_count()
                total_new = new_real_count + new_ai_count
                
                if total_new == 0:
                    self.update_progress(100, 'completed', 'No new images to train! Model is up to date.', accuracy=1.0)
                    return None, 1.0
                
                self.update_progress(5, 'checking', f'Found {new_real_count} new real and {new_ai_count} new AI images in datasetstrain')
            
            # Prepare dataset (get file paths from datasetstrain)
            training_images, total_images, trained_images = self.prepare_dataset_for_training(use_only_new)
            
            if total_images == 0:
                self.update_progress(100, 'completed', 'No images found in datasetstrain for training!', accuracy=0.0)
                return None, 0.0
            
            training_type = "incremental" if (use_only_new and model_exists) else "full"
            self.update_progress(10, 'preparing', f'{training_type.capitalize()} training with {total_images} images from datasetstrain...')
            
            # Clean dataset by checking file integrity
            corrupted_count = self.clean_dataset_files(training_images)
            if corrupted_count > 0:
                self.update_progress(15, 'cleaning', f'Removed {corrupted_count} corrupted images')
            
            self.update_progress(20, 'loading_data', 'Setting up data generators...')
            
            # Create data generators using file paths
            train_generator, val_generator, class_indices = self.create_data_generators(training_images)
            
            # Check if we have enough data
            if len(train_generator.image_paths) == 0:
                raise ValueError("No training samples found after cleaning!")
            
            self.update_progress(40, 'creating_model', 'Creating neural network model...')
            
            # Create or load model
            if use_only_new and model_exists:
                # Load existing model for incremental training
                self.model = keras.models.load_model(self.model_save_path)
                print("🔄 Loaded existing model for incremental training")
                # Use fewer epochs for incremental training
                current_epochs = 8
            else:
                # Create new model for full training
                self.model = self.create_model()
                current_epochs = self.epochs
            
            print("🧠 Model architecture:")
            self.model.summary()
            
            # Calculate steps per epoch
            train_steps = max(1, len(train_generator))
            val_steps = max(1, len(val_generator))
            
            print(f"✅ Training steps per epoch: {train_steps}")
            print(f"✅ Validation steps per epoch: {val_steps}")
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
                ProgressCallback(self.update_progress, current_epochs)
            ]
            
            self.update_progress(50, 'starting_training', 'Starting model training...')
            
            # Train model with generators
            print(f"🎯 Training model ({training_type} training)...")
            history = self.model.fit(
                train_generator,
                steps_per_epoch=train_steps,
                epochs=current_epochs,
                validation_data=val_generator,
                validation_steps=val_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            self.update_progress(95, 'evaluating', 'Finalizing model training...')
            
            # Save model
            self.model.save(self.model_save_path)
            print(f"💾 Model saved to {self.model_save_path}")
            
            # Save trained images list
            self.save_trained_images(trained_images)
            print(f"💾 Saved {len(trained_images)} trained images to tracking file")
            
            # Move trained images from datasetstrain to datasets folder
            self.move_trained_images_to_dataset()
            
            # Plot training history
            self.plot_training_history(history)
            
            # Get final accuracy from history
            final_accuracy = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else history.history['accuracy'][-1]
            
            self.update_progress(100, 'completed', f'{training_type.capitalize()} training completed! Final accuracy: {final_accuracy:.4f}', accuracy=final_accuracy)
            
            return history, final_accuracy
            
        except Exception as e:
            self.update_progress(0, 'error', f'Training error: {str(e)}')
            print(f"❌ Training error: {e}")
            import traceback
            print(f"❌ Detailed traceback: {traceback.format_exc()}")
            raise
    
    def train(self):
        """Original train method for backward compatibility"""
        return self.train_with_progress()
    
    def plot_training_history(self, history):
        """Plot training history"""
        try:
            plt.figure(figsize=(12, 4))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            if 'accuracy' in history.history:
                plt.plot(history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            if 'loss' in history.history:
                plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('models/training_history.png')
            plt.close()
            
        except Exception as e:
            print(f"❌ Error plotting history: {e}")

def train_model():
    """Convenience function to train the model"""
    trainer = ImageDetectorTrainer()
    return trainer.train()

if __name__ == "__main__":
    train_model()