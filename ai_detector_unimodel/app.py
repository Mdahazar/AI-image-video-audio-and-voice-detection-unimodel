from flask import Flask, render_template, request, jsonify
from text_analyzer.detector import detect_ai_human
from text_analyzer.corrector import correct_text
from image_analyzer.detector import detect_ai_image
from image_analyzer.trainer import train_model
from video_analyzer.detector import detect_ai_video
from video_analyzer.trainer import train_video_model
from audio_analyzer.detector import detect_ai_audio
from audio_analyzer.trainer import train_audio_model
from text_analyzer.detector import train_detector, train_detector_with_progress
import os
import uuid
from datetime import datetime
import time
import json
import threading
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Increased to 500MB for large text files

class TrainingProgress:
    def __init__(self):
        self.current = 0
        self.stage = 'idle'
        self.message = ''
        self.accuracy = 0.0
        self.epoch = 0
        self.total_epochs = 0
        self.error = False
    
    def update(self, current, stage, message, accuracy=0.0, epoch=0, total_epochs=0):
        self.current = current
        self.stage = stage
        self.message = message
        self.accuracy = accuracy
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.error = False
    
    def to_dict(self):
        return {
            'current': self.current,
            'stage': self.stage,
            'message': self.message,
            'accuracy': self.accuracy,
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'error': self.error
        }

class TextTrainingProgress:
    def __init__(self):
        self.current = 0
        self.stage = 'idle'
        self.message = ''
        self.accuracy = 0.0
        self.error = False
    
    def update(self, current, stage, message, accuracy=0.0):
        self.current = current
        self.stage = stage
        self.message = message
        self.accuracy = accuracy
        self.error = False
    
    def to_dict(self):
        return {
            'current': self.current,
            'stage': self.stage,
            'message': self.message,
            'accuracy': self.accuracy,
            'error': self.error
        }

# Global training progress
training_progress = TrainingProgress()
video_training_progress = TrainingProgress()
audio_training_progress = TrainingProgress()
text_training_progress = TextTrainingProgress()

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create datasets folders if they don't exist
os.makedirs('datasets/real', exist_ok=True)
os.makedirs('datasets/ai', exist_ok=True)
os.makedirs('datasets/real_video', exist_ok=True)
os.makedirs('datasets/ai_video', exist_ok=True)
os.makedirs('datasets/real_audio', exist_ok=True)
os.makedirs('datasets/ai_audio', exist_ok=True)
os.makedirs('datasets/human_text', exist_ok=True)
os.makedirs('datasets/ai_text', exist_ok=True)

# Create datasetstrain folders for temporary training
os.makedirs('datasetstrain/real', exist_ok=True)
os.makedirs('datasetstrain/ai', exist_ok=True)
os.makedirs('datasetstrain/real_video', exist_ok=True)
os.makedirs('datasetstrain/ai_video', exist_ok=True)
os.makedirs('datasetstrain/real_audio', exist_ok=True)
os.makedirs('datasetstrain/ai_audio', exist_ok=True)

# ==================== VIDEO HASH DATABASE INITIALIZATION ====================
def initialize_video_detector():
    """Initialize video detector and build/update hash database if needed"""
    try:
        from video_analyzer.detector import VideoDetector
        
        detector = VideoDetector()
        
        # Check if database needs to be built or updated
        db_path = 'models/video_hash_database.pkl'
        
        if not os.path.exists(db_path):
            print("🏗️ Building video hash database for the first time...")
            detector.build_hash_database()
        else:
            # Check if datasets have been modified since last database build
            db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0
            
            # Check modification times of dataset folders
            real_path = 'datasets/real_video'
            ai_path = 'datasets/ai_video'
            
            dataset_modified = False
            
            # Check if any video files are newer than the database
            if os.path.exists(real_path):
                for filename in os.listdir(real_path):
                    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                        file_path = os.path.join(real_path, filename)
                        if os.path.getmtime(file_path) > db_mtime:
                            dataset_modified = True
                            break
            
            if not dataset_modified and os.path.exists(ai_path):
                for filename in os.listdir(ai_path):
                    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                        file_path = os.path.join(ai_path, filename)
                        if os.path.getmtime(file_path) > db_mtime:
                            dataset_modified = True
                            break
            
            if dataset_modified:
                print("🔄 Datasets have been updated, rebuilding video hash database...")
                detector.build_hash_database()
            else:
                print(f"✅ Video hash database loaded with {len(detector.video_hash_db)} entries")
        
        return detector
        
    except Exception as e:
        print(f"❌ Error initializing video detector: {e}")
        return None

# Initialize video detector on app startup
print("🎬 Initializing video detector...")
video_detector = initialize_video_detector()
# ==================== END VIDEO HASH DATABASE INITIALIZATION ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze_page():
    return render_template('analyze.html')

@app.route('/image-check')
def image_check_page():
    return render_template('image_check.html')

@app.route('/train-model')
def train_model_page():
    return render_template('train_model.html')

@app.route('/train-video-model')
def train_video_model_page():
    return render_template('train_video_model.html')

@app.route('/video-check')
def video_check_page():
    return render_template('video_check.html')

@app.route('/train-audio-model')
def train_audio_model_page():
    return render_template('train_audio_model.html')

@app.route('/audio-check')
def audio_check_page():
    return render_template('audio_check.html')

# NEW TEXT ANALYSIS ROUTES
@app.route('/train-text-model')
def train_text_model_page():
    return render_template('train_text_model.html')

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    try:
        text = request.json.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text to analyze'})
        
        if len(text) < 10:
            return jsonify({
                'result': 'Human Written',
                'details': ['Text too short for accurate analysis']
            })
        
        result = detect_ai_human(text)
        return jsonify({'result': result})
        
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'})

@app.route('/correct-text', methods=['POST'])
def correct_text_route():
    try:
        text = request.json.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text to correct'})
        
        result = correct_text(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Correction error: {str(e)}'})

@app.route('/train-text-model', methods=['POST'])
def train_text_model():
    if request.method == 'POST':
        try:
            # Reset progress
            global text_training_progress
            text_training_progress = TextTrainingProgress()
            
            # Check if request is too large
            if request.content_length and request.content_length > 400 * 1024 * 1024:
                return jsonify({'error': 'Training data too large. Please use smaller files or split your data.'})
            
            human_texts = []
            ai_texts = []
            
            # Update progress
            text_training_progress.update(10, 'reading_files', 'Reading uploaded files...')
            
            # Handle file uploads first (they might be large)
            if 'human_file_data' in request.files:
                human_file = request.files['human_file_data']
                if human_file and human_file.filename:
                    try:
                        human_file_text = human_file.read().decode('utf-8')
                        human_file_texts = [text.strip() for text in human_file_text.split('\n') if text.strip()]
                        human_texts.extend(human_file_texts)
                        print(f"📁 Loaded {len(human_file_texts)} human texts from file")
                    except Exception as e:
                        text_training_progress.update(0, 'error', f'Error reading human text file: {str(e)}')
                        return jsonify({'error': f'Error reading human text file: {str(e)}'})
            
            if 'ai_file_data' in request.files:
                ai_file = request.files['ai_file_data']
                if ai_file and ai_file.filename:
                    try:
                        ai_file_text = ai_file.read().decode('utf-8')
                        ai_file_texts = [text.strip() for text in ai_file_text.split('\n') if text.strip()]
                        ai_texts.extend(ai_file_texts)
                        print(f"📁 Loaded {len(ai_file_texts)} AI texts from file")
                    except Exception as e:
                        text_training_progress.update(0, 'error', f'Error reading AI text file: {str(e)}')
                        return jsonify({'error': f'Error reading AI text file: {str(e)}'})
            
            # Get training data from form (for smaller amounts of text)
            text_training_progress.update(20, 'processing_form', 'Processing form data...')
            
            form_human_texts = request.form.get('human_texts', '').split('\n')
            form_ai_texts = request.form.get('ai_texts', '').split('\n')
            
            # Clean and filter empty lines from form data
            form_human_texts = [text.strip() for text in form_human_texts if text.strip()]
            form_ai_texts = [text.strip() for text in form_ai_texts if text.strip()]
            
            # Combine file and form data
            human_texts.extend(form_human_texts)
            ai_texts.extend(form_ai_texts)
            
            # Remove duplicates
            human_texts = list(set(human_texts))
            ai_texts = list(set(ai_texts))
            
            print(f"📊 Total samples: {len(human_texts)} human, {len(ai_texts)} AI")
            
            if len(human_texts) < 5 or len(ai_texts) < 5:
                text_training_progress.update(0, 'error', 'Need at least 5 samples each of human and AI texts')
                return jsonify({'error': 'Need at least 5 samples each of human and AI texts'})
            
            # Limit the number of samples to prevent memory issues
            max_samples = 100000  # Limit to 100K samples per category
            if len(human_texts) > max_samples:
                human_texts = human_texts[:max_samples]
                print(f"⚠️ Limited human texts to {max_samples} samples")
            
            if len(ai_texts) > max_samples:
                ai_texts = ai_texts[:max_samples]
                print(f"⚠️ Limited AI texts to {max_samples} samples")
            
            # Start training in background thread
            def train_in_background():
                try:
                    text_training_progress.update(30, 'training', 'Starting model training...')
                    
                    # Train with progress updates
                    accuracy = train_detector_with_progress(human_texts, ai_texts, text_training_progress)
                    
                    text_training_progress.update(100, 'completed', 
                                                f'Training completed successfully! Accuracy: {accuracy:.1%}',
                                                accuracy=accuracy)
                    print(f"✅ Training completed with accuracy: {accuracy:.1%}")
                    
                except Exception as e:
                    text_training_progress.update(0, 'error', f'Training error: {str(e)}')
                    print(f"❌ Training error: {str(e)}")
            
            # Start training in background
            thread = threading.Thread(target=train_in_background)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Training started successfully!',
                'human_samples': len(human_texts),
                'ai_samples': len(ai_texts),
                'from_files': len(human_texts) + len(ai_texts)
            })
            
        except Exception as e:
            text_training_progress.update(0, 'error', f'Training error: {str(e)}')
            return jsonify({'error': f'Training error: {str(e)}'})

@app.route('/get-text-training-status')
def get_text_training_status():
    return jsonify(text_training_progress.to_dict())

# EXISTING IMAGE, VIDEO, AUDIO ROUTES
@app.route('/check-image', methods=['POST'])
def check_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'})
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'})
        
        # Generate unique filename
        file_ext = os.path.splitext(image_file.filename)[1]
        filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        image_file.save(filepath)
        
        # Analyze the image
        result = detect_ai_image(filepath)
        
        # Add file info to result
        result['filename'] = filename
        result['upload_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Image analysis error: {str(e)}'})

@app.route('/check-video', methods=['POST'])
def check_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'})
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video selected'})
        
        # Generate unique filename
        file_ext = os.path.splitext(video_file.filename)[1]
        filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        video_file.save(filepath)
        
        # Analyze the video
        result = detect_ai_video(filepath)
        
        # Add file info to result
        result['filename'] = filename
        result['upload_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Video analysis error: {str(e)}'})

@app.route('/check-audio', methods=['POST'])
def check_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio selected'})
        
        # Generate unique filename
        file_ext = os.path.splitext(audio_file.filename)[1]
        filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        audio_file.save(filepath)
        
        # Analyze the audio
        result = detect_ai_audio(filepath)
        
        # Add file info to result
        result['filename'] = filename
        result['upload_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Audio analysis error: {str(e)}'})

@app.route('/upload-training-image', methods=['POST'])
def upload_training_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'})
        
        image_file = request.files['image']
        category = request.form.get('category', 'real')
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'})
        
        # Validate category
        if category not in ['real', 'ai']:
            return jsonify({'error': 'Invalid category'})
        
        # Create category directory in datasetstrain if it doesn't exist
        category_path = os.path.join('datasetstrain', category)
        os.makedirs(category_path, exist_ok=True)
        
        # Generate unique filename
        file_ext = os.path.splitext(image_file.filename)[1]
        filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(category_path, filename)
        
        # Save the file to datasetstrain
        image_file.save(filepath)
        
        return jsonify({
            'status': 'success',
            'message': f'Image uploaded to {category} category in datasetstrain',
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'})

@app.route('/upload-training-video', methods=['POST'])
def upload_training_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'})
        
        video_file = request.files['video']
        category = request.form.get('category', 'real')
        
        if video_file.filename == '':
            return jsonify({'error': 'No video selected'})
        
        # Validate category
        if category not in ['real', 'ai']:
            return jsonify({'error': 'Invalid category'})
        
        # Validate file size (100MB)
        if video_file.content_length > 100 * 1024 * 1024:
            return jsonify({'error': 'File too large. Maximum size is 100MB.'})
        
        # Create category directory in datasetstrain if it doesn't exist
        category_folder = 'real_video' if category == 'real' else 'ai_video'
        category_path = os.path.join('datasetstrain', category_folder)
        os.makedirs(category_path, exist_ok=True)
        
        # Generate unique filename
        file_ext = os.path.splitext(video_file.filename)[1]
        filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(category_path, filename)
        
        # Save the file to datasetstrain
        video_file.save(filepath)
        
        return jsonify({
            'status': 'success',
            'message': f'Video uploaded to {category} category in datasetstrain',
            'filename': filename,
            'category': category
        })
        
    except Exception as e:
        return jsonify({'error': f'Video upload error: {str(e)}'})

@app.route('/upload-training-audio', methods=['POST'])
def upload_training_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'})
        
        audio_file = request.files['audio']
        category = request.form.get('category', 'real')
        
        if audio_file.filename == '':
            return jsonify({'error': 'No audio selected'})
        
        # Validate category
        if category not in ['real', 'ai']:
            return jsonify({'error': 'Invalid category'})
        
        # Validate file size (50MB)
        if audio_file.content_length > 50 * 1024 * 1024:
            return jsonify({'error': 'File too large. Maximum size is 50MB.'})
        
        # Create category directory in datasetstrain if it doesn't exist
        category_folder = 'real_audio' if category == 'real' else 'ai_audio'
        category_path = os.path.join('datasetstrain', category_folder)
        os.makedirs(category_path, exist_ok=True)
        
        # Generate unique filename
        file_ext = os.path.splitext(audio_file.filename)[1]
        filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(category_path, filename)
        
        # Save the file to datasetstrain (temporary training folder)
        audio_file.save(filepath)
        
        return jsonify({
            'status': 'success',
            'message': f'Audio uploaded to {category} category in datasetstrain',
            'filename': filename,
            'category': category
        })
        
    except Exception as e:
        return jsonify({'error': f'Audio upload error: {str(e)}'})

@app.route('/get-dataset-stats', methods=['GET'])
def get_dataset_stats():
    try:
        # Count images from the main datasets folder (not datasetstrain)
        real_path = os.path.join('datasets', 'real')
        ai_path = os.path.join('datasets', 'ai')
        
        real_count = 0
        ai_count = 0
        
        # Count real images from datasets folder
        if os.path.exists(real_path):
            real_count = len([f for f in os.listdir(real_path) 
                            if os.path.isfile(os.path.join(real_path, f)) and 
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        
        # Count AI images from datasets folder
        if os.path.exists(ai_path):
            ai_count = len([f for f in os.listdir(ai_path) 
                          if os.path.isfile(os.path.join(ai_path, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        
        return jsonify({
            'real_count': real_count,
            'ai_count': ai_count,
            'total_count': real_count + ai_count
        })
        
    except Exception as e:
        return jsonify({'error': f'Stats error: {str(e)}'})

@app.route('/get-datasetstrain-stats', methods=['GET'])
def get_datasetstrain_stats():
    """Get stats for datasetstrain folder (new images waiting to be trained)"""
    try:
        # Count images from the datasetstrain folder
        real_path = os.path.join('datasetstrain', 'real')
        ai_path = os.path.join('datasetstrain', 'ai')
        
        real_count = 0
        ai_count = 0
        
        # Count real images from datasetstrain folder
        if os.path.exists(real_path):
            real_count = len([f for f in os.listdir(real_path) 
                            if os.path.isfile(os.path.join(real_path, f)) and 
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        
        # Count AI images from datasetstrain folder
        if os.path.exists(ai_path):
            ai_count = len([f for f in os.listdir(ai_path) 
                          if os.path.isfile(os.path.join(ai_path, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        
        return jsonify({
            'real_count': real_count,
            'ai_count': ai_count,
            'total_count': real_count + ai_count
        })
        
    except Exception as e:
        return jsonify({'error': f'Datasetstrain stats error: {str(e)}'})

@app.route('/get-video-dataset-stats', methods=['GET'])
def get_video_dataset_stats():
    try:
        real_path = os.path.join('datasets', 'real_video')
        ai_path = os.path.join('datasets', 'ai_video')
        
        real_count = 0
        ai_count = 0
        
        # Count real videos
        if os.path.exists(real_path):
            real_count = len([f for f in os.listdir(real_path) 
                            if os.path.isfile(os.path.join(real_path, f)) and 
                            f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv'))])
        
        # Count AI videos
        if os.path.exists(ai_path):
            ai_count = len([f for f in os.listdir(ai_path) 
                          if os.path.isfile(os.path.join(ai_path, f)) and 
                          f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv'))])
        
        return jsonify({
            'real_count': real_count,
            'ai_count': ai_count,
            'total_count': real_count + ai_count
        })
        
    except Exception as e:
        return jsonify({'error': f'Video stats error: {str(e)}'})

@app.route('/get-audio-dataset-stats', methods=['GET'])
def get_audio_dataset_stats():
    try:
        # Count audio from the main datasets folder (not datasetstrain)
        real_path = os.path.join('datasets', 'real_audio')
        ai_path = os.path.join('datasets', 'ai_audio')
        
        real_count = 0
        ai_count = 0
        
        # Count real audio from datasets folder
        if os.path.exists(real_path):
            real_count = len([f for f in os.listdir(real_path) 
                            if os.path.isfile(os.path.join(real_path, f)) and 
                            f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac'))])
        
        # Count AI audio from datasets folder
        if os.path.exists(ai_path):
            ai_count = len([f for f in os.listdir(ai_path) 
                          if os.path.isfile(os.path.join(ai_path, f)) and 
                          f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac'))])
        
        return jsonify({
            'real_count': real_count,
            'ai_count': ai_count,
            'total_count': real_count + ai_count
        })
        
    except Exception as e:
        return jsonify({'error': f'Audio stats error: {str(e)}'})

@app.route('/get-audio-datasetstrain-stats', methods=['GET'])
def get_audio_datasetstrain_stats():
    """Get stats for datasetstrain folder (new audio waiting to be trained)"""
    try:
        # Count audio from the datasetstrain folder
        real_path = os.path.join('datasetstrain', 'real_audio')
        ai_path = os.path.join('datasetstrain', 'ai_audio')
        
        real_count = 0
        ai_count = 0
        
        # Count real audio from datasetstrain folder
        if os.path.exists(real_path):
            real_count = len([f for f in os.listdir(real_path) 
                            if os.path.isfile(os.path.join(real_path, f)) and 
                            f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac'))])
        
        # Count AI audio from datasetstrain folder
        if os.path.exists(ai_path):
            ai_count = len([f for f in os.listdir(ai_path) 
                          if os.path.isfile(os.path.join(ai_path, f)) and 
                          f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac'))])
        
        return jsonify({
            'real_count': real_count,
            'ai_count': ai_count,
            'total_count': real_count + ai_count
        })
        
    except Exception as e:
        return jsonify({'error': f'Audio datasetstrain stats error: {str(e)}'})

def move_trained_audio_to_datasets():
    """Move audio files from datasetstrain to datasets after successful training"""
    try:
        # Define source and destination paths
        source_real = os.path.join('datasetstrain', 'real_audio')
        source_ai = os.path.join('datasetstrain', 'ai_audio')
        dest_real = os.path.join('datasets', 'real_audio')
        dest_ai = os.path.join('datasets', 'ai_audio')
        
        # Create destination directories if they don't exist
        os.makedirs(dest_real, exist_ok=True)
        os.makedirs(dest_ai, exist_ok=True)
        
        moved_count = 0
        
        # Move real audio files
        if os.path.exists(source_real):
            for filename in os.listdir(source_real):
                if filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
                    src_path = os.path.join(source_real, filename)
                    dest_path = os.path.join(dest_real, filename)
                    
                    # Ensure unique filename in destination
                    counter = 1
                    name, ext = os.path.splitext(filename)
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(dest_real, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(src_path, dest_path)
                    moved_count += 1
        
        # Move AI audio files
        if os.path.exists(source_ai):
            for filename in os.listdir(source_ai):
                if filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
                    src_path = os.path.join(source_ai, filename)
                    dest_path = os.path.join(dest_ai, filename)
                    
                    # Ensure unique filename in destination
                    counter = 1
                    name, ext = os.path.splitext(filename)
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(dest_ai, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(src_path, dest_path)
                    moved_count += 1
        
        print(f"✅ Moved {moved_count} audio files from datasetstrain to datasets")
        return moved_count
        
    except Exception as e:
        print(f"❌ Error moving audio files: {e}")
        return 0

def run_training_in_thread(use_only_new=True):
    """Run training in background thread"""
    global training_progress
    try:
        print("🎯 Starting training in background thread...")
        
        # Import inside the thread to avoid circular imports
        from image_analyzer.trainer import ImageDetectorTrainer
        
        training_progress.update(5, 'initializing', 'Creating trainer instance...')
        trainer = ImageDetectorTrainer(progress_callback=training_progress.update)
        
        print("🚀 Beginning training process...")
        history, accuracy = trainer.train_with_progress(use_only_new=use_only_new)
        
        training_progress.update(100, 'completed', f'Training completed successfully! Final accuracy: {accuracy:.4f}', accuracy=accuracy)
        print(f"✅ Training completed with accuracy: {accuracy:.4f}")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Training error: {error_details}")
        training_progress.update(0, 'error', f'Training failed: {str(e)}')
        training_progress.error = True

def run_video_training_with_callback(use_only_new=True, callback=None):
    """Run video training in background thread with callback for hash database update"""
    global video_training_progress
    try:
        print("🎯 Starting video training in background thread...")
        
        # Import inside the thread to avoid circular imports
        from video_analyzer.trainer import VideoDetectorTrainer
        
        video_training_progress.update(5, 'initializing', 'Creating video trainer instance...')
        trainer = VideoDetectorTrainer(progress_callback=video_training_progress.update)
        
        print("🚀 Beginning video training process...")
        history, accuracy = trainer.train_with_progress(use_only_new=use_only_new)
        
        video_training_progress.update(100, 'completed', f'Video training completed successfully! Final accuracy: {accuracy:.4f}', accuracy=accuracy)
        print(f"✅ Video training completed with accuracy: {accuracy:.4f}")
        
        # Call the callback to update hash database after training
        if callback:
            callback()
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Video training error: {error_details}")
        video_training_progress.update(0, 'error', f'Video training failed: {str(e)}')
        video_training_progress.error = True

def run_audio_training_in_thread(use_only_new=True):
    """Run audio training in background thread"""
    global audio_training_progress
    try:
        print("🎯 Starting audio training in background thread...")
        
        # Import inside the thread to avoid circular imports
        from audio_analyzer.trainer import AudioDetectorTrainer
        
        audio_training_progress.update(5, 'initializing', 'Creating audio trainer instance...')
        trainer = AudioDetectorTrainer(progress_callback=audio_training_progress.update)
        
        print("🚀 Beginning audio training process...")
        history, accuracy = trainer.train_with_progress(use_only_new=use_only_new)
        
        # Move trained audio files to datasets folder after successful training
        if accuracy > 0:  # Only move if training was successful
            moved_count = move_trained_audio_to_datasets()
            if moved_count > 0:
                print(f"📁 Moved {moved_count} audio files to datasets folder")
        
        audio_training_progress.update(100, 'completed', f'Audio training completed successfully! Final accuracy: {accuracy:.4f}', accuracy=accuracy)
        print(f"✅ Audio training completed with accuracy: {accuracy:.4f}")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Audio training error: {error_details}")
        audio_training_progress.update(0, 'error', f'Audio training failed: {str(e)}')
        audio_training_progress.error = True

@app.route('/start-training', methods=['POST'])
def start_training():
    global training_progress
    try:
        # Get training type from request
        use_only_new = request.json.get('use_only_new', True) if request.json else True
        
        # Reset progress
        training_progress = TrainingProgress()
        
        if use_only_new:
            training_progress.update(0, 'starting', 'Starting incremental training with new images only...')
        else:
            training_progress.update(0, 'starting', 'Starting full training with all images...')
        
        # Start training in background thread with parameter
        thread = threading.Thread(target=run_training_in_thread, kwargs={'use_only_new': use_only_new})
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Training started in background',
            'training_type': 'incremental' if use_only_new else 'full'
        })
        
    except Exception as e:
        training_progress.update(0, 'error', f'Failed to start training: {str(e)}')
        training_progress.error = True
        return jsonify({
            'status': 'error',
            'message': f'Training failed to start: {str(e)}'
        })

@app.route('/start-video-training', methods=['POST'])
def start_video_training():
    global video_training_progress
    try:
        # Get training type from request
        use_only_new = request.json.get('use_only_new', True) if request.json else True
        
        # Reset progress
        video_training_progress = TrainingProgress()
        
        if use_only_new:
            video_training_progress.update(0, 'starting', 'Starting incremental training with new videos only...')
        else:
            video_training_progress.update(0, 'starting', 'Starting full training with all videos...')
        
        # Callback to update hash database after training
        def training_complete_callback():
            print("🔄 Training completed, updating video hash database...")
            try:
                from video_analyzer.detector import VideoDetector
                detector = VideoDetector()
                detector.build_hash_database()
                print("✅ Video hash database updated with new training data")
            except Exception as e:
                print(f"❌ Error updating video hash database: {e}")
        
        # Start training in background thread with callback
        thread = threading.Thread(target=run_video_training_with_callback, 
                                kwargs={'use_only_new': use_only_new, 
                                       'callback': training_complete_callback})
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Video training started in background',
            'training_type': 'incremental' if use_only_new else 'full'
        })
        
    except Exception as e:
        video_training_progress.update(0, 'error', f'Failed to start video training: {str(e)}')
        video_training_progress.error = True
        return jsonify({
            'status': 'error',
            'message': f'Video training failed to start: {str(e)}'
        })

@app.route('/start-audio-training', methods=['POST'])
def start_audio_training():
    global audio_training_progress
    try:
        # Get training type from request
        use_only_new = request.json.get('use_only_new', True) if request.json else True
        
        # Reset progress
        audio_training_progress = TrainingProgress()
        
        if use_only_new:
            audio_training_progress.update(0, 'starting', 'Starting incremental training with new audio only...')
        else:
            audio_training_progress.update(0, 'starting', 'Starting full training with all audio...')
        
        # Start training in background thread with parameter
        thread = threading.Thread(target=run_audio_training_in_thread, kwargs={'use_only_new': use_only_new})
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Audio training started in background',
            'training_type': 'incremental' if use_only_new else 'full'
        })
        
    except Exception as e:
        audio_training_progress.update(0, 'error', f'Failed to start audio training: {str(e)}')
        audio_training_progress.error = True
        return jsonify({
            'status': 'error',
            'message': f'Audio training failed to start: {str(e)}'
        })

@app.route('/get-training-status')
def get_training_status():
    return jsonify(training_progress.to_dict())

@app.route('/get-video-training-status')
def get_video_training_status():
    return jsonify(video_training_progress.to_dict())

@app.route('/get-audio-training-status')
def get_audio_training_status():
    return jsonify(audio_training_progress.to_dict())

@app.route('/clear-dataset', methods=['POST'])
def clear_dataset():
    """Optional: Clear the dataset if needed"""
    try:
        real_path = os.path.join('datasets', 'real')
        ai_path = os.path.join('datasets', 'ai')
        
        # Remove all files in real directory
        if os.path.exists(real_path):
            for filename in os.listdir(real_path):
                file_path = os.path.join(real_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        # Remove all files in AI directory
        if os.path.exists(ai_path):
            for filename in os.listdir(ai_path):
                file_path = os.path.join(ai_path, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        return jsonify({
            'status': 'success',
            'message': 'Dataset cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error clearing dataset: {str(e)}'
        })

@app.route('/clear-datasetstrain', methods=['POST'])
def clear_datasetstrain():
    """Clear the datasetstrain folder"""
    try:
        real_path = os.path.join('datasetstrain', 'real')
        ai_path = os.path.join('datasetstrain', 'ai')
        real_audio_path = os.path.join('datasetstrain', 'real_audio')
        ai_audio_path = os.path.join('datasetstrain', 'ai_audio')
        real_video_path = os.path.join('datasetstrain', 'real_video')
        ai_video_path = os.path.join('datasetstrain', 'ai_video')
        
        # Remove all files in all datasetstrain directories
        for path in [real_path, ai_path, real_audio_path, ai_audio_path, real_video_path, ai_video_path]:
            if os.path.exists(path):
                for filename in os.listdir(path):
                    file_path = os.path.join(path, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
        
        return jsonify({
            'status': 'success',
            'message': 'Datasetstrain cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error clearing datasetstrain: {str(e)}'
        })

# Add error handler for large files
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large. Maximum size for videos is 100MB, for images is 16MB, for audio is 50MB, for text files is 400MB.'
    }), 413

@app.route('/get-text-dataset-stats', methods=['GET'])
def get_text_dataset_stats():
    try:
        from text_analyzer.detector import get_text_dataset_stats
        stats = get_text_dataset_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Text stats error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)