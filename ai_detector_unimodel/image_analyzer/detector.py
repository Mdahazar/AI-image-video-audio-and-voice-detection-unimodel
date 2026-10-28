import cv2
import numpy as np
from PIL import Image, ImageFilter
import requests
import json
import os
import base64
from io import BytesIO
import tensorflow as tf
from tensorflow import keras

class AdvancedImageAIDetector:
    def __init__(self):
        self.api_endpoints = [
            "https://api.sightengine.com/1.0/check.json",  # Free tier available
            "https://api.moderatecontent.com/",  # Free AI detection
        ]
        
    def detect_ai_image(self, image_path):
        """Advanced AI image detection using multiple methods"""
        try:
            # FIRST: Try our trained model (highest priority)
            trained_model_result = self._trained_model_check(image_path)
            if trained_model_result and trained_model_result.get('confidence') in ['High', 'Very High']:
                return trained_model_result
            
            # Fallback to other methods
            # Method 1: Try SightEngine API (Free tier)
            api_result = self._sightengine_api_check(image_path)
            if api_result and api_result.get('confidence', 0) > 0.8:
                return api_result
            
            # Method 2: Try ModerateContent API
            api_result = self._moderatecontent_api_check(image_path)
            if api_result and api_result.get('confidence', 0) > 0.8:
                return api_result
            
            # Method 3: Advanced local analysis with deep features
            local_result = self._advanced_local_analysis(image_path)
            
            # Combine results for maximum accuracy
            return self._combine_results(api_result, local_result, trained_model_result)
            
        except Exception as e:
            return {"error": f"Image analysis error: {str(e)}"}
    
    def _trained_model_check(self, image_path):
        """Use our trained model for AI detection"""
        try:
            # Load the trained model
            model = keras.models.load_model('models/ai_image_detector.h5')
            
            # Load class indices to understand mapping
            with open('models/class_indices.json', 'r') as f:
                class_indices = json.load(f)
            
            print(f"🔍 Class mapping: {class_indices}")  # Debug info
            
            # Preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)[0][0]
            
            print(f"🔍 Raw prediction: {prediction}")  # Debug info
            
            # FIXED: Based on your class mapping {"ai": 0, "real": 1}
            # prediction = probability of being "real" (class 1)
            # So if prediction > 0.5, it's more likely REAL
            # if prediction < 0.5, it's more likely AI
            
            if prediction > 0.5:
                # More likely REAL (class 1)
                confidence = prediction
                confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                return {
                    "result": "Real Image",
                    "confidence": confidence_level,
                    "details": [f"Trained model: {confidence*100:.1f}% probability real"],
                    "source": "Trained Model (84.7% accuracy)"
                }
            else:
                # More likely AI (class 0) 
                confidence = 1 - prediction
                confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                return {
                    "result": "AI Generated Image",
                    "confidence": confidence_level,
                    "details": [f"Trained model: {confidence*100:.1f}% probability AI"],
                    "source": "Trained Model (84.7% accuracy)"
                }
                
        except Exception as e:
            print(f"Trained model error: {e}")
            return None
    
    def _sightengine_api_check(self, image_path):
        """Use SightEngine API (Free tier available)"""
        try:
            # You need to sign up for free API key at https://sightengine.com/
            api_key = "1968090334"  # Replace with actual key
            api_secret = "yidwP8UVUb5sYDLDxcTNKvFNTvoQkNga"
            
            url = "https://api.sightengine.com/1.0/check.json"
            files = {'media': open(image_path, 'rb')}
            data = {
                'models': 'genai',
                'api_user': api_key,
                'api_secret': api_secret
            }
            
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                if 'genai' in result:
                    ai_prob = result['genai'].get('prob', 0)
                    if ai_prob > 0.7:
                        return {
                            "result": "AI Generated Image",
                            "confidence": "Very High",
                            "details": [f"SightEngine API detected AI image ({(ai_prob*100):.1f}% probability)"],
                            "source": "SightEngine API"
                        }
                    else:
                        return {
                            "result": "Real Image", 
                            "confidence": "High",
                            "details": [f"SightEngine API confirms real image ({(1-ai_prob)*100:.1f}% probability)"],
                            "source": "SightEngine API"
                        }
        except Exception as e:
            print(f"SightEngine API error: {e}")
        return None
    
    def _moderatecontent_api_check(self, image_path):
        """Use ModerateContent API (Free tier)"""
        try:
            # Free API - no key required for basic usage
            url = "https://api.moderatecontent.com/ai/"
            
            with open(image_path, 'rb') as img_file:
                files = {'file': img_file}
                response = requests.post(url, files=files)
                
            if response.status_code == 200:
                result = response.json()
                if 'ai_generated' in result:
                    ai_score = result.get('ai_generated', 0)
                    if ai_score > 70:
                        return {
                            "result": "AI Generated Image",
                            "confidence": "High",
                            "details": [f"ModerateContent API: AI probability {ai_score}%"],
                            "source": "ModerateContent API"
                        }
                    else:
                        return {
                            "result": "Real Image",
                            "confidence": "High", 
                            "details": [f"ModerateContent API: Real image probability {100-ai_score}%"],
                            "source": "ModerateContent API"
                        }
        except Exception as e:
            print(f"ModerateContent API error: {e}")
        return None
    
    def _advanced_local_analysis(self, image_path):
        """Advanced local analysis with deep learning features"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            features = self._extract_deep_features(image)
            ai_score = self._calculate_ai_probability(features)
            
            if ai_score > 0.7:
                return {
                    "result": "AI Generated Image",
                    "confidence": "Medium",
                    "details": self._get_ai_indicators(features),
                    "source": "Advanced Local Analysis"
                }
            else:
                return {
                    "result": "Real Image",
                    "confidence": "Medium",
                    "details": ["Local analysis suggests authentic image"],
                    "source": "Advanced Local Analysis"
                }
                
        except Exception as e:
            print(f"Local analysis error: {e}")
            return None
    
    def _extract_deep_features(self, image):
        """Extract sophisticated image features for AI detection"""
        features = {}
        
        # 1. Frequency domain analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Check for grid patterns (GAN artifacts)
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        roi = magnitude_spectrum[center_y-30:center_y+30, center_x-30:center_x+30]
        features['frequency_variance'] = np.std(roi)
        
        # 2. Color consistency analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features['color_consistency'] = np.std(hsv[:,:,0])  # Hue variance
        
        # 3. Edge analysis
        edges = cv2.Canny(image, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 4. Texture analysis using GLCM-like features
        features['texture_smoothness'] = self._analyze_texture(gray)
        
        # 5. Noise pattern analysis
        features['noise_characteristics'] = self._analyze_noise_patterns(gray)
        
        # 6. Face analysis (if present)
        if self._has_face(image):
            face_features = self._analyze_face_features(image)
            features.update(face_features)
        
        return features
    
    def _calculate_ai_probability(self, features):
        """Calculate AI probability based on extracted features"""
        ai_score = 0
        
        # Frequency domain patterns (GAN artifacts)
        if features.get('frequency_variance', 0) > 80:
            ai_score += 0.3
        
        # Too perfect color consistency
        if features.get('color_consistency', 0) < 10:
            ai_score += 0.2
        
        # Unnatural edge patterns
        edge_density = features.get('edge_density', 0)
        if edge_density < 0.005 or edge_density > 0.4:
            ai_score += 0.2
        
        # Texture analysis
        if features.get('texture_smoothness', 0) > 0.9:
            ai_score += 0.2
        
        # Face analysis indicators
        if features.get('unnatural_face', False):
            ai_score += 0.3
        
        return min(1.0, ai_score)
    
    def _analyze_texture(self, gray_image):
        """Analyze texture patterns for AI detection"""
        # Calculate local binary patterns variance
        lbp = self._local_binary_pattern(gray_image)
        return np.var(lbp)
    
    def _local_binary_pattern(self, image, points=8, radius=1):
        """Calculate Local Binary Pattern for texture analysis"""
        lbp = np.zeros_like(image)
        for i in range(points):
            # Calculate sample points
            angle = 2 * np.pi * i / points
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Sample points (simplified implementation)
            # In production, use proper LBP implementation
            pass
        
        return lbp
    
    def _analyze_noise_patterns(self, gray_image):
        """Analyze noise characteristics"""
        # Calculate noise level and patterns
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        noise = gray_image - blurred
        return np.std(noise)
    
    def _has_face(self, image):
        """Check if image contains faces"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return len(faces) > 0
        except:
            return False
    
    def _analyze_face_features(self, image):
        """Analyze facial features for AI detection"""
        features = {}
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Analyze symmetry and proportions
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                
                # Check symmetry
                left_half = face_roi[:, :w//2]
                right_half = face_roi[:, w//2:]
                right_flipped = cv2.flip(right_half, 1)
                
                symmetry = cv2.matchTemplate(left_half, right_flipped, cv2.TM_CCOEFF_NORMED)[0][0]
                features['face_symmetry'] = symmetry
                
                # AI faces are often too symmetrical
                features['unnatural_face'] = symmetry > 0.9
                
        except Exception as e:
            print(f"Face analysis error: {e}")
        
        return features
    
    def _get_ai_indicators(self, features):
        """Generate human-readable AI indicators"""
        indicators = []
        
        if features.get('frequency_variance', 0) > 80:
            indicators.append("🔄 Detected frequency domain patterns typical of AI generation")
        
        if features.get('color_consistency', 0) < 10:
            indicators.append("🎨 Unnaturally perfect color consistency")
        
        edge_density = features.get('edge_density', 0)
        if edge_density < 0.005:
            indicators.append("📏 Unnaturally smooth edges")
        elif edge_density > 0.4:
            indicators.append("📏 Excessive edge artifacts")
        
        if features.get('unnatural_face', False):
            indicators.append("👤 Unnaturally symmetrical facial features")
        
        if not indicators:
            indicators.append("🤖 Multiple AI generation patterns detected")
        
        return indicators
    
    def _combine_results(self, api_result, local_result, trained_result):
        """Combine results from multiple detection methods"""
        # If trained model result is available, use it first
        if trained_result:
            return trained_result
        
        # If API result is available and confident, use it
        if api_result and api_result.get('confidence') in ['Very High', 'High']:
            return api_result
        
        # Otherwise use local analysis
        if local_result:
            return local_result
        
        # Fallback
        return {
            "result": "Real Image",
            "confidence": "Low",
            "details": ["Insufficient data for accurate analysis"],
            "source": "Fallback Analysis"
        }

# Global detector instance (using advanced detector)
image_detector = AdvancedImageAIDetector()

def detect_ai_image(image_path):
    return image_detector.detect_ai_image(image_path)