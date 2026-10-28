import numpy as np
import joblib
import re
import os
import sys
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from datetime import datetime
import string
from spellchecker import SpellChecker

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ProfessionalAIDetector:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        self.model_path = os.path.join(self.models_dir, 'ai_text_detector.pkl')
        
        # Text datasets directory
        self.datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
        self.human_text_dir = os.path.join(self.datasets_dir, 'human_text')
        self.ai_text_dir = os.path.join(self.datasets_dir, 'ai_text')
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.human_text_dir, exist_ok=True)
        os.makedirs(self.ai_text_dir, exist_ok=True)
        
        # AI patterns for detection
        self.ai_patterns = [
            r'\bas an ai\b', r'\blanguage model\b', r'\bchatgpt\b', r'\bgpt-\d',
            r'\bopenai\b', r'\btrained (?:on|by)\b', r'\bmy training\b',
            r'\bmy knowledge\b', r'\bi am an ai\b', r'\baccording to my\b'
        ]
        
        self.ai_structural = [
            r'\bfirst.*\bthen.*\bfinally\b', r'\bstart by.*\bthen.*\bfinally\b',
            r'\bin conclusion\b', r'\bto summarize\b', r'\bin summary\b',
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b'
        ]

        # Cache for loaded training texts
        self.human_texts_cache = None
        self.ai_texts_cache = None
        self.cache_loaded = False

        # Initialize spell checker
        self.spell = SpellChecker()
        self.spell.distance = 1  # Set edit distance for spell checking

    def _check_spelling_errors(self, text):
        """Check for spelling mistakes in text using spellchecker"""
        if not text or len(text.strip()) < 10:
            return 0, []  # Not enough text to analyze
        
        # Tokenize and clean words
        words = word_tokenize(text.lower())
        # Filter only alphabetic words with reasonable length (3+ characters)
        words = [word for word in words if word.isalpha() and len(word) > 2]
        
        if not words:
            return 0, []
        
        # Find misspelled words
        misspelled = list(self.spell.unknown(words))
        
        return len(misspelled), misspelled

    def _has_obvious_spelling_errors(self, text):
        """Check for obvious spelling mistakes that indicate human writing"""
        # Common human typing errors
        common_typos = [
            r'\bteh\b', r'\badn\b', r'\btahn\b', r'\bwiht\b', r'\brealy\b',
            r'\blittel\b', r'\bdefinately\b', r'\brecieve\b', r'\bseperate\b',
            r'\boccured\b', r'\buntill\b', r'\bbegining\b', r'\benviroment\b',
            r'\baccidentaly\b', r'\baccommodate\b', r'\bachieve\b', r'\barguement\b',
            r'\bcommited\b', r'\bconcious\b', r'\bdefinate\b', r'\bdilema\b',
            r'\bembarass\b', r'\bexistance\b', r'\bfirey\b', r'\bforiegn\b',
            r'\bgoverment\b', r'\bgrammer\b', r'\bgratefull\b', r'\bgrief\b',
            r'\bguage\b', r'\bhappend\b', r'\bharass\b', r'\bheros\b',
            r'\bhindrence\b', r'\bhumorous\b', r'\bignor\b', r'\bimediately\b',
            r'\bincidently\b', r'\bindependant\b', r'\bjudgement\b', r'\bknowlegde\b',
            r'\blabatory\b', r'\bliscense\b', r'\bmaintenence\b', r'\bmanuever\b',
            r'\bmischevious\b', r'\bneccessary\b', r'\bneice\b', r'\bnoticable\b',
            r'\boccassion\b', r'\boccurance\b', r'\bperseverence\b', r'\bpersonel\b',
            r'\bposession\b', r'\bprivelege\b', r'\bproffesor\b', r'\bpublically\b',
            r'\breciept\b', r'\brecomend\b', r'\brefered\b', r'\brelevent\b',
            r'\brememberance\b', r'\bresistence\b', r'\bresponsability\b', r'\brestaraunt\b',
            r'\brythm\b', r'\bsecratary\b', r'\bseige\b', r'\bsentance\b',
            r'\bseperate\b', r'\bsimiliar\b', r'\bsincerly\b', r'\bspeach\b',
            r'\bstrenght\b', r'\bsucess\b', r'\bsupercede\b', r'\bsupose\b',
            r'\bsurprize\b', r'\bthier\b', r'\bthroughly\b', r'\btommorow\b',
            r'\btonght\b', r'\btruely\b', r'\btyrany\b', r'\bunderate\b',
            r'\buntill\b', r'\bvaccuum\b', r'\bvegeterian\b', r'\bvillian\b',
            r'\bweird\b', r'\bwensday\b', r'\bwich\b', r'\bwritting\b',
            r'\byatch\b', r'\byoung\b'
        ]
        
        text_lower = text.lower()
        for typo in common_typos:
            if re.search(typo, text_lower):
                return True
        
        # Check for repeated characters (common in human typing)
        if re.search(r'(\w)\1{2,}', text):  # 3 or more repeated characters
            return True
        
        # Check for missing spaces
        if re.search(r'\b[a-z][A-Z]', text):  # camelCase without space
            return True
        
        # Check for obvious case errors
        if re.search(r'\b[a-z]+[A-Z]', text):  # Mixed case in middle of word
            return True
            
        return False

    def _check_grammar_issues(self, text):
        """Check for common grammar issues"""
        issues = []
        
        # Check for repeated words
        if re.search(r'\b(\w+)\s+\1\b', text, re.IGNORECASE):
            issues.append("repeated_words")
        
        # Check for missing apostrophes in common contractions
        contractions = [
            (r'\bdont\b', "don't"), (r'\bwont\b', "won't"), (r'\bcant\b', "can't"),
            (r'\bim\b', "I'm"), (r'\byoure\b', "you're"), (r'\btheyre\b', "they're"),
            (r'\bwere\b', "we're"), (r'\bshes\b', "she's"), (r'\bhes\b', "he's"),
            (r'\bits\b', "it's"), (r'\bthats\b', "that's"), (r'\bwhats\b', "what's")
        ]
        
        text_lower = text.lower()
        for wrong, correct in contractions:
            if re.search(wrong, text_lower):
                issues.append(f"missing_apostrophe_{correct}")
        
        return issues

    def _load_all_training_texts(self):
        """Load all training texts into cache"""
        if self.cache_loaded:
            return self.human_texts_cache, self.ai_texts_cache
            
        self.human_texts_cache = []
        self.ai_texts_cache = []
        
        try:
            # Load human texts
            for filename in os.listdir(self.human_text_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.human_text_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.human_texts_cache.extend(data.get('texts', []))
            
            # Load AI texts
            for filename in os.listdir(self.ai_text_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.ai_text_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.ai_texts_cache.extend(data.get('texts', []))
            
            self.cache_loaded = True
            print(f"📂 Cache loaded: {len(self.human_texts_cache)} human, {len(self.ai_texts_cache)} AI texts")
            
        except Exception as e:
            print(f"❌ Error loading training texts: {e}")
        
        return self.human_texts_cache, self.ai_texts_cache

    def _check_exact_match(self, text):
        """Check if text exactly matches any training example"""
        human_texts, ai_texts = self._load_all_training_texts()
        
        # Clean the input text for comparison
        clean_input = text.strip().lower()
        
        # Check for exact matches in human texts
        for human_text in human_texts:
            if human_text.strip().lower() == clean_input:
                return "Human Written"  # Exact match found in human dataset
        
        # Check for exact matches in AI texts
        for ai_text in ai_texts:
            if ai_text.strip().lower() == clean_input:
                return "AI Generated"  # Exact match found in AI dataset
        
        return None  # No exact match found

    def _is_perfect_text(self, text):
        """Check if text is 100% perfect (no spelling/grammar errors)"""
        spelling_errors, misspelled_words = self._check_spelling_errors(text)
        has_obvious_errors = self._has_obvious_spelling_errors(text)
        grammar_issues = self._check_grammar_issues(text)
        
        return spelling_errors == 0 and not has_obvious_errors and len(grammar_issues) == 0

    def save_training_data(self, human_texts, ai_texts):
        """Save training texts to datasets folder"""
        try:
            # Save human texts
            human_file = os.path.join(self.human_text_dir, f'human_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(human_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'count': len(human_texts),
                    'texts': human_texts
                }, f, indent=2, ensure_ascii=False)
            
            # Save AI texts
            ai_file = os.path.join(self.ai_text_dir, f'ai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(ai_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'count': len(ai_texts),
                    'texts': ai_texts
                }, f, indent=2, ensure_ascii=False)
            
            # Invalidate cache since we added new data
            self.cache_loaded = False
            
            print(f"💾 Training data saved: {len(human_texts)} human, {len(ai_texts)} AI texts")
            return True
            
        except Exception as e:
            print(f"❌ Error saving training data: {e}")
            return False

    def load_training_data(self):
        """Load all training data from datasets folder"""
        return self._load_all_training_texts()

    def get_dataset_stats(self):
        """Get statistics about the training data"""
        human_texts, ai_texts = self._load_all_training_texts()
        
        return {
            'human_count': len(human_texts),
            'ai_count': len(ai_texts),
            'total_count': len(human_texts) + len(ai_texts)
        }

    def extract_features(self, text):
        """Extract features from text for AI detection"""
        if not text or len(text.strip()) < 10:
            return np.zeros(20)
            
        features = []
        text_lower = text.lower()
        words = [w for w in word_tokenize(text_lower) if w.isalpha()]
        sentences = sent_tokenize(text)
        
        # Basic statistics
        features.append(len(text))
        features.append(len(words))
        features.append(len(sentences))
        
        if len(words) > 0:
            # Vocabulary features
            features.append(len(set(words)) / len(words))  # Lexical diversity
            features.append(np.mean([len(w) for w in words]))  # Avg word length
            
            # Sentence structure
            if len(sentences) > 1:
                sentence_lengths = [len(s.split()) for s in sentences]
                features.append(np.mean(sentence_lengths))
                features.append(np.std(sentence_lengths))
            else:
                features.extend([0, 0])
            
            # AI pattern features
            features.append(len(re.findall(r'\b(furthermore|moreover|however|therefore)\b', text_lower)))
            features.append(len(re.findall(r'\b(in conclusion|to summarize|in summary)\b', text_lower)))
            features.append(len(re.findall(r'\b(first|then|next|finally)\b', text_lower)))
            features.append(len(re.findall(r'\b(as an ai|language model|trained on)\b', text_lower)))
            
            # Technical terms
            tech_terms = ['algorithm', 'model', 'data', 'system', 'process']
            features.append(sum(1 for w in words if w in tech_terms) / len(words))
            
            # Human patterns
            features.append(len(re.findall(r'\b(i think|i feel|in my opinion)\b', text_lower)))
            features.append(len(re.findall(r'\b(maybe|perhaps|probably)\b', text_lower)))
            features.append(len(re.findall(r'\b(lol|haha|omg|wow)\b', text_lower)))
            
        else:
            features.extend([0] * 10)
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0)
            
        return np.array(features[:20])
    
    def train_model(self, human_texts, ai_texts, save_model=True):
        """Train the AI detection model"""
        print("🤖 Training AI Text Detector...")
        
        # Save training data to datasets folder
        self.save_training_data(human_texts, ai_texts)
        
        # Also load existing data for comprehensive training
        existing_human, existing_ai = self._load_all_training_texts()
        
        # Combine new and existing data
        all_human = existing_human + human_texts
        all_ai = existing_ai + ai_texts
        
        X = []
        y = []
        
        # Add human texts (label 0)
        for text in all_human:
            if text and len(text.strip()) > 20:
                features = self.extract_features(text)
                X.append(features)
                y.append(0)
        
        # Add AI texts (label 1)
        for text in all_ai:
            if text and len(text.strip()) > 20:
                features = self.extract_features(text)
                X.append(features)
                y.append(1)
        
        if len(X) < 10:
            print("❌ Not enough training data")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        print(f"✅ AI Text Detector trained successfully!")
        print(f"📊 Accuracy: {accuracy:.1%}")
        print(f"📝 Training samples: {len(X)} ({len(all_human)} human, {len(all_ai)} AI)")
        
        if save_model:
            self.save_model()
        
        return accuracy

    def train_model_with_progress(self, human_texts, ai_texts, progress_tracker, save_model=True):
        """Train the AI detection model with progress tracking"""
        print("🤖 Training AI Text Detector with progress tracking...")
        
        # Update progress - saving data
        progress_tracker.update(40, 'saving_data', 'Saving training data...')
        
        # Save training data to datasets folder
        self.save_training_data(human_texts, ai_texts)
        
        # Update progress - loading existing data
        progress_tracker.update(50, 'loading_data', 'Loading existing training data...')
        
        # Also load existing data for comprehensive training
        existing_human, existing_ai = self._load_all_training_texts()
        
        # Combine new and existing data
        all_human = existing_human + human_texts
        all_ai = existing_ai + ai_texts
        
        X = []
        y = []
        
        # Update progress - extracting features
        progress_tracker.update(60, 'extracting_features', 'Extracting features from texts...')
        
        total_texts = len(all_human) + len(all_ai)
        processed = 0
        
        # Add human texts (label 0)
        for text in all_human:
            if text and len(text.strip()) > 20:
                features = self.extract_features(text)
                X.append(features)
                y.append(0)
            
            processed += 1
            if processed % 1000 == 0:  # Update progress every 1000 texts
                progress = 60 + (processed / total_texts) * 20
                progress_tracker.update(progress, 'extracting_features', 
                                      f'Processing human texts... ({processed}/{total_texts})')
        
        # Add AI texts (label 1)
        for text in all_ai:
            if text and len(text.strip()) > 20:
                features = self.extract_features(text)
                X.append(features)
                y.append(1)
            
            processed += 1
            if processed % 1000 == 0:  # Update progress every 1000 texts
                progress = 60 + (processed / total_texts) * 20
                progress_tracker.update(progress, 'extracting_features', 
                                      f'Processing AI texts... ({processed}/{total_texts})')
        
        if len(X) < 10:
            progress_tracker.update(0, 'error', 'Not enough training data after processing')
            raise Exception("Not enough training data")
        
        X = np.array(X)
        y = np.array(y)
        
        # Update progress - splitting data
        progress_tracker.update(85, 'splitting_data', 'Splitting data for training...')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Update progress - training model
        progress_tracker.update(90, 'training_model', 'Training Random Forest model...')
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Update progress - evaluating
        progress_tracker.update(95, 'evaluating', 'Evaluating model accuracy...')
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        # Save model
        if save_model:
            self.save_model()
        
        return accuracy
    
    def predict(self, text):
        """Predict if text is AI or Human"""
        if not text or len(text.strip()) < 5:
            return "Human Written"
        
        print(f"🔍 Analyzing text: '{text[:50]}...'")
        
        # STEP 1: Check for exact match in training data (HIGHEST PRIORITY)
        exact_match = self._check_exact_match(text)
        if exact_match:
            print("🎯 Exact match found in training data")
            return exact_match
        
        # STEP 2: Check for spelling/grammar errors
        spelling_errors, misspelled_words = self._check_spelling_errors(text)
        has_obvious_errors = self._has_obvious_spelling_errors(text)
        grammar_issues = self._check_grammar_issues(text)
        
        print(f"🔍 Spell check: {spelling_errors} errors, words: {misspelled_words}")
        print(f"🔍 Grammar issues: {grammar_issues}")
        print(f"🔍 Obvious errors: {has_obvious_errors}")
        
        # If ANY spelling errors, grammar issues, or obvious errors found, classify as human
        if spelling_errors > 0 or has_obvious_errors or grammar_issues:
            print("🎯 Spelling/grammar errors detected - definitely Human Written")
            return "Human Written"
        
        # STEP 3: Use trained ML model
        if not self.is_trained:
            if not self.load_model():
                # If no model available, use rule-based detection
                return self._rule_based_detection(text)
        
        features = self.extract_features(text).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        
        if prediction == 1:
            print("🤖 ML Model prediction: AI Generated")
            return "AI Generated"
        else:
            print("👤 ML Model prediction: Human Written")
            return "Human Written"
    
    def _rule_based_detection(self, text):
        """Rule-based fallback when no trained model available"""
        text_lower = text.lower()
        
        # Strong AI indicators
        for pattern in self.ai_patterns:
            if re.search(pattern, text_lower):
                print("🤖 Rule-based: AI patterns detected")
                return "AI Generated"
        
        # Strong human indicators
        human_patterns = [
            r'\blol\b', r'\btbh\b', r'\bomg\b', r'\bhaha\b', r'\bwow\b',
            r'\bi think\b', r'\bi guess\b', r'\bmaybe\b', r'\bperhaps\b'
        ]
        
        for pattern in human_patterns:
            if re.search(pattern, text_lower):
                print("👤 Rule-based: Human patterns detected")
                return "Human Written"
        
        # If text is 100% perfect and no other indicators, assume AI
        if self._is_perfect_text(text):
            print("🤖 Perfect text with no human patterns - assuming AI Generated")
            return "AI Generated"
        
        print("👤 Default fallback: Human Written")
        return "Human Written"
    
    def save_model(self):
        """Save trained model"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'is_trained': self.is_trained
            }, self.model_path)
            print(f"💾 Model saved to: {self.model_path}")
            return True
        return False
    
    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_path):
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.is_trained = data['is_trained']
                print(f"📂 Model loaded from: {self.model_path}")
                return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
        
        return False

# Global detector instance
detector = ProfessionalAIDetector()

def detect_ai_human(text):
    """MAIN FUNCTION - Returns 'AI Generated' or 'Human Written'"""
    return detector.predict(text)

def train_detector(human_texts, ai_texts):
    """Train the detector with new data"""
    return detector.train_model(human_texts, ai_texts)

def train_detector_with_progress(human_texts, ai_texts, progress_tracker):
    """Train the detector with new data and progress tracking"""
    return detector.train_model_with_progress(human_texts, ai_texts, progress_tracker)

def initialize_detector():
    """Initialize the detector"""
    if detector.load_model():
        return True
    print("ℹ️ Using rule-based detection (train model for better accuracy)")
    return False

def get_text_dataset_stats():
    """Get text dataset statistics"""
    return detector.get_dataset_stats()

# Initialize when imported
initialize_detector()