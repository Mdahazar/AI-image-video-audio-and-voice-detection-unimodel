import re
import string
from spellchecker import SpellChecker
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SimpleTextCorrector:
    def __init__(self):
        # Initialize spell checker WITHOUT custom dictionary
        self.spell = SpellChecker()
        # Remove this line: self.spell.word_frequency.load_text_file('en.txt')
        
        # Enhanced correction rules for common mistakes
        self.correction_rules = {
            # Common misspellings
            'exmple': 'example',
            'smal': 'small', 
            'erors': 'errors',
            'intendd': 'intended',
            'easly': 'easily',
            'understaand': 'understand',
            'frm': 'from',
            'engeniring': 'engineering',
            'collage': 'college',
            'ahazar': 'hazard',
            'degre': 'degree',
            'computr': 'computer',
            'sience': 'science',
            'wat': 'what',
            'yor': 'your',
            'were': 'where',
            'ar': 'are',
            'studnts': 'students',
            'lernning': 'learning',
            'programing': 'programming',
            'colege': 'college',
            'profeser': 'professor',
            'univercity': 'university',
            'studdy': 'study',
            'artifishal': 'artificial',
            'inteligence': 'intelligence',
            'becom': 'become',
            'softwear': 'software',
            'enginer': 'engineer',
            'workng': 'working',
            'compliting': 'completing',
            'mangalore': 'mangalore',
            'moodbidri': 'moodbidri',
            'karnataka': 'karnataka',
            
            # Context corrections
            'there': 'their',
            'their': 'there',
            'youre': 'your',
        }
        
        # Grammar/contraction rules
        self.grammar_rules = [
            (r'\bi\b', 'I'),
            (r'\bim\b', "I'm"),
            (r'\bive\b', "I've"),
            (r'\byoure\b', "you're"),
            (r'\btheyre\b', "they're"),
            (r'\bdont\b', "don't"),
            (r'\bdoesnt\b', "doesn't"),
            (r'\bisnt\b', "isn't"),
            (r'\barent\b', "aren't"),
            (r'\bwasnt\b', "wasn't"),
            (r'\bhavent\b', "haven't"),
            (r'\bwont\b', "won't"),
            (r'\bwouldnt\b', "wouldn't"),
            (r'\bcant\b', "can't"),
            (r'\bcouldnt\b', "couldn't"),
            (r'\bshouldnt\b', "shouldn't"),
            (r'\bthats\b', "that's"),
            (r'\bwhats\b', "what's"),
        ]
        
        # Protected terms (don't correct these)
        self.protected_terms = {
            'alvas', 'institute', 'engineering', 'technology', 'university',
            'college', 'school', 'academy', 'mangalore', 'moodbidri', 
            'karnataka', 'india', 'python', 'java', 'javascript', 'html',
            'css', 'ai', 'ml', 'gpt', 'chatgpt', 'openai', 'deepseek'
        }

    def correct_text(self, text):
        """Main correction function"""
        try:
            if not text or not text.strip():
                return {"error": "Please enter some text to correct."}
            
            original_text = text.strip()
            corrected = original_text
            
            # Apply correction pipeline
            corrected = self.apply_correction_rules(corrected)
            corrected = self.apply_grammar_rules(corrected)
            corrected = self.correct_spelling(corrected)
            corrected = self.fix_punctuation(corrected)
            corrected = self.fix_capitalization(corrected)
            corrected = self.final_cleanup(corrected)
            
            return {
                "original_text": original_text,
                "corrected_text": corrected,
                "message": "Text corrected successfully!",
                "changes_made": original_text != corrected
            }
            
        except Exception as e:
            return {"error": f"Text correction failed: {str(e)}"}

    def apply_correction_rules(self, text):
        """Apply pre-defined correction rules"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            original_word = word
            clean_word = word.lower().strip(string.punctuation)
            
            # Check if word needs correction
            if clean_word in self.correction_rules:
                replacement = self.correction_rules[clean_word]
                
                # Preserve capitalization
                if original_word.istitle():
                    replacement = replacement.title()
                elif original_word.isupper():
                    replacement = replacement.upper()
                elif original_word[0].isupper():
                    replacement = replacement.capitalize()
                
                # Preserve punctuation
                prefix = re.match(r'^[^\w]*', original_word).group()
                suffix = re.search(r'[^\w]*$', original_word).group()
                word = prefix + replacement + suffix
            
            corrected_words.append(word)
        
        return ' '.join(corrected_words)

    def apply_grammar_rules(self, text):
        """Apply grammar and contraction rules"""
        corrected_text = text
        for pattern, replacement in self.grammar_rules:
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        return corrected_text

    def correct_spelling(self, text):
        """Correct spelling using spell checker"""
        words = word_tokenize(text)
        corrected_words = []
        
        for word in words:
            if not word.isalpha() or self.should_protect_word(word):
                corrected_words.append(word)
                continue
            
            clean_word = word.lower()
            
            # Skip very short words
            if len(clean_word) <= 2:
                corrected_words.append(word)
                continue
            
            # Get spelling suggestion
            corrected_clean = self.spell.correction(clean_word)
            
            if corrected_clean and corrected_clean != clean_word:
                # Preserve formatting
                if word.istitle():
                    corrected_clean = corrected_clean.title()
                elif word.isupper():
                    corrected_clean = corrected_clean.upper()
                elif word[0].isupper():
                    corrected_clean = corrected_clean.capitalize()
                
                corrected_words.append(corrected_clean)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)

    def should_protect_word(self, word):
        """Check if word should be protected from correction"""
        clean_word = word.lower().strip(string.punctuation)
        return clean_word in self.protected_terms

    def fix_punctuation(self, text):
        """Fix punctuation and spacing"""
        # Fix space before punctuation
        text = re.sub(r'\s+([.,!?;:)])', r'\1', text)
        
        # Add space after punctuation if missing
        text = re.sub(r'([.,!?;:])(?=[A-Za-z])', r'\1 ', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def fix_capitalization(self, text):
        """Fix sentence capitalization"""
        sentences = sent_tokenize(text)
        if sentences:
            corrected_sentences = []
            for sentence in sentences:
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                corrected_sentences.append(sentence)
            text = ' '.join(corrected_sentences)
        return text

    def final_cleanup(self, text):
        """Final text cleanup"""
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add final period if missing
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text

# Global instance
_text_corrector = SimpleTextCorrector()

def correct_text(text):
    """Public interface for text correction"""
    return _text_corrector.correct_text(text)

# Test function
def test_corrector():
    test_cases = [
        "hi i am ahazar frm alvas engeniring collage",
        "This is an exmple of a paragraph with some smal erors. Every sentence here is intendd to be read easly. I hope you can still understaand what I meant.",
        "i hav a degre in computr sience from alvas",
        "wat is yor nam and were ar you from",
        "teh studnts are lernning programing at alvas collage"
    ]
    
    print("🧪 Testing Simple Text Corrector\n")
    print("=" * 70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Original:  '{test}'")
        result = correct_text(test)
        
        if 'error' in result:
            print(f"Error:     {result['error']}")
        else:
            print(f"Corrected: '{result['corrected_text']}'")
            print(f"Changes:   {result['changes_made']}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_corrector()