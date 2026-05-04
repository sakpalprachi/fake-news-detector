import os
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fake_news_detector.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the Fake News Detector."""
    
    # Model settings
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # File paths
    MODEL_PATH = "model.pkl"
    VECTORIZER_PATH = "vectorizer.pkl"
    CONFUSION_MATRIX_PATH = "confusion_matrix.png"
    
    # UI settings
    MAX_TEXT_LENGTH = 10000
    CONFIDENCE_THRESHOLD = 70

class PredictionHistory:
    """Class to manage prediction history."""
    
    def __init__(self, history_file="prediction_history.json"):
        self.history_file = history_file
        self.history = self.load_history()
    
    def load_history(self) -> List[Dict[str, Any]]:
        """Load prediction history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                return []
        return []
    
    def save_history(self):
        """Save prediction history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def add_prediction(self, text: str, prediction: str, confidence: float, probabilities: Dict[str, float]):
        """Add a new prediction to history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        }
        
        self.history.append(entry)
        
        # Keep only last 1000 predictions
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        self.save_history()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        if not self.history:
            return {}
        
        total = len(self.history)
        fake_count = sum(1 for h in self.history if h["prediction"] == "Fake")
        real_count = total - fake_count
        avg_confidence = sum(h["confidence"] for h in self.history) / total
        
        return {
            "total_predictions": total,
            "fake_predictions": fake_count,
            "real_predictions": real_count,
            "average_confidence": avg_confidence,
            "fake_percentage": (fake_count / total) * 100,
            "real_percentage": (real_count / total) * 100
        }

def validate_text_input(text: str, max_length: int = Config.MAX_TEXT_LENGTH) -> tuple[bool, str]:
    """Validate text input for prediction."""
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    if len(text) > max_length:
        return False, f"Text too long. Maximum length is {max_length} characters"
    
    # Check for minimum content
    if len(text.strip()) < 10:
        return False, "Text too short. Please provide at least 10 characters"
    
    return True, "Valid input"

def format_confidence(confidence: float) -> str:
    """Format confidence score for display."""
    if confidence >= 90:
        return f"🟢 {confidence:.1f}% (Very High)"
    elif confidence >= 75:
        return f"🟡 {confidence:.1f}% (High)"
    elif confidence >= 60:
        return f"🟠 {confidence:.1f}% (Medium)"
    else:
        return f"🔴 {confidence:.1f}% (Low)"

def get_prediction_emoji(prediction: str) -> str:
    """Get emoji for prediction."""
    return "❌" if prediction == "Fake" else "✅"

def create_sample_datasets():
    """Create sample CSV datasets for testing."""
    import pandas as pd
    
    # Fake news sample
    fake_data = {
        'title': [
            "Aliens Found Living on Earth",
            "Miracle Cure Discovered in Kitchen",
            "Celebrity Secretly Time Traveler",
            "Government Hiding Truth About Moon"
        ],
        'text': [
            "Shocking revelations from anonymous sources claim that aliens have been living among us for decades, disguised as ordinary citizens.",
            "Scientists discover that common household spices can cure all diseases, but pharmaceutical companies are suppressing the information.",
            "Famous actor admits to being a time traveler from the future, here to prevent historical disasters.",
            "Leaked documents prove the moon landing was faked and NASA has been lying to the public for over 50 years."
        ]
    }
    
    # Real news sample
    real_data = {
        'title': [
            "Scientists Develop New Cancer Treatment",
            "Economy Shows Signs of Recovery",
            "New Climate Agreement Signed",
            "Technology Advances in Renewable Energy"
        ],
        'text': [
            "Researchers at leading universities have developed a promising new treatment for certain types of cancer, showing positive results in clinical trials.",
            "Economic indicators suggest that the recovery is gaining momentum, with unemployment rates falling and consumer confidence rising.",
            "World leaders have signed a historic agreement to combat climate change, committing to significant reductions in carbon emissions.",
            "Breakthroughs in solar and wind technology are making renewable energy more efficient and cost-effective than ever before."
        ]
    }
    
    # Create DataFrames and save
    fake_df = pd.DataFrame(fake_data)
    real_df = pd.DataFrame(real_data)
    
    fake_df.to_csv('fake_news_sample.csv', index=False)
    real_df.to_csv('real_news_sample.csv', index=False)
    
    logger.info("Sample datasets created: fake_news_sample.csv, real_news_sample.csv")

def setup_project():
    """Initial project setup."""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create sample datasets
    create_sample_datasets()
    
    logger.info("Project setup completed successfully")

if __name__ == "__main__":
    setup_project()
