import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean text data by removing punctuation, converting to lowercase,
        and removing stopwords (optional).
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, fake_news_path=None, real_news_path=None):
        """
        Load and preprocess fake and real news datasets.
        If paths are None, create sample data for demonstration.
        """
        if fake_news_path and real_news_path:
            # Load actual datasets
            try:
                fake_df = pd.read_csv(fake_news_path)
                real_df = pd.read_csv(real_news_path)
            except FileNotFoundError:
                print("Dataset files not found. Creating sample data...")
                return self.create_sample_data()
        else:
            # Create sample data for demonstration
            return self.create_sample_data()
        
        # Add labels
        fake_df['label'] = 0  # Fake news
        real_df['label'] = 1  # Real news
        
        # Combine datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)
        
        # Preprocess text
        df = self.preprocess_dataframe(df)
        
        return df
    
    def create_sample_data(self):
        """
        Create sample fake and real news data for demonstration.
        """
        sample_data = {
            'title': [
                "Breaking: Scientists Discover Cure for Cancer",
                "President Announces New Economic Policy",
                "Celebrity Found Alive on Mars",
                "Local Team Wins Championship",
                "Aliens Land in New York City",
                "Stock Market Reaches New Heights",
                "Miracle Weight Loss Pill Discovered",
                "Government Passes New Education Bill"
            ],
            'text': [
                "Scientists have made a breakthrough discovery that could cure all forms of cancer. The research team found a compound that targets cancer cells without harming healthy tissue.",
                "The president today announced comprehensive economic reforms aimed at boosting growth and reducing unemployment. The plan includes tax cuts and infrastructure investments.",
                "In a shocking development, a famous celebrity was reportedly found alive on Mars by NASA rovers. The celebrity had been missing for over a year.",
                "The hometown team achieved victory in the national championship game last night, defeating their rivals with a score of 3-1 in an exciting match.",
                "Multiple witnesses reported seeing UFOs landing in Central Park today. Authorities have cordoned off the area as investigations continue.",
                "The stock market reached record highs today as investors showed renewed confidence in the economic recovery. Major indices all posted significant gains.",
                "A new miracle pill promises to help people lose 50 pounds in just one week without diet or exercise. Scientists are skeptical of the claims.",
                "Congress passed new legislation today that will increase funding for public schools and provide scholarships for low-income students."
            ],
            'label': [0, 1, 0, 1, 0, 1, 0, 1]  # 0=Fake, 1=Real
        }
        
        df = pd.DataFrame(sample_data)
        return self.preprocess_dataframe(df)
    
    def preprocess_dataframe(self, df):
        """
        Preprocess the entire dataframe.
        """
        # Handle missing values
        df = df.fillna('')
        
        # Combine title and text
        if 'title' in df.columns and 'text' in df.columns:
            df['combined_text'] = df['title'] + ' ' + df['text']
        elif 'text' in df.columns:
            df['combined_text'] = df['text']
        else:
            raise ValueError("DataFrame must have either 'text' column or both 'title' and 'text' columns")
        
        # Clean the combined text
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        return df
    
    def get_features_and_labels(self, df):
        """
        Extract features and labels from preprocessed dataframe.
        """
        X = df['cleaned_text']
        y = df['label']
        return X, y
