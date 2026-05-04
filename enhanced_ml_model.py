import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from data_preprocessing import DataPreprocessor
import json
from datetime import datetime

class EnhancedFakeNewsDetector:
    def __init__(self):
        # Enhanced TF-IDF configuration
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            max_df=0.7,
            min_df=2
        )
        
        # Multiple ML models
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000, C=1.0, solver='liblinear'),
            'naive_bayes': MultinomialNB(alpha=0.1),
            'passive_aggressive': SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        }
        
        # Ensemble model
        self.ensemble = None
        self.best_model = None
        self.model_scores = {}
        
        self.preprocessor = DataPreprocessor()
        self.is_trained = False
    
    def train_single_model(self, model_name, X_train, y_train, X_test, y_test, cv_folds_local=None):
        """Train a single model and return its performance metrics"""
        model = self.models[model_name]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds_local or 5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
    
    def train(self, fake_news_path=None, real_news_path=None):
        """Train all models and create ensemble"""
        print("🚀 Enhanced Fake News Detection - Training Multiple Models...")
        
        # Load and preprocess data
        df = self.preprocessor.load_and_preprocess_data(fake_news_path, real_news_path)
        X, y = self.preprocessor.get_features_and_labels(df)
        
        print(f"📊 Dataset size: {len(X)} samples")
        print(f"📰 Fake news: {sum(y == 0)}, Real news: {sum(y == 1)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Reduce dataset size for cross-validation
        if len(X_train) < 10:
            cv_folds_local = 3  # Minimum folds for small datasets
        else:
            cv_folds_local = 5
        
        # Vectorize text data
        print("🔤 Vectorizing text data...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train all models
        print("🤖 Training multiple models...")
        model_results = {}
        
        for model_name in self.models.keys():
            print(f"  Training {model_name.replace('_', ' ').title()}...")
            result = self.train_single_model(model_name, X_train_tfidf, y_train, X_test_tfidf, y_test, cv_folds_local)
            model_results[model_name] = result
            self.model_scores[model_name] = result['accuracy']
            
            print(f"    ✅ Accuracy: {result['accuracy']:.4f}")
            print(f"    📈 CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
        
        # Select best model
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"📊 Best Accuracy: {self.model_scores[best_model_name]:.4f}")
        
        # Create ensemble with top 3 models (only models with predict_proba)
        top_models = sorted(self.model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        ensemble_models = []
        
        for name, _ in top_models:
            model = self.models[name]
            # Check if model supports predict_proba
            if hasattr(model, 'predict_proba'):
                ensemble_models.append((name, model))
                print(f"  ✅ Added {name.replace('_', ' ').title()} to ensemble")
            else:
                print(f"  ⚠️ Skipped {name.replace('_', ' ').title()} (no predict_proba)")
        
        if ensemble_models:
            print(f"🎯 Creating ensemble with {len(ensemble_models)} models: {[name.replace('_', ' ').title() for name, _ in ensemble_models]}")
            
            # Create voting ensemble
            self.ensemble = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'  # Use probability averaging
            )
        else:
            print("⚠️ No suitable models for ensemble (all require predict_proba)")
            self.ensemble = None
        
        # Train ensemble
        self.ensemble.fit(X_train_tfidf, y_train)
        
        # Evaluate ensemble
        ensemble_pred = self.ensemble.predict(X_test_tfidf)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"🚀 Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        # Save detailed results
        results_summary = {
            'training_date': datetime.now().isoformat(),
            'individual_models': {},
            'ensemble_accuracy': ensemble_accuracy,
            'best_model': best_model_name,
            'dataset_size': len(X)
        }
        
        for name, result in model_results.items():
            results_summary['individual_models'][name] = {
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std'])
            }
        
        # Save results
        with open('model_performance.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        self.is_trained = True
        print("✅ Training completed successfully!")
        
        return {
            'best_accuracy': self.model_scores[best_model_name],
            'ensemble_accuracy': ensemble_accuracy,
            'best_model': best_model_name,
            'all_scores': self.model_scores
        }
    
    def predict(self, text, use_ensemble=True):
        """Make prediction using ensemble or best single model"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Preprocess input text
        cleaned_text = self.preprocessor.clean_text(text)
        
        if not cleaned_text.strip():
            raise ValueError("Input text is empty after preprocessing.")
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([cleaned_text])
        
        # Choose prediction method
        if use_ensemble and self.ensemble is not None:
            # Use ensemble prediction
            prediction = self.ensemble.predict(text_tfidf)[0]
            probabilities = self.ensemble.predict_proba(text_tfidf)[0]
            method_used = "Ensemble (Top 3 Models)"
        else:
            # Use best single model
            prediction = self.best_model.predict(text_tfidf)[0]
            probabilities = self.best_model.predict_proba(text_tfidf)[0]
            method_used = "Best Single Model"
        
        # Get confidence score
        confidence = max(probabilities) * 100
        
        # Smart label
        if confidence < 60:
            final_label = "Uncertain ⚠️"
        elif prediction == 1:
            final_label = "Real News ✅"
        else:
            final_label = "Fake News ❌"
        
        # Suspicious keywords detection
        fake_keywords = ["breaking", "shocking", "viral", "alert", "exclusive", "miracle", "conspiracy"]
        found_words = [word for word in fake_keywords if word in cleaned_text.lower()]
        
        result = {
            'prediction': final_label,
            'raw_prediction': int(prediction),
            'confidence': round(confidence, 2),
            'probabilities': {
                'fake': round(probabilities[0] * 100, 2),
                'real': round(probabilities[1] * 100, 2)
            },
            'method_used': method_used,
            'suspicious_words': found_words,
            'model_performance': self.model_scores if hasattr(self, 'model_scores') else {}
        }
        
        return result
    
    def get_model_comparison(self):
        """Get detailed comparison of all models"""
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        return {
            'status': 'success',
            'models': self.model_scores,
            'best_model': max(self.model_scores, key=self.model_scores.get),
            'ensemble_available': self.ensemble is not None
        }
    
    def save_model(self, vectorizer_path='vectorizer.pkl', model_path='enhanced_model.pkl'):
        """Save the trained models and vectorizer"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'ensemble': self.ensemble,
            'best_model_name': max(self.model_scores, key=self.model_scores.get) if self.model_scores else None,
            'model_scores': self.model_scores
        }
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Enhanced models saved as '{model_path}' and vectorizer as '{vectorizer_path}'")
    
    def load_model(self, vectorizer_path='vectorizer.pkl', model_path='enhanced_model.pkl'):
        """Load the trained models and vectorizer"""
        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.ensemble = model_data['ensemble']
            self.best_model = self.models[model_data['best_model_name']]
            self.model_scores = model_data['model_scores']
            self.is_trained = True
            
            print("✅ Enhanced models loaded successfully!")
            
        except FileNotFoundError:
            print("❌ Model files not found. Please train the models first.")
            self.is_trained = False

# Training script
if __name__ == "__main__":
    detector = EnhancedFakeNewsDetector()
    
    # Train all models
    results = detector.train()
    
    # Save models
    detector.save_model()
    
    # Test predictions
    test_texts = [
        "Scientists discover breakthrough in cancer research treatment",
        "Aliens found living among us shocking viral news conspiracy",
        "Stock market reaches new heights amid economic recovery",
        "Government announces new healthcare policy for citizens",
        "Celebrity spotted with mysterious creature in backyard"
    ]
    
    print("\n🧪 Testing Enhanced Model:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        result = detector.predict(text)
        
        print(f"\nTest {i}:")
        print(f"Text: {text[:60]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Method: {result['method_used']}")
        print(f"Suspicious Words: {result['suspicious_words']}")
        print("-" * 30)
