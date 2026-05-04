import pandas as pd
import joblib
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import DataPreprocessor


class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words="english",
            ngram_range=(1, 2)   # improves accuracy a lot
        )

        self.model = LogisticRegression(
            max_iter=2000,
            C=2.0   # better decision boundary
        )

        self.is_trained = False

    # ================= TRAIN =================
    def train(self):
        # Create sample data if CSV doesn't exist
        if not os.path.exists("fake_news_dataset.csv"):
            data = {
                "text": [
                    "Scientists discover breakthrough in cancer research",
                    "Aliens found living among us shocking viral news", 
                    "Stock market rises due to economic recovery",
                    "Government announces new healthcare policy",
                    "Celebrity spotted with mysterious creature",
                    "Breaking: Miracle cure discovered in kitchen",
                    "Economy shows signs of improvement",
                    "Shocking revelation about moon landing"
                ],
                "label": [1, 0, 1, 1, 0, 0, 1, 0]  # 1=Real, 0=Fake
            }
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv("fake_news_dataset.csv")

        # safety cleanup
        df = df.dropna()
        df["text"] = df["text"].astype(str)

        X = df["text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y   # IMPORTANT for accuracy
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)

        preds = self.model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        print(f"✅ Model Accuracy: {acc * 100:.2f}%")

        self.is_trained = True

    # ================= PREDICT =================
    def predict(self, text):
        text = str(text).lower()

        vec = self.vectorizer.transform([text])
        proba = self.model.predict_proba(vec)[0]

        fake_prob = proba[0]
        real_prob = proba[1]

        confidence = max(fake_prob, real_prob)

        # SMART THRESHOLD (fix uncertain issue)
        if real_prob >= 0.70:
            label = "🟢 Real News"
        elif fake_prob >= 0.70:
            label = "🔴 Fake News"
        else:
            label = "🟡 Uncertain (Low Confidence)"

        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "fake": round(fake_prob * 100, 2),
                "real": round(real_prob * 100, 2)
            }
        }

    # ================= SAVE =================
    def save_model(self):
        joblib.dump(self.model, "model.pkl")
        joblib.dump(self.vectorizer, "vectorizer.pkl")

    # ================= LOAD =================
    def load_model(self):
        self.model = joblib.load("model.pkl")
        self.vectorizer = joblib.load("vectorizer.pkl")
        self.is_trained = True