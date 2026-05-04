#!/usr/bin/env python3
"""
Fake News Detector - Startup Script
This script initializes and starts both the API server and Streamlit app.
"""

import os
import sys
import subprocess
import time
import threading
from utils import setup_project, logger

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import pandas
        import numpy
        import sklearn
        import streamlit
        import fastapi
        import uvicorn
        import nltk
        import requests
        logger.info("✅ All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False

def train_model_if_needed():
    """Train the model if it doesn't exist."""
    if not (os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl')):
        logger.info("🤖 Training model for the first time...")
        try:
            from ml_model import FakeNewsDetector
            detector = FakeNewsDetector()
            accuracy = detector.train()
            detector.save_model()
            logger.info(f"✅ Model trained successfully with accuracy: {accuracy:.4f}")
            return True
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")
            return False
    else:
        logger.info("✅ Model files already exist")
        return True

def start_api_server():
    """Start the FastAPI server."""
    logger.info("🌐 Starting FastAPI server...")
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        logger.info("API server stopped")
    except Exception as e:
        logger.error(f"Error starting API server: {e}")

def start_streamlit_app():
    """Start the Streamlit application."""
    logger.info("🎨 Starting Streamlit application...")
    time.sleep(3)  # Give API server time to start
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        logger.info("Streamlit app stopped")
    except Exception as e:
        logger.error(f"Error starting Streamlit app: {e}")

def main():
    """Main startup function."""
    print("=" * 60)
    print("🔍 FAKE NEWS DETECTOR - STARTUP")
    print("=" * 60)
    
    # Setup project
    logger.info("🔧 Setting up project...")
    setup_project()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Train model if needed
    if not train_model_if_needed():
        logger.error("❌ Failed to train model. Exiting.")
        sys.exit(1)
    
    print("\n🚀 Starting services...")
    print("📍 API Server: http://localhost:8000")
    print("📍 Web App: http://localhost:8501")
    print("📍 API Docs: http://localhost:8000/docs")
    print("\n⏹️  Press Ctrl+C to stop both services")
    print("=" * 60)
    
    try:
        # Start API server in main thread
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        
        # Start Streamlit app
        start_streamlit_app()
        
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down services...")
        print("\n👋 Thank you for using Fake News Detector!")

if __name__ == "__main__":
    main()
