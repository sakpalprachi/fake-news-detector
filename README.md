# 🔍 Fake News Detection Web Application

An advanced AI-powered system for detecting fake news using Machine Learning, Python, and modern web technologies.

## 🌟 Features

### 🤖 Machine Learning
- **TF-IDF Vectorizer** for text feature extraction
- **Logistic Regression** classifier for high accuracy
- **Automated data preprocessing** with text cleaning
- **Model accuracy tracking** and performance metrics
- **Confidence scoring** for predictions

### 🌐 FastAPI Backend
- **RESTful API** with `/predict` endpoint
- **Batch prediction** support
- **Model retraining** capabilities
- **Health check** endpoints
- **CORS support** for web integration
- **Comprehensive error handling**

### 🎨 Streamlit Frontend
- **Modern, responsive UI** with gradient designs
- **Real-time predictions** with confidence scores
- **Interactive charts** and visualizations
- **Loading spinners** and progress indicators
- **Example texts** for quick testing
- **Prediction history** tracking

### 📊 Advanced Features
- **Confidence visualization** with gauge charts
- **Probability distribution** analysis
- **Feature importance** extraction
- **Model performance** metrics
- **Error handling** and validation
- **Logging system** for monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- pip package manager

### Installation

1. **Clone or download** the project:
```bash
git clone <repository-url>
cd fakenewsdetector
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:

   **Option 1: Start both API and Web App**
   ```bash
   # Terminal 1: Start the API server
   python api.py
   
   # Terminal 2: Start the Streamlit app
   streamlit run app.py
   ```

   **Option 2: Use the startup script**
   ```bash
   python startup.py
   ```

4. **Access the application**:
   - Web Interface: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

## 📁 Project Structure

```
fakenewsdetector/
├── app.py                 # Streamlit frontend application
├── api.py                 # FastAPI backend server
├── ml_model.py           # Machine learning model
├── data_preprocessing.py  # Data preprocessing utilities
├── utils.py              # Helper functions and utilities
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── startup.py            # Application startup script
├── model.pkl             # Trained model (auto-generated)
├── vectorizer.pkl        # TF-IDF vectorizer (auto-generated)
├── confusion_matrix.png  # Model performance visualization
└── fake_news_detector.log # Application logs
```

## 🎯 Usage

### Web Interface
1. Open http://localhost:8501 in your browser
2. Enter news title (optional) and content
3. Click "🔍 Check News" to analyze
4. View prediction results with confidence scores
5. Explore detailed analysis and visualizations

### API Usage

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your news text here"}'
```

#### With Title
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Breaking News Title",
       "text": "Your news content here"
     }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '[
       {"text": "First news text"},
       {"text": "Second news text"}
     ]'
```

#### Model Information
```bash
curl -X GET "http://localhost:8000/model/info"
```

## 🔧 Configuration

### Model Settings
Edit `Config` class in `utils.py` to modify:
- `MAX_FEATURES`: TF-IDF maximum features (default: 5000)
- `NGRAM_RANGE`: N-gram range for text features (default: (1, 2))
- `TEST_SIZE`: Training/test split ratio (default: 0.2)
- `CONFIDENCE_THRESHOLD`: Minimum confidence threshold (default: 70)

### API Settings
- `API_HOST`: Server host (default: "0.0.0.0")
- `API_PORT`: Server port (default: 8000)

## 📊 Model Performance

The system uses:
- **TF-IDF Vectorization** for text feature extraction
- **Logistic Regression** for classification
- **Stratified sampling** for balanced training
- **Cross-validation** for robust evaluation

### Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: False positive minimization
- **Recall**: False negative minimization
- **F1-Score**: Balanced precision-recall metric

## 🧪 Testing

### Sample Data
The system includes sample datasets for testing:
- `fake_news_sample.csv`: Sample fake news articles
- `real_news_sample.csv`: Sample real news articles

### Test the Model
```python
from ml_model import FakeNewsDetector

# Initialize and train
detector = FakeNewsDetector()
accuracy = detector.train()

# Test prediction
result = detector.predict("Your news text here")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

## 🔍 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/model/retrain` | Retrain model |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/model/features` | Feature importance |

## 🎨 Customization

### Adding Custom Datasets
1. Place CSV files in the project directory
2. Ensure columns: `title`, `text`
3. Update file paths in training code

### Model Training with Custom Data
```python
from ml_model import FakeNewsDetector

detector = FakeNewsDetector()
detector.train(
    fake_news_path="your_fake_news.csv",
    real_news_path="your_real_news.csv"
)
detector.save_model()
```

### UI Customization
- Modify CSS styles in `app.py`'s `load_css()` function
- Update color schemes and layouts
- Add new visualizations using Plotly

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Error**
   - Ensure API server is running (`python api.py`)
   - Check if port 8000 is available
   - Verify firewall settings

2. **Model Not Found**
   - Run the training script first
   - Check if `model.pkl` and `vectorizer.pkl` exist
   - Use `/model/retrain` endpoint to retrain

3. **Memory Issues**
   - Reduce `MAX_FEATURES` in configuration
   - Use smaller datasets for training
   - Close unused applications

4. **Slow Predictions**
   - Optimize TF-IDF parameters
   - Use smaller n-gram ranges
   - Consider model quantization

### Logs
Check `fake_news_detector.log` for detailed error information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- scikit-learn for machine learning tools
- Streamlit for the web framework
- FastAPI for the backend API
- Plotly for data visualization
- NLTK for text processing

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Examine the application logs
- Create an issue on GitHub

---

**Built with ❤️ using Python, Machine Learning, and Modern Web Technologies**
