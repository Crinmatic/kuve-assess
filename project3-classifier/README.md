# Text Classification: Sentiment Analysis with DistilBERT

A production-ready sentiment analysis system using fine-tuned DistilBERT on IMDb movie reviews, with FastAPI deployment.

![Screenshot](Screenshot%202025-10-30%20at%2012.11.18.png)

## Overview

This project implements end-to-end text classification for binary sentiment analysis (Positive/Negative). The system includes comprehensive data analysis, model training with deep learning, rigorous evaluation, and a production-ready REST API for inference.

### Features

- **EDA**: Comprehensive exploratory data analysis with visualizations
- **Deep Learning**: Fine-tuned DistilBERT transformer model (66M parameters)
- **High Performance**: 92-94% accuracy on IMDb test set
- **Production API**: FastAPI server with interactive documentation
- **Comprehensive Evaluation**: Multiple metrics, confusion matrix, ROC curves, error analysis
- **Offline Operation**: Runs entirely locally after setup

## Architecture

### Components

1. **Data Processing**
   - Dataset: IMDb movie reviews (50,000 samples)
   - Preprocessing: Text cleaning, HTML removal, normalization
   - Tokenization: DistilBERT tokenizer (256 max tokens)
   - Analysis: Class distribution, text length, word frequency

2. **Model**
   - Base model: `distilbert-base-uncased` (66M parameters)
   - Task: Binary sequence classification
   - Fine-tuning: 3 epochs with early stopping
   - Optimization: AdamW optimizer, mixed precision training
   - Regularization: Weight decay, dropout

3. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Visualizations: Confusion matrix, ROC curve, error analysis
   - Error analysis: Misclassification investigation

4. **API Server**
   - Framework: FastAPI with automatic OpenAPI docs
   - Endpoints: Single prediction, batch prediction, health check
   - Features: Request validation, error handling, performance monitoring

## Technology Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| Model | DistilBERT | distilbert-base-uncased, 66M parameters |
| Framework | PyTorch | 2.0+, mixed precision training |
| Transformers | Hugging Face | 4.30+, model training and inference |
| API | FastAPI | 0.104+, async web framework |
| Server | Uvicorn | Production ASGI server |
| Data | Datasets | Hugging Face datasets library |
| Evaluation | scikit-learn | Classification metrics |
| Visualization | Matplotlib, Seaborn | EDA and results visualization |

## Requirements

- Python 3.8+
- 8GB+ RAM (16GB recommended for training)
- ~2GB disk space for model and data
- GPU optional (3-5min training) vs CPU (10-15min training)

## Installation

### 1. Navigate to Project

```bash
cd project3-classifier
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch and Transformers
- FastAPI and Uvicorn
- Data processing libraries
- Visualization tools

## Usage

### Training (Already Completed)

The model has been trained and saved in the `model/` directory. To retrain or see the process:

```bash
jupyter notebook train_sentiment_model.ipynb
```

The notebook includes:
1. Data loading and EDA
2. Preprocessing and feature analysis
3. Model training with progress tracking
4. Comprehensive evaluation
5. Model saving and inference testing

### Running the API Server

Start the FastAPI server:

```bash
python api.py
```

Server will be available at:
- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Testing the API

Run the comprehensive test suite:

```bash
python test_api.py
```

Tests include:
- Health check
- Single text prediction
- Batch prediction
- Edge cases
- Performance benchmarking

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely amazing!"}'
```

Response:
```json
{
  "text": "This movie was absolutely amazing!",
  "sentiment": "Positive",
  "confidence": 0.9987,
  "label_probabilities": {
    "negative": 0.0013,
    "positive": 0.9987
  },
  "processing_time_ms": 45.23
}
```

#### 3. Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great movie!",
      "Terrible waste of time.",
      "It was okay, nothing special."
    ]
  }'
```

### Python Client Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Loved this movie! Highly recommended!"}
)
result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["Amazing!", "Boring.", "Not bad."]}
)
results = response.json()
for r in results['results']:
    print(f"{r['text']}: {r['sentiment']} ({r['confidence']:.2%})")
```

## Performance

### Model Metrics

- **Accuracy**: 92-94%
- **F1-Score**: 0.92-0.94
- **Precision**: 0.91-0.93
- **Recall**: 0.92-0.94
- **ROC-AUC**: 0.97-0.99

### API Performance

- **Single prediction**: 40-80ms (CPU) / 10-30ms (GPU)
- **Batch throughput**: 15-25 requests/second (CPU)
- **Memory usage**: ~1-2GB (model loaded)
- **Max text length**: 5000 characters (256 tokens)

## Project Structure

```
project3-classifier/
├── train_sentiment_model.ipynb   # Training notebook with EDA
├── api.py                         # FastAPI inference server
├── test_api.py                    # API test suite
├── requirements.txt               # Python dependencies
├── model/                         # Saved model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── vocab.txt
├── class_distribution.png         # EDA visualization
├── confusion_matrix.png           # Model evaluation
├── roc_curve.png                  # ROC-AUC curve
├── text_length_analysis.png       # Text length distribution
├── wordcloud_analysis.png         # Word frequency visualization
└── README.md
```

## Model Details

### Architecture

- **Base Model**: DistilBERT (distilled version of BERT)
- **Parameters**: 66 million (6 layers, 768 hidden, 12 attention heads)
- **Tokenizer**: WordPiece tokenization
- **Max Sequence Length**: 256 tokens
- **Classification Head**: Linear layer with dropout

### Training Configuration

- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Scheduler**: Linear with warmup
- **Batch Size**: 16 (train), 32 (eval)
- **Epochs**: 3 with early stopping
- **Mixed Precision**: FP16 for efficiency
- **Device**: Auto-detected (CUDA/CPU)

### Data

- **Source**: IMDb movie reviews dataset
- **Training samples**: 25,000
- **Test samples**: 25,000
- **Classes**: Binary (Negative: 0, Positive: 1)
- **Balance**: 50/50 class distribution

## Visualizations

The training notebook generates:

1. **Class Distribution**: Bar chart showing balanced classes
2. **Text Length Analysis**: Distribution of review lengths
3. **Word Clouds**: Most frequent words by sentiment
4. **Confusion Matrix**: True vs predicted classifications
5. **ROC Curve**: Model discrimination ability

## Limitations

1. **Domain-Specific**: Trained on movie reviews, may not generalize to other domains
2. **Binary Classification**: Only Positive/Negative (no neutral)
3. **English Only**: No multi-language support
4. **Context Length**: Limited to 256 tokens (~200 words)
5. **Sarcasm**: May struggle with heavily sarcastic text

## Future Improvements

- Multi-class sentiment (add neutral, very positive, very negative)
- Multi-domain training (reviews, tweets, news)
- Larger models (BERT-base, RoBERTa)
- GPU optimization for faster inference
- Batch processing optimization
- Confidence calibration
- Explainability features (attention visualization)
- Docker containerization

## Testing

### Example Test Cases

The test suite includes:
- **Clear sentiment**: "Amazing movie!" → Positive
- **Strong negative**: "Terrible waste of time." → Negative
- **Nuanced**: "Great visuals but weak plot." → Mixed
- **Short text**: "Loved it!" → Positive
- **Edge cases**: Empty, very long, special characters

### Running Tests

```bash
# Start API server in one terminal
python api.py

```
