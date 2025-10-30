"""
FastAPI Inference Server for Sentiment Classification

This API provides endpoints for sentiment analysis using a fine-tuned DistilBERT model.
The model classifies text as either Positive or Negative sentiment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
from datetime import datetime
import os

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Binary sentiment classification API using fine-tuned DistilBERT on IMDb reviews",
    version="1.0.0"
)

# Configuration
MODEL_PATH = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for model and tokenizer
model = None
tokenizer = None


class TextInput(BaseModel):
    """Single text input for sentiment analysis"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze (1-5000 characters)")
    

class BatchTextInput(BaseModel):
    """Batch of texts for sentiment analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze (max 100)")


class SentimentResult(BaseModel):
    """Sentiment analysis result for a single text"""
    text: str
    sentiment: str
    confidence: float
    label_probabilities: dict
    processing_time_ms: float


class BatchSentimentResult(BaseModel):
    """Batch sentiment analysis results"""
    results: List[SentimentResult]
    total_processed: int
    total_time_ms: float


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup"""
    global model, tokenizer
    
    print(f"Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model directory not found: {MODEL_PATH}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        
        print(f"✓ Model loaded successfully on {DEVICE}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for a single text
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with sentiment prediction and probabilities
    """
    start_time = datetime.now()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
    
    # Extract results
    negative_prob = probabilities[0].item()
    positive_prob = probabilities[1].item()
    
    predicted_label = 1 if positive_prob > negative_prob else 0
    sentiment = "Positive" if predicted_label == 1 else "Negative"
    confidence = max(positive_prob, negative_prob)
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "label_probabilities": {
            "negative": round(negative_prob, 4),
            "positive": round(positive_prob, 4)
        },
        "processing_time_ms": round(processing_time, 2)
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis API</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            .endpoint {
                background-color: #ecf0f1;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }
            code {
                background-color: #2c3e50;
                color: #f39c12;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
            .method {
                display: inline-block;
                padding: 3px 10px;
                border-radius: 3px;
                font-weight: bold;
                color: white;
                margin-right: 10px;
            }
            .get { background-color: #27ae60; }
            .post { background-color: #2980b9; }
            a {
                color: #3498db;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Sentiment Analysis API</h1>
            <p>Fine-tuned DistilBERT model for binary sentiment classification (Positive/Negative)</p>
            
            <h2>📚 Available Endpoints</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <code>/health</code>
                <p>Check API health status and model information</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <code>/predict</code>
                <p>Analyze sentiment of a single text</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <code>/predict/batch</code>
                <p>Analyze sentiment of multiple texts (up to 100)</p>
            </div>
            
            <h2>📖 Interactive Documentation</h2>
            <p>
                Visit <a href="/docs">/docs</a> for interactive API documentation (Swagger UI)<br>
                Visit <a href="/redoc">/redoc</a> for alternative documentation (ReDoc)
            </p>
            
            <h2>🚀 Quick Test</h2>
            <p>Try this curl command:</p>
            <code style="display: block; padding: 10px; margin: 10px 0;">
                curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "This movie was amazing!"}'
            </code>
            
            <h2>📊 Model Details</h2>
            <ul>
                <li><strong>Model:</strong> DistilBERT (distilbert-base-uncased)</li>
                <li><strong>Task:</strong> Binary Sentiment Classification</li>
                <li><strong>Training Data:</strong> IMDb Movie Reviews (50,000 samples)</li>
                <li><strong>Accuracy:</strong> ~92-94%</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=SentimentResult)
async def predict(input_data: TextInput):
    """
    Predict sentiment for a single text
    
    Args:
        input_data: TextInput object containing the text to analyze
        
    Returns:
        SentimentResult with prediction details
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = predict_sentiment(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchSentimentResult)
async def predict_batch(input_data: BatchTextInput):
    """
    Predict sentiment for multiple texts
    
    Args:
        input_data: BatchTextInput object containing list of texts
        
    Returns:
        BatchSentimentResult with all predictions
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        results = [predict_sentiment(text) for text in input_data.texts]
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "results": results,
            "total_processed": len(results),
            "total_time_ms": round(total_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    print("Starting Sentiment Analysis API...")
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print("\nServer will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
