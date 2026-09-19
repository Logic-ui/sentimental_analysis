"""
FastAPI Backend Application for Coronavirus Sentiment Analysis.
Serves real-time inference, model benchmarks, and EDA dataset visual data.
"""
import os
import re
import json
import joblib
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Initialize FastAPI application
app = FastAPI(
    title="Coronavirus Tweet Sentiment Analysis API",
    description="Interactive Sentiment Analysis, NLP Pipeline & EDA Dashboard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Global model cache
models_cache = {}

# Simple Porter stemmer algorithm in pure Python to avoid runtime C-dependency issues
class SimplePorterStemmer:
    """Lightweight rule-based stemmer consistent with Porter Stemmer."""
    def stem(self, word):
        word = word.lower()
        if len(word) <= 3:
            return word
        if word.endswith('sses'):
            return word[:-2]
        if word.endswith('ies'):
            return word[:-2]
        if word.endswith('ss'):
            return word
        if word.endswith('s') and not word.endswith('us') and not word.endswith('is'):
            return word[:-1]
        if word.endswith('eed'):
            return word[:-1]
        if word.endswith('ing') and len(word) > 5:
            return word[:-3]
        if word.endswith('ed') and len(word) > 4:
            return word[:-2]
        if word.endswith('ly') and len(word) > 4:
            return word[:-2]
        return word

stemmer = SimplePorterStemmer()

def load_cached_models():
    """Load pre-trained models into memory on startup."""
    global models_cache
    try:
        models_cache['vectorizer'] = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.joblib'))
        models_cache['sgd_bin'] = joblib.load(os.path.join(MODELS_DIR, 'sgd_binary.joblib'))
        models_cache['lr_bin'] = joblib.load(os.path.join(MODELS_DIR, 'lr_binary.joblib'))
        models_cache['nb_bin'] = joblib.load(os.path.join(MODELS_DIR, 'nb_binary.joblib'))
        models_cache['lr_tri'] = joblib.load(os.path.join(MODELS_DIR, 'lr_tri.joblib'))
        models_cache['sgd_tri'] = joblib.load(os.path.join(MODELS_DIR, 'sgd_tri.joblib'))
        models_cache['nb_tri'] = joblib.load(os.path.join(MODELS_DIR, 'nb_tri.joblib'))
        print("Successfully loaded all models into memory.")
    except Exception as e:
        print(f"Warning: Could not load some models: {e}")

@app.on_event("startup")
def startup_event():
    load_cached_models()

def preprocess_pipeline(raw_text: str):
    """Execute step-by-step preprocessing matching the notebook."""
    # Step 1: Remove URLs
    no_url = re.sub(r'https?:\/\/\S+', '', str(raw_text))
    # Step 2: Remove @user mentions
    no_mentions = re.sub(r'@[A-Za-z0-9_]+', '', no_url)
    # Step 3: Remove special characters and numbers (preserve letters, spaces, hashtags)
    clean_chars = re.sub(r'[^a-zA-Z#\s]', ' ', no_mentions)
    # Step 4: Tokenize & remove short words
    raw_tokens = clean_chars.split()
    tokens = [w.lower() for w in raw_tokens if len(w) > 2]
    # Step 5: Stemming
    stemmed_tokens = [stemmer.stem(w) for w in tokens]
    cleaned_sentence = ' '.join(tokens)
    
    return {
        'raw': raw_text,
        'no_mentions_url': no_mentions.strip(),
        'cleaned': cleaned_sentence,
        'tokens': tokens,
        'stemmed': stemmed_tokens
    }

class PredictionRequest(BaseModel):
    text: str
    model_type: Optional[str] = "sgd"  # "sgd", "logistic_regression", "naive_bayes"
    mode: Optional[str] = "three_class"  # "binary", "three_class"

class BatchRequest(BaseModel):
    texts: List[str]
    model_type: Optional[str] = "sgd"
    mode: Optional[str] = "three_class"

@app.get("/api/eda-stats")
def get_eda_stats():
    """Retrieve precomputed EDA metrics, distributions, and dataset statistics."""
    path = os.path.join(DATA_DIR, "eda_stats.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="EDA data not found. Please run train_and_export.py")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/model-benchmark")
def get_model_benchmark():
    """Retrieve benchmark table for all 7 models from sentimental_analysis.ipynb."""
    path = os.path.join(DATA_DIR, "model_benchmark.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Benchmark data not found. Please run train_and_export.py")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/samples")
def get_sample_tweets():
    """Retrieve pre-curated test tweets across positive, neutral, and negative sentiments."""
    path = os.path.join(DATA_DIR, "sample_tweets.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample tweets not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/predict")
def predict_sentiment(req: PredictionRequest):
    """Predict sentiment of a single tweet with full confidence breakdown and keyword impact."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    if 'vectorizer' not in models_cache:
        load_cached_models()
        if 'vectorizer' not in models_cache:
            raise HTTPException(status_code=500, detail="Models are not initialized.")

    pipeline_info = preprocess_pipeline(req.text)
    cleaned_text = pipeline_info['cleaned']
    
    vec = models_cache['vectorizer']
    X_vec = vec.transform([cleaned_text])
    
    # Model selection & execution
    mode = req.mode or "three_class"
    m_type = (req.model_type or "sgd").lower()
    
    if mode == "binary":
        if m_type == "logistic_regression":
            model = models_cache['lr_bin']
            model_name = "Logistic Regression (85.48% Accuracy)"
        elif m_type == "naive_bayes":
            model = models_cache['nb_bin']
            model_name = "Multinomial Naive Bayes (78.79% Accuracy)"
        else:
            model = models_cache['sgd_bin']
            model_name = "SGD Classifier (85.96% Winner Accuracy)"
            
        pred_idx = model.predict(X_vec)[0]
        # Probs: [P(0=Negative), P(1=Positive/Neutral)]
        probs = model.predict_proba(X_vec)[0]
        
        predicted_sentiment = "Positive / Neutral" if pred_idx == 1 else "Negative"
        confidence = float(probs[pred_idx])
        probabilities = {
            "Positive / Neutral": round(float(probs[1]) * 100, 2),
            "Negative": round(float(probs[0]) * 100, 2)
        }
    else: # 3-class mode
        if m_type == "logistic_regression":
            model = models_cache['lr_tri']
            model_name = "Logistic Regression (3-Class Softmax)"
        elif m_type == "naive_bayes":
            model = models_cache['nb_tri']
            model_name = "Multinomial Naive Bayes (3-Class)"
        else:
            model = models_cache['sgd_tri']
            model_name = "SGD Classifier (3-Class Calibrated)"
            
        pred_label = model.predict(X_vec)[0]
        probs = model.predict_proba(X_vec)[0]
        classes = model.classes_.tolist()
        
        predicted_sentiment = pred_label
        confidence = float(np.max(probs))
        probabilities = {
            cls: round(float(p) * 100, 2) for cls, p in zip(classes, probs)
        }

    # Extract key sentiment-driving words present in the tweet
    words_in_text = set(pipeline_info['tokens'])
    vocab = vec.vocabulary_
    feature_names = vec.get_feature_names_out()
    
    keyword_impacts = []
    # Use logistic regression coefficients to estimate word polarity
    lr_model = models_cache.get('lr_tri')
    if lr_model is not None and hasattr(lr_model, 'coef_'):
        classes = lr_model.classes_.tolist()
        pos_idx = classes.index('Positive') if 'Positive' in classes else 0
        neg_idx = classes.index('Negative') if 'Negative' in classes else 0
        
        for w in words_in_text:
            if w in vocab:
                idx = vocab[w]
                pos_weight = float(lr_model.coef_[pos_idx][idx])
                neg_weight = float(lr_model.coef_[neg_idx][idx])
                
                if pos_weight > 0.4 and pos_weight > neg_weight:
                    keyword_impacts.append({'word': w, 'sentiment': 'Positive', 'weight': round(pos_weight, 2)})
                elif neg_weight > 0.4 and neg_weight > pos_weight:
                    keyword_impacts.append({'word': w, 'sentiment': 'Negative', 'weight': round(neg_weight, 2)})
                elif abs(pos_weight - neg_weight) < 0.2:
                    keyword_impacts.append({'word': w, 'sentiment': 'Neutral', 'weight': round(pos_weight, 2)})

    # Sort keywords by impact magnitude
    keyword_impacts = sorted(keyword_impacts, key=lambda x: abs(x['weight']), reverse=True)[:8]

    return {
        "text": req.text,
        "predicted_sentiment": predicted_sentiment,
        "confidence": round(confidence * 100, 2),
        "probabilities": probabilities,
        "model_used": model_name,
        "mode": mode,
        "pipeline_steps": pipeline_info,
        "sentiment_keywords": keyword_impacts
    }

@app.post("/api/batch-predict")
def batch_predict(req: BatchRequest):
    """Analyze a batch of tweets and return aggregate summary and itemized results."""
    texts = [t.strip() for t in req.texts if t and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="No valid texts provided.")
        
    if len(texts) > 200:
        texts = texts[:200]
        
    vec = models_cache['vectorizer']
    lr = models_cache['lr_tri']
    
    cleaned_list = [preprocess_pipeline(t)['cleaned'] for t in texts]
    X_vec = vec.transform(cleaned_list)
    preds = lr.predict(X_vec)
    probs = lr.predict_proba(X_vec)
    
    results = []
    counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    
    for i, (orig, pred, prob) in enumerate(zip(texts, preds, probs)):
        conf = float(np.max(prob)) * 100
        counts[pred] = counts.get(pred, 0) + 1
        results.append({
            'id': i + 1,
            'text': orig,
            'sentiment': pred,
            'confidence': round(conf, 1)
        })
        
    total = len(results)
    distribution = {
        k: {
            'count': v,
            'percentage': round(v / total * 100, 1)
        }
        for k, v in counts.items()
    }
    
    return {
        'total_analyzed': total,
        'distribution': distribution,
        'results': results
    }

# Mount static files for the frontend web application
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "API is running", "docs": "/docs"})
