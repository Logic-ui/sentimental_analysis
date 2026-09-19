"""
FastAPI Backend Application for Coronavirus Sentiment Analysis.
Serves real-time inference, model benchmarks, and EDA dataset visual data.
"""
import os
import re
import json
import time
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

def compute_emotion_dimensions(tokens, sentiment: str, confidence: float):
    """Estimate 5 pandemic emotional dimensions for radar chart visualization."""
    dimensions = {
        'Optimism & Gratitude': 15.0,
        'Panic & Urgency': 12.0,
        'Frustration & Grievance': 12.0,
        'Public Health & Safety': 15.0,
        'Factual / Informative': 20.0
    }
    
    token_set = set(tokens)
    
    gratitude_kw = {'thank', 'gratitud', 'hero', 'bless', 'amaz', 'wonder', 'love', 'kind', 'gener', 'help', 'apprec', 'safe', 'support', 'great', 'super'}
    panic_kw = {'panic', 'shortag', 'empti', 'toilet', 'paper', 'food', 'stock', 'hoard', 'rush', 'scare', 'fear', 'crisi', 'alarm', 'crazy', 'mad'}
    frust_kw = {'price', 'goug', 'greed', 'scam', 'shame', 'disgust', 'terribl', 'aw', 'wait', 'rude', 'complain', 'bad', 'worst', 'stupid'}
    health_kw = {'mask', 'sanit', 'distanc', 'spread', 'quarantin', 'isolat', 'symptom', 'doctor', 'nurs', 'health', 'stayhom', 'wash', 'clean', 'protect'}
    info_kw = {'updat', 'hour', 'open', 'close', 'store', 'market', 'order', 'notic', 'guidelin', 'announc', 'report', 'march', 'april', 'custom'}
    
    for t in token_set:
        for kw in gratitude_kw:
            if kw in t: dimensions['Optimism & Gratitude'] += 18.0
        for kw in panic_kw:
            if kw in t: dimensions['Panic & Urgency'] += 18.0
        for kw in frust_kw:
            if kw in t: dimensions['Frustration & Grievance'] += 18.0
        for kw in health_kw:
            if kw in t: dimensions['Public Health & Safety'] += 18.0
        for kw in info_kw:
            if kw in t: dimensions['Factual / Informative'] += 16.0
            
    conf_boost = float(confidence)
    if 'Positive' in sentiment:
        dimensions['Optimism & Gratitude'] += conf_boost * 0.35
        dimensions['Public Health & Safety'] += conf_boost * 0.15
    elif 'Negative' in sentiment:
        dimensions['Frustration & Grievance'] += conf_boost * 0.30
        dimensions['Panic & Urgency'] += conf_boost * 0.25
    else:
        dimensions['Factual / Informative'] += conf_boost * 0.40
        dimensions['Public Health & Safety'] += conf_boost * 0.15
        
    return {k: max(10.0, min(95.0, round(v, 1))) for k, v in dimensions.items()}

def compute_word_attributions(raw_text: str, vocab, lr_model):
    """Assign sentiment polarity to words directly in raw_text for interactive Text X-Ray highlighting."""
    if lr_model is None or not hasattr(lr_model, 'coef_'):
        return []
    classes = lr_model.classes_.tolist()
    pos_idx = classes.index('Positive') if 'Positive' in classes else 0
    neg_idx = classes.index('Negative') if 'Negative' in classes else 0
    
    raw_tokens = re.findall(r"[\w']+|[.,!?;:\-–—\(\)\"\/#@]", raw_text)
    attributions = []
    
    for token in raw_tokens:
        clean = token.lower().strip()
        stemmed = stemmer.stem(clean)
        polarity = 'none'
        score = 0.0
        
        target = clean if clean in vocab else (stemmed if stemmed in vocab else None)
        if target and target in vocab:
            idx = vocab[target]
            pos_weight = float(lr_model.coef_[pos_idx][idx])
            neg_weight = float(lr_model.coef_[neg_idx][idx])
            
            if pos_weight > 0.35 and pos_weight > neg_weight:
                polarity = 'positive'
                score = round(pos_weight, 2)
            elif neg_weight > 0.35 and neg_weight > pos_weight:
                polarity = 'negative'
                score = round(neg_weight, 2)
            elif abs(pos_weight - neg_weight) < 0.2 and (pos_weight > 0.25 or neg_weight > 0.25):
                polarity = 'neutral'
                score = round(pos_weight, 2)
                
        attributions.append({
            'text': token,
            'is_word': bool(re.match(r"^[A-Za-z0-9]+$", token)),
            'polarity': polarity,
            'score': score
        })
    return attributions

class PredictionRequest(BaseModel):
    text: str
    model_type: Optional[str] = "sgd"  # "sgd", "logistic_regression", "naive_bayes"
    mode: Optional[str] = "three_class"  # "binary", "three_class"

class ArenaRequest(BaseModel):
    text: str
    mode: Optional[str] = "three_class"

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
    """Predict sentiment of a single tweet with full confidence breakdown, keyword impact, emotions, and token attributions."""
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
                
                if pos_weight > 0.35 and pos_weight > neg_weight:
                    keyword_impacts.append({'word': w, 'sentiment': 'Positive', 'weight': round(pos_weight, 2)})
                elif neg_weight > 0.35 and neg_weight > pos_weight:
                    keyword_impacts.append({'word': w, 'sentiment': 'Negative', 'weight': round(neg_weight, 2)})
                elif abs(pos_weight - neg_weight) < 0.2:
                    keyword_impacts.append({'word': w, 'sentiment': 'Neutral', 'weight': round(pos_weight, 2)})

    # Sort keywords by impact magnitude
    keyword_impacts = sorted(keyword_impacts, key=lambda x: abs(x['weight']), reverse=True)[:8]

    # Compute emotion radar dimensions & token attributions
    conf_pct = round(confidence * 100, 1)
    emotions = compute_emotion_dimensions(pipeline_info['tokens'], predicted_sentiment, conf_pct)
    attributions = compute_word_attributions(req.text, vocab, lr_model)

    return {
        "text": req.text,
        "predicted_sentiment": predicted_sentiment,
        "confidence": conf_pct,
        "probabilities": probabilities,
        "model_used": model_name,
        "mode": mode,
        "pipeline_steps": pipeline_info,
        "sentiment_keywords": keyword_impacts,
        "emotions": emotions,
        "attributions": attributions
    }

@app.post("/api/predict-arena")
def predict_arena(req: ArenaRequest):
    """Run parallel multi-model arena comparison (SGD, Logistic Regression, Naive Bayes)."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if 'vectorizer' not in models_cache:
        load_cached_models()
        if 'vectorizer' not in models_cache:
            raise HTTPException(status_code=500, detail="Models are not initialized.")

    vec = models_cache['vectorizer']
    pipeline_info = preprocess_pipeline(req.text)
    cleaned_text = pipeline_info['cleaned']
    X_vec = vec.transform([cleaned_text])
    
    mode = req.mode or "three_class"
    models_to_test = [
        ('sgd', 'SGD Classifier', 'sgd_tri' if mode == 'three_class' else 'sgd_bin', '85.96% Accuracy (Winner 🏆)', 'var(--color-pos)'),
        ('logistic_regression', 'Logistic Regression', 'lr_tri' if mode == 'three_class' else 'lr_bin', '85.48% Accuracy (Runner-Up 🥈)', 'var(--color-neu)'),
        ('naive_bayes', 'Multinomial Naive Bayes', 'nb_tri' if mode == 'three_class' else 'nb_bin', '78.79% Accuracy (Probabilistic)', 'var(--color-accent)')
    ]
    
    results = []
    votes = {}
    
    for key, display_name, cache_key, badge_label, brand_color in models_to_test:
        model = models_cache[cache_key]
        t0 = time.perf_counter()
        
        if mode == "binary":
            pred_idx = model.predict(X_vec)[0]
            probs = model.predict_proba(X_vec)[0]
            pred_label = "Positive / Neutral" if pred_idx == 1 else "Negative"
            conf = float(probs[pred_idx])
            prob_dict = {
                "Positive / Neutral": round(float(probs[1]) * 100, 1),
                "Negative": round(float(probs[0]) * 100, 1)
            }
        else:
            pred_label = model.predict(X_vec)[0]
            probs = model.predict_proba(X_vec)[0]
            conf = float(np.max(probs))
            classes = model.classes_.tolist()
            prob_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)}
            
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        votes[pred_label] = votes.get(pred_label, 0) + 1
        
        results.append({
            'model_id': key,
            'name': display_name,
            'badge': badge_label,
            'brand_color': brand_color,
            'prediction': pred_label,
            'confidence': round(conf * 100, 1),
            'latency_ms': latency_ms,
            'probabilities': prob_dict
        })
        
    majority_sentiment = max(votes.items(), key=lambda x: x[1])
    if majority_sentiment[1] == 3:
        consensus_status = "unanimous"
        consensus_summary = f"3/3 Unanimous Agreement ({majority_sentiment[0]})"
    elif majority_sentiment[1] == 2:
        consensus_status = "majority"
        consensus_summary = f"2/3 Majority Verdict ({majority_sentiment[0]})"
    else:
        consensus_status = "split"
        consensus_summary = "Split Verdict Across Classifiers"
        
    lr_model = models_cache.get('lr_tri')
    vocab = vec.vocabulary_
    emotions = compute_emotion_dimensions(pipeline_info['tokens'], majority_sentiment[0], 85.0)
    attributions = compute_word_attributions(req.text, vocab, lr_model)
    
    return {
        'text': req.text,
        'mode': mode,
        'consensus_status': consensus_status,
        'consensus_summary': consensus_summary,
        'winning_sentiment': majority_sentiment[0],
        'models': results,
        'emotions': emotions,
        'attributions': attributions,
        'pipeline_steps': pipeline_info
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
