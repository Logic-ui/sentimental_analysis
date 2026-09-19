"""
Launch script for the Coronavirus Tweet Sentiment Analysis Web Application.
Checks model and data prerequisites, then boots the FastAPI Uvicorn server.
"""
import os
import sys
import subprocess
import uvicorn

def ensure_prerequisites():
    required_files = [
        os.path.join('app', 'models', 'vectorizer.joblib'),
        os.path.join('app', 'models', 'sgd_binary.joblib'),
        os.path.join('app', 'data', 'eda_stats.json'),
        os.path.join('app', 'data', 'model_benchmark.json')
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("Model artifacts or EDA datasets missing. Running train_and_export.py...")
        subprocess.run([sys.executable, 'train_and_export.py'], check=True)
    else:
        print("All ML models and EDA datasets verified.")

def main():
    ensure_prerequisites()
    
    port = int(os.environ.get("PORT", 8000))
    host = "127.0.0.1"
    
    print("\n" + "=" * 65)
    print("  COVID-19 Sentiment AI Web Application Server")
    print(f"  Local Dashboard: http://{host}:{port}")
    print(f"  Interactive API Docs: http://{host}:{port}/docs")
    print("=" * 65 + "\n")
    
    uvicorn.run("app.main:app", host=host, port=port, reload=False, log_level="info")

if __name__ == '__main__':
    main()
