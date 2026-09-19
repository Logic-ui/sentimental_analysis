import urllib.request
import json

def get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())

def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode())

def test_all():
    print("1. Testing /api/eda-stats...")
    status, eda = get("http://127.0.0.1:8000/api/eda-stats")
    assert status == 200
    assert eda["summary"]["total_tweets"] == 41157
    print(f" -> OK: {eda['summary']['total_tweets']:,} tweets, {eda['summary']['unique_locations']:,} locations.")

    print("2. Testing /api/model-benchmark...")
    status, bench = get("http://127.0.0.1:8000/api/model-benchmark")
    assert status == 200
    assert len(bench["leaderboard"]) == 7
    print(f" -> OK: {len(bench['leaderboard'])} benchmarked models. Winner: {bench['leaderboard'][0]['model']} ({bench['leaderboard'][0]['accuracy']}%)")

    print("3. Testing /api/samples...")
    status, samples = get("http://127.0.0.1:8000/api/samples")
    assert status == 200
    assert len(samples) >= 5
    print(f" -> OK: {len(samples)} real pandemic sample tweets available.")

    print("4. Testing /api/predict on curated positive sample...")
    pos_text = samples[0]["text"]
    status, pred_pos = post("http://127.0.0.1:8000/api/predict", {
        "text": pos_text,
        "model_type": "sgd",
        "mode": "three_class"
    })
    assert status == 200
    assert pred_pos["predicted_sentiment"] == "Positive"
    print(f" -> OK: Predicted={pred_pos['predicted_sentiment']}, Confidence={pred_pos['confidence']}%, Model={pred_pos['model_used']}")

    print("5. Testing /api/predict on binary mode (notebook primary benchmark)...")
    status, pred_bin = post("http://127.0.0.1:8000/api/predict", {
        "text": pos_text,
        "model_type": "sgd",
        "mode": "binary"
    })
    assert status == 200
    assert pred_bin["predicted_sentiment"] == "Positive / Neutral"
    print(f" -> OK: Binary={pred_bin['predicted_sentiment']}, Confidence={pred_bin['confidence']}%")

    print("6. Testing /api/batch-predict...")
    status, batch = post("http://127.0.0.1:8000/api/batch-predict", {
        "texts": [s["text"] for s in samples[:4]]
    })
    assert status == 200
    assert batch["total_analyzed"] == 4
    print(f" -> OK: Batch analyzed {batch['total_analyzed']} items successfully. Distribution: {batch['distribution']}")

    print("7. Testing /api/predict emotion dimensions & attributions payload...")
    assert "emotions" in pred_pos
    assert "Optimism & Gratitude" in pred_pos["emotions"]
    assert "attributions" in pred_pos
    assert len(pred_pos["attributions"]) > 0
    print(f" -> OK: Emotions computed: {pred_pos['emotions']}")
    print(f" -> OK: Text X-Ray token attributions verified: {len(pred_pos['attributions'])} tokens extracted.")

    print("8. Testing /api/predict-arena (Multi-Model Battle Arena)...")
    status, arena = post("http://127.0.0.1:8000/api/predict-arena", {
        "text": pos_text,
        "mode": "three_class"
    })
    assert status == 200
    assert len(arena["models"]) == 3
    assert arena["consensus_status"] in ["unanimous", "majority", "split"]
    print(f" -> OK: Arena battle executed across 3 models: {arena['consensus_summary']}.")
    for m in arena["models"]:
        print(f"    - {m['name']}: {m['prediction']} ({m['confidence']}%, Latency: {m['latency_ms']}ms)")

    print("\n[PASS] ALL BACKEND API & ARENA TESTS PASSED!")

if __name__ == "__main__":
    test_all()
