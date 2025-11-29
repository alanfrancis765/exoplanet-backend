from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join("ML_model", "model_alan.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

FEATURE_COLUMNS = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff',
    'koi_slogg', 'koi_srad', 'koi_model_snr'
]


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "API Running "})

# -------------------------------
# SINGLE PREDICTION
# -------------------------------
@app.route("/api/predict", methods=["POST"])
def predict_single():
    try:
        data = request.get_json()
        print("=" * 50)
        print("Request data:", data)

        # Create feature array
        features = np.array([[data[col] for col in FEATURE_COLUMNS]], dtype=float)

        # Predict
        pred_class = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            confidence = float(max(probs) * 100)
            probabilities = {
                str(cls): round(float(p * 100), 2)
                for cls, p in zip(model.classes_, probs)
            }
        else:
            confidence = 85.0
            probabilities = None

        result = {
            "prediction": str(pred_class),
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "input_data": data
        }

        print("✓ Prediction success:", result)
        print("=" * 50)

        return jsonify(result)

    except Exception as e:
        print("✗ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------------------
# BATCH PREDICTION
# -------------------------------
@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'CSV only'}), 400

        df = pd.read_csv(file)

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400

        results = []
        
        LABEL_MAP = {
            "CANDIDATE": "Candidate",
            "CONFIRMED": "Confirmed Exoplanet",
            "FALSE POSITIVE": "False Positive"
        }

        for i, row in df.iterrows():
            row_data = row[FEATURE_COLUMNS].to_dict()
            features = np.array([[row[col] for col in FEATURE_COLUMNS]], dtype=float)

            pred_class = model.predict(features)[0]
            pred_label = LABEL_MAP.get(str(pred_class), str(pred_class))

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence = float(max(probs) * 100)
                probabilities = {
                    LABEL_MAP.get(str(cls), str(cls)): round(float(p * 100), 2)
                    for cls, p in zip(model.classes_, probs)
                }
            else:
                confidence = 85
                probabilities = None

            results.append({
                "prediction": pred_label,
                "confidence": round(confidence, 2),
                "probabilities": probabilities,
                "input_data": row_data
            })

        summary = get_summary(results)

        return jsonify({
            "total_records": len(results),
            "results": results,
            "summary": summary
        })

    except Exception as e:
        print("✗ ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


# -------------------------------
# MODEL INFO
# -------------------------------
@app.route("/api/model-info", methods=["GET"])
def model_info():
    if model:
        return jsonify({
            "model_loaded": True,
            "model_type": type(model).__name__,
            "supports_probabilities": hasattr(model, "predict_proba"),
            "features": FEATURE_COLUMNS,
            "classes": list(model.classes_)
        })
    return jsonify({"model_loaded": False})


# -------------------------------
# SUMMARY
# -------------------------------
def get_summary(results):
    total = len(results)

    candidates = sum(1 for r in results if r["prediction"] == "Candidate")
    confirmed = sum(1 for r in results if r["prediction"] == "Confirmed Exoplanet")
    false_positives = sum(1 for r in results if r["prediction"] == "False Positive")

    avg_confidence = sum(r["confidence"] for r in results) / total if total else 0

    return {
        "total": total,
        "candidates": candidates,
        "exoplanets": confirmed,
        "false_positives": false_positives,
        "exoplanet_percentage": round((confirmed / total) * 100, 2) if total else 0,
        "average_confidence": round(avg_confidence, 2)
    }


# -------------------------------
# START SERVER
# -------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NASA EXOPLANET DETECTION SYSTEM")
    print("=" * 60)
    print(f"Model Loaded: {'YES' if model else 'NO'}")

    if model:
        print("Model Type:", type(model).__name__)
        print("Classes:", list(model.classes_))

    print("Features:", len(FEATURE_COLUMNS))
    print("=" * 60)
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
