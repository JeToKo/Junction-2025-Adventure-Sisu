from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = "AI_ai_good_agent.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Failed to load model at {MODEL_PATH}: {e}")

def predict_array(arr, thresh=0.5):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    probs = model.predict(arr, verbose=0).flatten()
    bools = (probs >= thresh).tolist()
    # convert numpy bools to native bools
    return [bool(x) for x in bools]

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 500
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid or missing JSON body"}), 400

    # single vector (list) under "features"
    if "features" in data:
        try:
            result = predict_array(data["features"])
            return jsonify({"prediction": result[0]})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # batch under "batch"
    if "batch" in data:
        try:
            result = predict_array(data["batch"])
            return jsonify({"predictions": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return jsonify({"error": "provide 'features' or 'batch' in JSON body"}), 400

if __name__ == "__main__":
    # For local development; use a production server (gunicorn/uvicorn) in deployment.
    app.run(host="0.0.0.0", port=5000, debug=False)
