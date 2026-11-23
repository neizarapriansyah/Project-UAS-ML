"""ISYARA AI - backend Flask.

Browser mengirim frame webcam (JPEG base64) ke /api/predict.
Model: models/mobilenetv2_bisindo.keras (dari notebook bisindo_klasifikasi.ipynb).
Input model: RGB 224x224 dengan piksel 0-255. Jangan dibagi 255 atau dinormalisasi
lagi di sini, karena preprocess_input MobileNetV2 sudah ada di dalam model.
"""
import base64
import io
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
# model dicari di web/model/ lalu di ../models/ (folder models/ pada repo)
MODEL_DIR = next((d for d in (os.path.join(BASE, "model"), os.path.join(BASE, "..", "models"))
                  if os.path.exists(os.path.join(d, "mobilenetv2_bisindo.keras"))),
                 os.path.join(BASE, "..", "models"))
MODEL_FILE = "mobilenetv2_bisindo.keras"
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMG_SIZE = 224

app = Flask(__name__)
model = None


def load_model():
    global model, LABELS, IMG_SIZE
    names = os.path.join(MODEL_DIR, "class_names.json")
    if os.path.exists(names):
        LABELS = json.load(open(names))
    info = os.path.join(MODEL_DIR, "model_info.json")
    if os.path.exists(info):
        IMG_SIZE = json.load(open(info)).get("input_size", IMG_SIZE)
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.exists(path):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), "float32"), training=False)  # pemanasan


def predict_proba(img):
    if model is not None:
        x = np.asarray(img, dtype="float32")[None]  # 0-255, tanpa normalisasi tambahan
        return model(x, training=False).numpy()[0]
    # Mode demo (model tidak ditemukan): huruf dari warna rata-rata gambar
    idx = int(np.asarray(img.resize((4, 4)), dtype="float32").sum()) % len(LABELS)
    p = np.full(len(LABELS), 0.2 / (len(LABELS) - 1))
    p[idx] = 0.8
    return p


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = (request.get_json(silent=True) or {}).get("image", "")
    try:
        raw = base64.b64decode(data.split(",", 1)[-1])
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    except Exception:
        return jsonify(ok=False, error="Gambar tidak valid."), 400
    try:
        proba = predict_proba(img)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    order = np.argsort(proba)[::-1][:3]
    top = [{"label": LABELS[i], "score": round(float(proba[i]), 4)} for i in order]
    return jsonify(ok=True, demo=(model is None), top=top)


load_model()

if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
