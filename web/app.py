"""ISYARA AI - backend Flask.

Browser mengirim frame webcam atau gambar unggahan (JPEG base64) ke /api/predict.
Model: models/mobilenetv2_bisindo.keras (dari notebook bisindo_klasifikasi.ipynb).
Input model: RGB 224x224 dengan piksel 0-255. Jangan dibagi 255 atau dinormalisasi
lagi di sini, karena preprocess_input MobileNetV2 sudah ada di dalam model.

Cara menjalankan (dari folder web/):
    python app.py
lalu buka http://127.0.0.1:5000 di browser.
"""
import base64
import io
import json
import os
import sys
import threading
import traceback

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "mobilenetv2_bisindo.keras"
# model dicari di web/model/ lalu di ../models/ (folder models/ pada repo)
CANDIDATE_DIRS = [os.path.join(BASE, "model"), os.path.normpath(os.path.join(BASE, "..", "models"))]
MODEL_DIR = next((d for d in CANDIDATE_DIRS if os.path.exists(os.path.join(d, MODEL_FILE))), CANDIDATE_DIRS[-1])

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMG_SIZE = 224

app = Flask(__name__)
model = None
model_error = None            # alasan kalau model gagal dimuat, ditampilkan di web
model_lock = threading.Lock()  # satu prediksi dalam satu waktu


def load_model():
    global model, model_error, LABELS, IMG_SIZE
    names = os.path.join(MODEL_DIR, "class_names.json")
    if os.path.exists(names):
        with open(names) as f:
            data = json.load(f)
        # mendukung format lama (dict berisi class_names) dan format baru (list)
        LABELS = data["class_names"] if isinstance(data, dict) else data
    info = os.path.join(MODEL_DIR, "model_info.json")
    if os.path.exists(info):
        with open(info) as f:
            IMG_SIZE = json.load(f).get("input_size", IMG_SIZE)

    path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(path):
        model_error = f"file {MODEL_FILE} tidak ditemukan di {CANDIDATE_DIRS}"
        return
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
    except ImportError:
        model_error = (f"TensorFlow belum terpasang untuk Python {sys.version.split()[0]}. "
                       "TensorFlow belum mendukung Python 3.14, pakai Python 3.10 sampai 3.13 "
                       "lalu jalankan pip install -r requirements.txt")
        return
    try:
        model = tf.keras.models.load_model(path)
        model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), "float32"), training=False)  # pemanasan
        if model.output_shape[-1] != len(LABELS):
            model_error = f"jumlah output model ({model.output_shape[-1]}) tidak sama dengan jumlah label ({len(LABELS)})"
            model = None
    except Exception as e:
        model = None
        model_error = f"model gagal dimuat: {e}"
        traceback.print_exc()


def predict_proba(img):
    x = np.asarray(img, dtype="float32")[None]  # 0-255, tanpa normalisasi tambahan
    with model_lock:
        return model(x, training=False).numpy()[0]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(model_loaded=model is not None, error=model_error,
                   model_dir=MODEL_DIR, input_size=IMG_SIZE, labels=len(LABELS))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify(ok=False, error="Model belum siap: " + (model_error or "tidak diketahui")), 503

    data = (request.get_json(silent=True) or {}).get("image", "")
    try:
        raw = base64.b64decode(data.split(",", 1)[-1])
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    except Exception:
        return jsonify(ok=False, error="Gambar tidak valid."), 400
    try:
        proba = predict_proba(img)
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=f"Prediksi gagal: {e}"), 500

    order = np.argsort(proba)[::-1][:3]
    top = [{"label": LABELS[i], "score": round(float(proba[i]), 4)} for i in order]
    return jsonify(ok=True, top=top)


load_model()
print("Folder model :", MODEL_DIR)
print("Status model :", "siap" if model is not None else "GAGAL, " + str(model_error))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)
