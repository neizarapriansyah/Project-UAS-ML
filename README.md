# ISYARA AI

Website deteksi alfabet BISINDO real-time (Flask + webcam + MobileNetV2).

## Menjalankan
    pip install -r requirements.txt
    python app.py
Buka http://127.0.0.1:5000 (kamera hanya jalan di localhost atau HTTPS).

## Model
Backend mencari model di `web/model/`, lalu di `../models/` (folder `models/` pada repo, jadi tidak perlu disalin). Isinya `mobilenetv2_bisindo.keras`, `class_names.json`, `model_info.json`
(hasil `bisindo_klasifikasi.ipynb`). Input: RGB 224x224, piksel 0-255.
`preprocess_input` MobileNetV2 sudah ada DI DALAM model, jadi backend tidak boleh
membagi 255 atau menskalakan ke [-1, 1] (akurasi turun ke ~4% kalau dilakukan).
Mengganti model: timpa file di `model/` dan sesuaikan `MODEL_FILE` di `app.py`.

## API
POST /api/predict  {"image": "data:image/jpeg;base64,..."}
-> {"ok": true, "demo": false, "top": [{"label":"A","score":0.93}, ...]}
