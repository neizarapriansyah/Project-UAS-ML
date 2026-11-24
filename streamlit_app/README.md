# ISYARA AI (versi Streamlit)

## Menjalankan
    pip install -r requirements.txt
    streamlit run app.py
Buka alamat yang muncul di terminal (biasanya http://localhost:8501).
Kamera hanya jalan di localhost atau HTTPS.

## Struktur yang diharapkan
    Project-UAS-ML/
    ├── models/            <- berisi mobilenetv2_bisindo.keras dan class_names.json
    └── streamlit_app/     <- folder ini

Model dicari di `streamlit_app/model/` lalu di `../models/`.
Input model: RGB 224x224, piksel 0-255 (preprocess_input sudah ada di dalam model, jangan dinormalisasi lagi).

## Catatan
- Kamera live memakai `streamlit-webrtc`. Saat di-deploy ke server publik (mis. Streamlit Cloud),
  koneksi video kadang butuh server TURN selain STUN bawaan.
- Tema warna ada di `.streamlit/config.toml`.
