"""ISYARA AI (versi Streamlit) - deteksi alfabet BISINDO real-time.

Jalankan:  streamlit run app.py
Model dicari di streamlit_app/model/ lalu di ../models/ (folder models/ pada repo).
Input model: RGB 224x224, piksel 0-255 (preprocess_input MobileNetV2 sudah ada di dalam model).
"""
import json
import threading
import time
from pathlib import Path

import av
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).resolve().parent
MODEL_FILE = "mobilenetv2_bisindo.keras"
MIN_CONF = 0.5          # di bawah ini huruf diredupkan
ACCENT = (255, 155, 84)

st.set_page_config(page_title="ISYARA AI", page_icon="👨‍🦰", layout="wide")
st.markdown("""<style>
.big{font:800 9rem/1.05 sans-serif;color:#ff9b54;text-align:center}
.big.dim{color:#2d2966}
.pct{font:800 2.8rem/1 sans-serif;color:#ff9b54;text-align:center;margin:6px 0 10px}
.cap{color:#aaa6d0;font-size:.9rem;text-align:center}
h1{color:#ff9b54!important}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    for d in (HERE / "model", HERE.parent / "models"):
        if (d / MODEL_FILE).exists():
            import tensorflow as tf
            m = tf.keras.models.load_model(d / MODEL_FILE)
            labels = json.loads((d / "class_names.json").read_text()) if (d / "class_names.json").exists() \
                else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            m(np.zeros((1, 224, 224, 3), "float32"), training=False)  # pemanasan
            return m, labels
    return None, None


@st.cache_resource
def shared_state():
    """Penghubung antara thread video dan halaman (tetap sama walau skrip dijalankan ulang)."""
    return {"lock": threading.Lock(), "t": 0.0, "letter": None, "score": 0.0}


@st.cache_data
def chart_and_pos():
    return Image.open(HERE / "galeri.jpg").convert("RGB"), json.loads((HERE / "pos.json").read_text())


model, LABELS = load_model()
if model is None:
    st.error("Model tidak ditemukan. Taruh `mobilenetv2_bisindo.keras` di `models/` pada repo "
             "(sejajar dengan folder streamlit_app) atau di `streamlit_app/model/`.")
    st.stop()


def predict(img: Image.Image):
    x = np.asarray(img.convert("RGB").resize((224, 224)), dtype="float32")[None]  # 0-255
    p = model(x, training=False).numpy()[0]
    i = int(p.argmax())
    return LABELS[i], float(p[i])


def center_square(img: Image.Image, frac=1.0):
    w, h = img.size
    s = int(min(w, h) * frac)
    x, y = (w - s) // 2, (h - s) // 2
    return img.crop((x, y, x + s, y + s))


def ref_crop(letter):
    chart, pos = chart_and_pos()
    W, H = chart.size
    x, y, w, h = pos[letter]
    return chart.crop((int(x / 100 * W), int(y / 100 * H), int((x + w) / 100 * W), int((y + h) / 100 * H)))


def make_panel(container):
    with container:
        return {k: st.empty() for k in ("letter", "pct", "bar", "ref")}


def render(panel, letter, score):
    ok = letter is not None and score >= MIN_CONF
    panel["letter"].markdown(f"<div class='big{'' if ok else ' dim'}'>{letter if letter else '?'}</div>",
                             unsafe_allow_html=True)
    panel["pct"].markdown(f"<div class='pct'>{round(score * 100)}%</div>" if letter else "<div class='pct'>–</div>",
                          unsafe_allow_html=True)
    panel["bar"].progress(min(max(score, 0.0), 1.0))
    if ok:
        panel["ref"].image(ref_crop(letter), caption="Isyarat referensi, bandingkan dengan isyarat Anda", width=180)
    else:
        panel["ref"].markdown("<div class='cap'>" + ("Keyakinan rendah, coba atur posisi tangan"
                              if letter else "Menunggu gambar") + "</div>", unsafe_allow_html=True)


# ---------------- tampilan ----------------
st.title("ISYARA AI")
st.write("Deteksi huruf Bahasa Isyarat Indonesia (BISINDO) secara real-time dengan AI dan kamera Anda.")
tab_live, tab_up, tab_gal = st.tabs(["Kamera live", "Unggah gambar", "Galeri huruf A-Z"])

with tab_gal:
    st.image(chart_and_pos()[0], caption="Bagan abjad BISINDO A–Z", width=720)

with tab_up:
    left, right = st.columns([1.3, 1])
    f = left.file_uploader("Pilih gambar isyarat (JPG/PNG)", type=["jpg", "jpeg", "png"])
    panel_up = make_panel(right)
    if f is not None:
        img = Image.open(f).convert("RGB")
        left.image(img, caption="Gambar yang diunggah (model menganalisis bagian tengah gambar)", use_container_width=True)
        letter, score = predict(center_square(img))
        render(panel_up, letter, score)
    else:
        render(panel_up, None, 0.0)

with tab_live:
    st.caption("Klik **Mulai kamera**, izinkan akses kamera, lalu tunjukkan isyarat di dalam kotak oranye.")
    left, right = st.columns([1.3, 1])
    state = shared_state()

    def on_frame(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        h, w, _ = img.shape
        s = int(min(h, w) * 0.84)
        x0, y0 = (w - s) // 2, (h - s) // 2
        now = time.time()
        if now - state["t"] > 0.2:                     # maksimal ±5 prediksi per detik
            state["t"] = now
            letter, score = predict(Image.fromarray(img[y0:y0 + s, x0:x0 + s]))
            with state["lock"]:
                state["letter"], state["score"] = letter, score
        t = 4                                           # gambar kotak panduan
        for sl in (img[y0:y0 + t, x0:x0 + s], img[y0 + s - t:y0 + s, x0:x0 + s],
                   img[y0:y0 + s, x0:x0 + t], img[y0:y0 + s, x0 + s - t:x0 + s]):
            sl[:] = ACCENT
        return av.VideoFrame.from_ndarray(img, format="rgb24")

    with left:
        ctx = webrtc_streamer(
            key="isyara", mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=on_frame, async_processing=True,
            translations={"start": "Mulai kamera", "stop": "Hentikan", "select_device": "Pilih kamera",
                          "device_ask_permission": "Izinkan akses kamera di browser",
                          "device_not_available": "Kamera tidak ditemukan",
                          "device_access_denied": "Akses kamera ditolak. Izinkan kamera lalu muat ulang halaman."})
    panel = make_panel(right)
    render(panel, None, 0.0)

    last = None
    while ctx.state.playing:                            # perbarui hasil secara langsung
        with state["lock"]:
            letter, score = state["letter"], state["score"]
        if letter is not None and (letter, round(score, 2)) != last:
            render(panel, letter, score)
            last = (letter, round(score, 2))
        time.sleep(0.15)
