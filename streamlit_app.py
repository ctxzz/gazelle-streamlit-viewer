import streamlit as st
import tempfile
import os
import sys
import cv2
import torch
from PIL import Image

# Gazelle本体のimport
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gazelle"))
from gazelle.model import get_gazelle_model

st.title("Gazelle 動画視線解析Viewer")

@st.cache_resource
def load_model(model_name, ckpt_path):
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    return model, transform

model_name = st.selectbox("Model", [
    "gazelle_dinov2_vitl14_inout",
    "gazelle_dinov2_vitb14_inout"
])
ckpt_path = st.text_input("Checkpoint Path", "checkpoints/gazelle_dinov2_vitl14_inout.pt")
if ckpt_path and os.path.exists(ckpt_path):
    model, transform = load_model(model_name, ckpt_path)
else:
    model, transform = None, None

uploaded_file = st.file_uploader("動画アップロード (mp4/avi/mov)", type=["mp4", "avi", "mov"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    st.video(video_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_interval = st.slider("フレーム間隔", 1, 30, 10)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            frames.append(pil_img)
        idx += 1
    cap.release()
    st.write(f"抽出フレーム数: {len(frames)}")

    if st.button("解析開始") and model is not None:
        st.write("解析中...")
        results = []
        for i, frame in enumerate(frames):
            width, height = frame.size
            bbox_norm = [0, 0, 1, 1]
            input_img = transform(frame).unsqueeze(0)
            pred = model.forward({"images": input_img, "bboxes": [bbox_norm]})
            results.append(pred)
        st.write("解析完了")

        for i, pred in enumerate(results):
            st.write(f"Frame {i}")
            if "heatmap" in pred:
                import numpy as np
                import matplotlib.pyplot as plt
                heatmap = pred["heatmap"][0][0].cpu().numpy()
                st.image(heatmap, caption=f"Heatmap {i}", use_column_width=True)
            if "gazex" in pred and "gazey" in pred:
                st.write(f"Gaze: ({pred['gazex']}, {pred['gazey']})")
