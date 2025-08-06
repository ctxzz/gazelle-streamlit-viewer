import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
import cv2
import os
import numpy as np
import tempfile

# gazelle/gazelle をパッケージとしてimportできるようにする
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "gazelle"))
from gazelle.model import get_gazelle_model

st.set_page_config(layout="wide")
st.title("GazeLLE Streamlit Viewer")

ckpt_dir = "checkpoints"
ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
if not ckpt_files:
    st.error("checkpointsディレクトリに.ptファイルがありません。公式リリースからモデルをダウンロードして配置してください。")
    st.stop()

model_names = [os.path.splitext(f)[0] for f in ckpt_files]
model_to_ckpt = dict(zip(model_names, ckpt_files))
model_name = None
ckpt_path = None

# 2カラムレイアウト
left, right = st.columns([1, 2])

with left:
    model_name = st.selectbox("モデル（チェックポイント）を選択", model_names)
    ckpt_path = os.path.join(ckpt_dir, model_to_ckpt[model_name])
    st.write(f"選択モデル: {model_name}")
    st.write(f"チェックポイントパス: {ckpt_path}")

    @st.cache_resource
    def load_model(model_name, ckpt_path):
        model, transform = get_gazelle_model(model_name)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_gazelle_state_dict(state_dict)
        model.eval()
        return model, transform

    model, transform = load_model(model_name, ckpt_path)

    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "frames" not in st.session_state:
        st.session_state["frames"] = None
    if "video_path" not in st.session_state:
        st.session_state["video_path"] = None

    uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "avi", "mov"])
    frame_interval = st.number_input("フレーム抽出間隔（フレーム数毎）", min_value=1, value=30, step=1)
    run_analysis = False

    if uploaded_file is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path = temp_video.name
        temp_video.write(uploaded_file.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video_path)
        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    pil_img = Image.fromarray(frame_rgb)
                    if isinstance(pil_img, Image.Image):
                        frames.append(pil_img)
                except Exception as e:
                    st.warning(f"フレーム{count}のPIL変換に失敗: {e}")
            count += 1
        cap.release()
        st.write(f"抽出フレーム数: {len(frames)}")

        if frames:
            st.session_state["frames"] = frames
            run_analysis = st.button("解析開始")
        else:
            st.warning("フレームが抽出できませんでした。")
            st.session_state["frames"] = None
            st.session_state["results"] = None
            st.session_state["video_path"] = None

    # 解析開始ボタンで一連の処理を自動実行
    if st.session_state.get("frames") and run_analysis:
        frames = st.session_state["frames"]
        st.write("解析中...")
        results = []
        for i, frame in enumerate(frames):
            try:
                width, height = frame.size
                input_img = transform(frame).unsqueeze(0)
                bbox_norm = [(0, 0, 1, 1)]
                input_dict = {
                    "images": input_img,
                    "bboxes": [bbox_norm]
                }
            except Exception as e:
                st.error(f"Frame {i} 前処理エラー: {e}")
                continue

            try:
                pred = model(input_dict)
                results.append(pred)
            except Exception as e:
                st.error(f"Frame {i} model呼び出しエラー: {e}")
        st.write("解析完了")
        st.session_state["results"] = results

        # 動画生成
        st.write("ヒートマップ重畳動画を生成中...")
        temp_out_video = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        out_path = temp_out_video.name
        temp_out_video.close()
        fourcc = cv2.VideoWriter_fourcc(*'VP80')  # webm
        out = cv2.VideoWriter(out_path, fourcc, 10, (1280, 720))
        for i, (frame, pred) in enumerate(zip(frames, results)):
            frame_np = np.array(frame)
            if frame_np.shape[2] == 4:
                frame_np = frame_np[:, :, :3]
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            heatmap = pred["heatmap"][0].detach().cpu().numpy().squeeze()
            if np.max(heatmap) > 0:
                heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
            else:
                heatmap = (heatmap * 255).astype(np.uint8)
            heatmap_resized = cv2.resize(heatmap, (1280, 720), interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame_bgr, 0.7, heatmap_color, 0.3, 0)
            out.write(overlay)
        out.release()
        st.session_state["video_path"] = out_path

with right:
    # 結果の表示
    if st.session_state.get("video_path") and os.path.exists(st.session_state["video_path"]):
        with open(st.session_state["video_path"], "rb") as f:
            video_bytes = f.read()
        st.success("動画生成・再生が完了しました！")
        st.video(video_bytes, format="video/webm")
        st.download_button("ヒートマップ重畳動画をダウンロード", video_bytes, file_name="output_with_heatmap.webm")
    else:
        st.info("解析と動画生成が完了するとここに動画が表示されます。")