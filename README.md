# Gazelle Streamlit Viewer

Gazelle（視線推定） + Streamlitによる動画解析Viewer

---

## セットアップ手順

### 1. サブモジュール初期化

```bash
git clone --recursive git@github.com:ctxzz/gazelle-streamlit-viewer.git
cd gazelle-streamlit-viewer
# 既存クローンの場合
git submodule update --init --recursive
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
pip install streamlit opencv-python Pillow matplotlib
```

### 3. モデルチェックポイントの配置

`checkpoints/` ディレクトリに [Gazelleの事前学習済みモデル](https://github.com/fkryan/gazelle#pretrained-models)（例: gazelle_dinov2_vitl14_inout.pt）を配置してください。

### 4. Streamlitアプリ起動

```bash
streamlit run streamlit_app.py
```

---

## ディレクトリ構成例

```
gazelle-streamlit-viewer/
├── gazelle/           # Gazelle本体 (サブモジュール)
├── streamlit_app.py   # Streamlit Viewer
├── requirements.txt
├── .gitmodules
├── .gitignore
└── README.md
```
