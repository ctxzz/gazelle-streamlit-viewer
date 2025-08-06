# Gazelle Streamlit Viewer

Gazelle（視線推定） + Streamlitによる動画解析ビューア

---

## セットアップ手順

### 1. サブモジュール初期化 & クローン

```bash
git clone --recursive https://github.com/ctxzz/gazelle-streamlit-viewer.git
cd gazelle-streamlit-viewer
git submodule update --init --recursive
```
**※サブモジュールとして Gazelle 本体も取得されます**

### 2. 依存パッケージのインストール

```bash
uv sync
```
> `uv` が未インストールの場合は `pip install uv` でインストールしてください。

### 3. モデルチェックポイントの配置

`checkpoints/` ディレクトリ（リポジトリ直下）にGazelleの事前学習済みモデル（例: `gazelle_dinov2_vitl14_inout.pt`）を配置してください。

### 4. Streamlitアプリ起動

```bash
uv run streamlit run streamlit_app.py
```

---

## ディレクトリ構成例

```
gazelle-streamlit-viewer/
├── gazelle/           # Gazelle本体 (サブモジュール)
│   ├── __init__.py
│   └── gazelle/
│       └── __init__.py
├── streamlit_app.py   # Streamlitアプリ本体
├── pyproject.toml
├── uv.lock
├── .gitmodules
├── .gitignore
├── README.md
└── checkpoints/       # チェックポイント配置用ディレクトリ
```

---

## 注意・補足

- **gazelle/ および gazelle/gazelle/ には空の `__init__.py` ファイルを必ず設置してください。**
    - これにより Python のパッケージ認識やインポートエラーを防げます。
    - サブモジュールの更新等で消えてしまった場合は、再度作成してください。
    - 例:
      ```bash
      touch gazelle/__init__.py gazelle/gazelle/__init__.py
      ```
- **動画解析やヒートマップ生成動画の出力は [webm形式](https://en.wikipedia.org/wiki/WebM) です。多くのブラウザやVLC等で再生できます。**
- モデルやサブモジュールの詳細については `gazelle/README.md` も参照してください。
- Windows/Mac/Linux いずれでも動作します（Python/Streamlitの実行環境があること）。
- 動画サイズや長さによっては処理時間がかかる場合があります。

---

## ライセンス

gazelle本体のライセンス条件に従ってください。
