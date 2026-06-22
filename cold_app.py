"""
Demo Skripsi: Sistem Rekomendasi Musik
Berbasis Neural Collaborative Filtering (NeuMF) + K-Means Clustering

Aplikasi ini mendemonstrasikan:
- Pemilihan pengguna dan riwayat interaksi
- Pipeline rekomendasi (flowchart + tabel fitur audio)
- Top-10 rekomendasi dengan evaluasi HIT/MISS
- Penjelasan metrik HR@10 dan NDCG@10
- Perbandingan 4 skenario eksperimen
"""

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="MusicRec Demo — NeuMF + K-Means",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #1DB954 0%, #191414 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Hit/Miss banners */
    .hit-banner {
        background: linear-gradient(90deg, #1DB954, #17a248);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(29,185,84,.35);
    }
    .miss-banner {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(231,76,60,.35);
    }

    /* Song card */
    .song-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .song-card:hover { border-color: #1DB954; }
    .song-card-positive {
        border: 2px solid #1DB954 !important;
        box-shadow: 0 0 12px rgba(29,185,84,.25);
    }

    /* Flowchart nodes (via HTML) */
    .flow-node-green  { background:#1DB954; color:#fff; padding:10px 16px; border-radius:8px;
                        font-weight:600; display:inline-block; margin:4px; }
    .flow-node-blue   { background:#3498db; color:#fff; padding:10px 16px; border-radius:8px;
                        font-weight:600; display:inline-block; margin:4px; }
    .flow-node-orange { background:#e67e22; color:#fff; padding:10px 16px; border-radius:8px;
                        font-weight:600; display:inline-block; margin:4px; }
    .flow-arrow       { font-size:1.5rem; color:#888; display:inline-block; margin:0 4px; vertical-align:middle; }
    .flow-row         { display:flex; align-items:center; justify-content:center;
                        flex-wrap:wrap; gap:4px; margin: 8px 0; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Sidebar */
    .sidebar-model-card {
        background: linear-gradient(135deg, #1e1e2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .sidebar-footer {
        color: #888;
        font-size: 0.78rem;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# MODEL ARCHITECTURE
# ==================================================

class NeuMFCluster(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        n_clusters,
        emb_dim=64,
        cluster_dim=16,
        dropout=0.2
    ):
        super().__init__()

        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)

        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)
        self.cluster_emb = nn.Embedding(n_clusters, cluster_dim)

        mlp_input_dim = emb_dim * 2 + cluster_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.output = nn.Linear(emb_dim + 32, 1)

    def forward(self, user, item, cluster):
        gmf = self.user_gmf(user) * self.item_gmf(item)

        mlp_in = torch.cat(
            [
                self.user_mlp(user),
                self.item_mlp(item),
                self.cluster_emb(cluster)
            ],
            dim=-1
        )

        x = torch.cat([gmf, self.mlp(mlp_in)], dim=-1)
        return self.output(x).squeeze()


class NeuMFBaseline(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, dropout=0.2):
        super().__init__()

        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)

        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.output = nn.Linear(emb_dim + 32, 1)

    def forward(self, user, item):
        gmf = self.user_gmf(user) * self.item_gmf(item)
        mlp_in = torch.cat([self.user_mlp(user), self.item_mlp(item)], dim=-1)
        x = torch.cat([gmf, self.mlp(mlp_in)], dim=-1)
        return self.output(x).squeeze()


# ==================================================
# PATH CONFIG
# ==================================================

DATA_DIR    = Path("data")
MODEL_DIR   = Path("models")
ENCODER_DIR = DATA_DIR / "pkl"

CLUSTER_PATH = Path("testing_code/data/item_cluster_pca3_k3_grid.csv")

# The 5 audio features selected for this project (confirmed from item_dataset.csv columns)
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness"
]


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_item_row(item_id_enc, item_df):
    row = item_df[item_df["item_id_enc"] == int(item_id_enc)]
    if row.empty:
        return None
    return row.iloc[0]


def get_text(row, column, default=""):
    if row is None:
        return default
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def get_cover_url(row):
    if row is None:
        return None
    cover = None
    if "cover_url_md" in row and pd.notna(row["cover_url_md"]):
        cover = row["cover_url_md"]
    elif "cover_url" in row and pd.notna(row["cover_url"]):
        cover = row["cover_url"]
    if cover is None:
        return None
    cover = str(cover)
    if cover.startswith("http"):
        return cover
    return None


def get_spotify_url(row):
    if row is None:
        return None
    if "spotify_url" not in row:
        return None
    url = row["spotify_url"]
    if pd.isna(url):
        return None
    url = str(url)
    if url.startswith("http"):
        return url
    return None


def get_cluster(row):
    if row is None:
        return 0
    if "cluster" not in row:
        return 0
    if pd.isna(row["cluster"]):
        return 0
    return int(row["cluster"])


def model_display_name(model_name):
    label_map = {
        "skenario4_cluster_5f_tpe": "Skenario 4 — NeuMF + Cluster + TPE (Final)",
        "skenario4_cluster_tpe":    "Skenario 4 — NeuMF + Cluster + TPE (Final)",
        "skenario2_cluster_5f":     "Skenario 2 — NeuMF + Cluster",
        "skenario2_cluster":        "Skenario 2 — NeuMF + Cluster",
        "skenario3_baseline_tpe":   "Skenario 3 — NeuMF + TPE",
        "skenario1_baseline":       "Skenario 1 — NeuMF Baseline"
    }
    return label_map.get(model_name, model_name)


def scenario_index_from_name(model_name):
    """Return which row index (0-based) in the scenario table corresponds to the active model."""
    mapping = {
        "skenario1_baseline":       0,
        "skenario2_cluster":        1,
        "skenario2_cluster_5f":     1,
        "skenario3_baseline_tpe":   2,
        "skenario4_cluster_tpe":    3,
        "skenario4_cluster_5f_tpe": 3,
    }
    return mapping.get(model_name, -1)


# ==================================================
# LOAD DATA AND MODEL
# ==================================================

@st.cache_resource(show_spinner=False)
def load_all():
    user_encoder_path = ENCODER_DIR / "user_encoder.pkl"
    item_encoder_path = ENCODER_DIR / "item_encoder.pkl"

    if not user_encoder_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {user_encoder_path}")
    if not item_encoder_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {item_encoder_path}")

    with open(user_encoder_path, "rb") as f:
        user_enc = pickle.load(f)
    with open(item_encoder_path, "rb") as f:
        item_enc = pickle.load(f)

    full_df_path  = DATA_DIR / "user_dataset_final.csv"
    train_df_path = DATA_DIR / "train_dataset.csv"
    test_df_path  = DATA_DIR / "test_dataset.csv"

    if not full_df_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {full_df_path}")
    if not train_df_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {train_df_path}")
    if not test_df_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {test_df_path}")

    full_df  = pd.read_csv(full_df_path)
    train_df = pd.read_csv(train_df_path)
    test_df  = pd.read_csv(test_df_path)

    item_5f_path      = DATA_DIR / "item_dataset_5f.csv"
    item_default_path = DATA_DIR / "item_dataset.csv"

    if item_5f_path.exists():
        item_df = pd.read_csv(item_5f_path)
    elif item_default_path.exists():
        item_df = pd.read_csv(item_default_path)
    else:
        raise FileNotFoundError("item_dataset_5f.csv atau item_dataset.csv tidak ditemukan.")

    if not CLUSTER_PATH.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {CLUSTER_PATH}")

    cluster_df = pd.read_csv(CLUSTER_PATH)

    track_to_enc = dict(
        zip(
            item_enc.classes_,
            item_enc.transform(item_enc.classes_)
        )
    )

    cluster_df["item_id_enc"] = cluster_df["track_id"].map(track_to_enc)
    cluster_df = cluster_df.dropna(subset=["item_id_enc"]).copy()
    cluster_df["item_id_enc"] = cluster_df["item_id_enc"].astype(int)

    unique_clusters = sorted(cluster_df["cluster"].unique())
    cluster_map = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    cluster_df["cluster"] = cluster_df["cluster"].map(cluster_map)

    item_cluster = dict(
        zip(
            cluster_df["item_id_enc"],
            cluster_df["cluster"]
        )
    )

    n_clusters = len(unique_clusters)

    item_df["item_id_enc"] = item_df["track_id"].map(track_to_enc)
    item_df["cluster"]     = item_df["item_id_enc"].map(item_cluster)

    if "cover_url_md" not in item_df.columns:
        if "cover_url" in item_df.columns:
            item_df["cover_url_md"] = item_df["cover_url"]
        else:
            item_df["cover_url_md"] = None

    n_users = int(full_df["user_id_enc"].max()) + 1
    n_items = int(full_df["item_id_enc"].max()) + 1

    model      = None
    model_info = {
        "name":         None,
        "uses_cluster": True,
        "hr10":         None,
        "ndcg10":       None
    }

    model_candidates = [
        (MODEL_DIR / "skenario4_cluster_5f_tpe.pt", True),
        (MODEL_DIR / "skenario4_cluster_tpe.pt",    True),
        (MODEL_DIR / "skenario2_cluster_5f.pt",     True),
        (MODEL_DIR / "skenario2_cluster.pt",         True),
        (MODEL_DIR / "skenario3_baseline_tpe.pt",   False),
        (MODEL_DIR / "skenario1_baseline.pt",       False)
    ]

    for model_path, uses_cluster in model_candidates:
        if not model_path.exists():
            continue

        ckpt   = torch.load(model_path, map_location="cpu")
        params = ckpt.get("best_params", {})

        if uses_cluster:
            model = NeuMFCluster(
                n_users=n_users,
                n_items=n_items,
                n_clusters=n_clusters,
                emb_dim=params.get("emb_dim", 64),
                cluster_dim=params.get("cluster_dim", 16),
                dropout=params.get("dropout", 0.2)
            )
        else:
            model = NeuMFBaseline(
                n_users=n_users,
                n_items=n_items,
                emb_dim=params.get("emb_dim", 64),
                dropout=params.get("dropout", 0.2)
            )

        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        model_info["name"]         = model_path.stem
        model_info["uses_cluster"] = uses_cluster
        model_info["hr10"]         = ckpt.get("HR@10_mean", None)
        model_info["ndcg10"]       = ckpt.get("NDCG@10_mean", None)

        break

    if model is None:
        raise FileNotFoundError("Tidak ada file model .pt yang cocok di folder models.")

    train_positive = (
        train_df[train_df["label"] == 1]
        .groupby("user_id_enc")["item_id_enc"]
        .apply(set)
        .to_dict()
    )

    train_counts = (
        train_df[train_df["label"] == 1]
        .groupby("user_id_enc")["item_id_enc"]
        .count()
        .reset_index(name="n_interactions")
    )

    test_positive_users = set(
        test_df[test_df["label"] == 1]["user_id_enc"].astype(int).unique()
    )

    user_df = train_counts[
        train_counts["user_id_enc"].isin(test_positive_users)
    ].copy()

    user_df["username"] = user_df["user_id_enc"].apply(
        lambda x: (
            user_enc.inverse_transform([int(x)])[0]
            if int(x) < len(user_enc.classes_)
            else f"user_{int(x)}"
        )
    )

    user_df = user_df.sort_values("n_interactions", ascending=False)

    cluster_labels = {
        0: "Cluster 0 — Energik & Dance",
        1: "Cluster 1 — Akustik & Melankolis",
        2: "Cluster 2 — Upbeat & Pop"
    }

    # Determine which audio features actually exist in item_df
    available_audio_features = [f for f in AUDIO_FEATURES if f in item_df.columns]

    return {
        "model":                   model,
        "model_info":              model_info,
        "user_enc":                user_enc,
        "item_enc":                item_enc,
        "full_df":                 full_df,
        "train_df":                train_df,
        "test_df":                 test_df,
        "item_df":                 item_df,
        "item_cluster":            item_cluster,
        "n_clusters":              n_clusters,
        "n_users":                 n_users,
        "n_items":                 n_items,
        "train_positive":          train_positive,
        "user_df":                 user_df,
        "cluster_labels":          cluster_labels,
        "available_audio_features": available_audio_features
    }


# ==================================================
# SINGLE USER HIT/MISS EVALUATION
# ==================================================

@torch.no_grad()
def run_single_user_hit_miss(
    user_id_enc,
    data,
    seed=42,
    num_neg=99,
    top_k=10
):
    model         = data["model"]
    model_info    = data["model_info"]
    test_df       = data["test_df"]
    item_df       = data["item_df"]
    item_cluster  = data["item_cluster"]
    train_positive = data["train_positive"]
    n_items       = data["n_items"]

    user_id_enc = int(user_id_enc)

    test_pos_user = test_df[
        (test_df["user_id_enc"].astype(int) == user_id_enc) &
        (test_df["label"] == 1)
    ].copy()

    if test_pos_user.empty:
        return None, None

    true_item = int(test_pos_user.iloc[0]["item_id_enc"])

    user_train_positive = train_positive.get(user_id_enc, set())

    negative_candidates = [
        item_id
        for item_id in range(n_items)
        if item_id != true_item and item_id not in user_train_positive
    ]

    if len(negative_candidates) == 0:
        return None, None

    rng      = np.random.default_rng(seed)
    n_sample = min(num_neg, len(negative_candidates))
    negatives = rng.choice(negative_candidates, size=n_sample, replace=False)

    items_eval = np.array(list(negatives) + [true_item], dtype=int)

    users_t = torch.full((len(items_eval),), user_id_enc, dtype=torch.long)
    items_t = torch.tensor(items_eval, dtype=torch.long)

    if model_info["uses_cluster"]:
        clusters_t = torch.tensor(
            [item_cluster.get(int(item_id), 0) for item_id in items_eval],
            dtype=torch.long
        )
        raw_scores = model(users_t, items_t, clusters_t)
    else:
        raw_scores = model(users_t, items_t)

    scores     = torch.sigmoid(raw_scores).cpu().numpy()
    sorted_idx = np.argsort(scores)[::-1]
    ranked_items  = items_eval[sorted_idx]
    ranked_scores = scores[sorted_idx]

    true_rank = int(np.where(ranked_items == true_item)[0][0]) + 1
    hit       = true_rank <= top_k

    true_row = get_item_row(true_item, item_df)

    summary = {
        "user_id_enc":   user_id_enc,
        "true_item_id":  true_item,
        "true_title":    get_text(true_row, "track_name",    f"item_{true_item}"),
        "true_artist":   get_text(true_row, "artist_names",  ""),
        "true_cluster":  get_cluster(true_row),
        "true_score":    float(scores[np.where(items_eval == true_item)[0][0]]),
        "true_rank":     true_rank,
        "hit":           hit,
        "status":        "HIT" if hit else "MISS",
        "num_candidates": len(items_eval),
        "num_negatives": n_sample,
        "top_k":         top_k
    }

    rows = []
    for rank, item_id in enumerate(ranked_items[:top_k], start=1):
        item_id = int(item_id)
        row     = get_item_row(item_id, item_df)
        rows.append({
            "rank":         rank,
            "item_id_enc":  item_id,
            "track_name":   get_text(row, "track_name",   f"item_{item_id}"),
            "artist_names": get_text(row, "artist_names", ""),
            "cluster":      get_cluster(row),
            "score":        float(ranked_scores[rank - 1]),
            "is_true_item": item_id == true_item,
            "cover_url":    get_cover_url(row),
            "spotify_url":  get_spotify_url(row)
        })

    top10_df = pd.DataFrame(rows)
    return summary, top10_df


# ==================================================
# APP — HEADER
# ==================================================

st.markdown("""
<div class="main-header">
    <h1>🎵 MusicRec Demo</h1>
    <p>Sistem Rekomendasi Musik — Neural Collaborative Filtering (NeuMF) + K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner("⏳ Memuat data dan model…"):
        data = load_all()
except Exception as e:
    st.error(f"❌ Gagal memuat aplikasi: {e}")
    st.stop()

model_info             = data["model_info"]
user_df                = data["user_df"]
cluster_labels         = data["cluster_labels"]
item_df                = data["item_df"]
train_df               = data["train_df"]
available_audio_features = data["available_audio_features"]

if user_df.empty:
    st.error("Tidak ada user yang punya interaksi train dan item positif di test_dataset.")
    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")

    user_options = {
        f"{row['username']} | {int(row['n_interactions'])} interaksi": int(row["user_id_enc"])
        for _, row in user_df.head(1000).iterrows()
    }

    selected_label    = st.selectbox("👤 Pilih User", list(user_options.keys()))
    selected_user     = user_options[selected_label]
    selected_username = selected_label.split(" | ")[0]

    seed = st.selectbox("🎲 Seed negatif", [42, 123, 456, 789], index=0)

    st.divider()

    st.markdown("**🤖 Model Aktif**")
    st.markdown(f"""
    <div class="sidebar-model-card">
        <b>{model_display_name(model_info['name'])}</b><br>
        HR@10 &nbsp;: <code>{'—' if model_info['hr10'] is None else f"{model_info['hr10']:.4f}"}</code><br>
        NDCG@10: <code>{'—' if model_info['ndcg10'] is None else f"{model_info['ndcg10']:.4f}"}</code>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<div class='sidebar-footer'>Sistem Rekomendasi Musik<br>NeuMF + K-Means Cluster</div>",
        unsafe_allow_html=True
    )


# ==================================================
# RUN EVALUATION
# ==================================================

summary, top10_df = run_single_user_hit_miss(
    user_id_enc=selected_user,
    data=data,
    seed=seed,
    num_neg=99,
    top_k=10
)

if summary is None:
    st.error("User ini tidak punya data evaluasi valid. Silakan pilih user lain.")
    st.stop()


# ==================================================
# 5 TABS
# ==================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👤 Pilih Pengguna",
    "🔬 Pipeline Rekomendasi",
    "🎵 Top-10 Rekomendasi",
    "📊 Penjelasan Metrik",
    "🏆 Perbandingan Skenario",
    "🗂️ Visualisasi Cluster"
])


# ──────────────────────────────────────────────────
# TAB 1 — PILIH PENGGUNA
# ──────────────────────────────────────────────────

with tab1:
    st.subheader(f"👤 Profil Pengguna: {selected_username}")

    user_row = user_df[user_df["user_id_enc"] == selected_user].iloc[0]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("User ID (Encoded)", str(selected_user))
    col_b.metric("Username",           selected_username)
    col_c.metric("Total Interaksi Training", int(user_row["n_interactions"]))

    st.divider()

    # Riwayat 10 lagu terakhir dari train_df
    st.subheader("🎧 Riwayat 10 Lagu Terakhir")

    user_train = (
        train_df[
            (train_df["user_id_enc"] == selected_user) &
            (train_df["label"] == 1)
        ]
        .tail(10)
        .copy()
    )

    if user_train.empty:
        st.warning("Tidak ada riwayat training untuk user ini.")
    else:
        # Join dengan item_df untuk mendapat nama lagu, artis, cluster
        merged = user_train.merge(
            item_df[["item_id_enc", "track_name", "artist_names", "cluster"]],
            on="item_id_enc",
            how="left"
        ).reset_index(drop=True)

        merged["Rank"] = range(1, len(merged) + 1)
        merged["Cluster (Angka)"] = merged["cluster"].fillna(0).astype(int)
        merged["Label Cluster"]   = merged["Cluster (Angka)"].map(cluster_labels).fillna("—")

        display_history = merged[[
            "Rank", "track_name", "artist_names", "Cluster (Angka)", "Label Cluster"
        ]].rename(columns={
            "track_name":   "Judul Lagu",
            "artist_names": "Artis"
        })

        st.dataframe(display_history, use_container_width=True, hide_index=True)

        st.caption(
            "🎵 *Riwayat ini digunakan model untuk mempelajari preferensi pengguna. "
            "Setiap lagu direpresentasikan sebagai embedding yang menjadi input NeuMF.*"
        )


# ──────────────────────────────────────────────────
# TAB 2 — PIPELINE REKOMENDASI
# ──────────────────────────────────────────────────

with tab2:
    st.subheader("🔬 Pipeline Rekomendasi")
    st.markdown("Visualisasi alur lengkap dari data mentah hingga rekomendasi Top-10.")

    # ---------- Bagian A: Flowchart ----------
    st.markdown("#### A. Diagram Alur Sistem")

    # Coba graphviz, fallback ke HTML
    try:
        st.graphviz_chart("""
        digraph Pipeline {
            rankdir=TB;
            node [fontname="Arial", style="filled", fontsize=12, margin="0.3,0.15"];

            // Input nodes (hijau)
            DataLagu    [label="Data Lagu\n(Spotify)", fillcolor="#1DB954", fontcolor="white", shape=box];
            UserID      [label="User ID", fillcolor="#1DB954", fontcolor="white", shape=box];

            // Proses nodes (biru)
            FiturAudio  [label="5 Fitur Audio\n(Danceability, Energy,\nValence, Acousticness,\nInstrumentalness)",
                         fillcolor="#3498db", fontcolor="white", shape=box];
            KMeans      [label="K-Means Clustering\n(K=3)", fillcolor="#2980b9", fontcolor="white", shape=box];
            EmbUser     [label="Embedding\nPengguna", fillcolor="#3498db", fontcolor="white", shape=box];
            EmbItem     [label="Embedding\nLagu + Cluster", fillcolor="#3498db", fontcolor="white", shape=box];
            NeuMF       [label="NeuMF Model\n(GMF + MLP)", fillcolor="#8e44ad", fontcolor="white", shape=box];
            Ranking     [label="Ranking\n100 Kandidat", fillcolor="#2980b9", fontcolor="white", shape=box];

            // Label cluster
            LabelCluster [label="Label Cluster\n(0 / 1 / 2)", fillcolor="#e67e22", fontcolor="white", shape=box];

            // Output (oranye)
            Top10       [label="🎵 Top-10\nRekomendasi", fillcolor="#e67e22", fontcolor="white",
                         shape=box, penwidth=2];

            // Edges
            DataLagu    -> FiturAudio;
            FiturAudio  -> KMeans;
            KMeans      -> LabelCluster;
            LabelCluster -> EmbItem;
            UserID      -> EmbUser;
            EmbUser     -> NeuMF;
            EmbItem     -> NeuMF;
            NeuMF       -> Ranking;
            Ranking     -> Top10;
        }
        """, use_container_width=True)

    except Exception:
        # Fallback HTML flowchart
        st.markdown("""
        <div style="background:#1e1e2e; padding:1.5rem; border-radius:12px; text-align:center;">
            <div class="flow-row">
                <span class="flow-node-green">Data Lagu (Spotify)</span>
                <span class="flow-arrow">→</span>
                <span class="flow-node-blue">5 Fitur Audio</span>
                <span class="flow-arrow">→</span>
                <span class="flow-node-blue">K-Means Clustering</span>
                <span class="flow-arrow">→</span>
                <span class="flow-node-blue">Label Cluster (0/1/2)</span>
            </div>
            <div class="flow-row" style="justify-content:center;">
                <span style="font-size:1.5rem; color:#888;">↓</span>
            </div>
            <div class="flow-row">
                <span class="flow-node-green">User ID</span>
                <span class="flow-arrow">→</span>
                <span class="flow-node-blue">Embedding Pengguna</span>
                <span class="flow-arrow">→</span>
                <span class="flow-node-blue" style="background:#8e44ad;">NeuMF Model (GMF+MLP)</span>
                <span class="flow-arrow">←</span>
                <span class="flow-node-blue">Embedding Lagu + Cluster</span>
            </div>
            <div class="flow-row"><span style="font-size:1.5rem;color:#888;">↓</span></div>
            <div class="flow-row">
                <span class="flow-node-blue">Skor Prediksi</span>
                <span class="flow-arrow">→</span>
                <span class="flow-node-blue">Ranking 100 Kandidat</span>
                <span class="flow-arrow">→</span>
                <span class="flow-node-orange">🎵 Top-10 Rekomendasi</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---------- Bagian B: Tabel Fitur Audio ----------
    st.markdown("#### B. Fitur Audio 10 Lagu Teratas")
    st.caption(
        "Tabel ini menunjukkan bahwa setiap lagu memiliki representasi fitur audio dan label cluster "
        "yang menjadi input tambahan ke model NeuMF."
    )

    if top10_df is not None and not top10_df.empty:
        # Build tabel lengkap dengan fitur audio
        rows_pipeline = []
        for _, row in top10_df.iterrows():
            item_row  = get_item_row(int(row["item_id_enc"]), item_df)
            cl_num    = int(row["cluster"])
            cl_label  = cluster_labels.get(cl_num, f"Cluster {cl_num}")
            keterangan = "✅ Item Uji" if bool(row["is_true_item"]) else "—"

            entry = {
                "Rank":          int(row["rank"]),
                "Judul Lagu":    row["track_name"],
                "Artis":         row["artist_names"],
            }
            # Tambahkan fitur audio yang tersedia
            for feat in available_audio_features:
                if item_row is not None and feat in item_row.index and pd.notna(item_row[feat]):
                    entry[feat] = round(float(item_row[feat]), 4)
                else:
                    entry[feat] = None

            entry["Cluster"]       = cl_num
            entry["Label Cluster"] = cl_label
            entry["Skor NCF"]      = round(float(row["score"]), 4)
            entry["Keterangan"]    = keterangan
            rows_pipeline.append(entry)

        pipeline_df = pd.DataFrame(rows_pipeline)
        st.dataframe(pipeline_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Tidak ada data untuk ditampilkan.")


# ──────────────────────────────────────────────────
# TAB 3 — TOP-10 REKOMENDASI
# ──────────────────────────────────────────────────

with tab3:
    st.subheader("🎵 Top-10 Rekomendasi")

    # Banner HIT / MISS
    if summary["hit"]:
        st.markdown(
            f'<div class="hit-banner">✅ HIT — Item positif muncul di posisi #{summary["true_rank"]} dari 100 kandidat!</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="miss-banner">❌ MISS — Item positif tidak muncul di Top-10 (rank #{summary["true_rank"]} dari 100 kandidat)</div>',
            unsafe_allow_html=True
        )

    # 4 metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status",            summary["status"])
    m2.metric("Rank Item Positif", f"#{summary['true_rank']}")
    m3.metric("Jumlah Kandidat",   summary["num_candidates"])
    m4.metric("Top-K",             summary["top_k"])

    st.divider()

    # Kartu lagu satu-per-satu
    for _, row in top10_df.iterrows():
        rank       = int(row["rank"])
        title      = row["track_name"]
        artist     = row["artist_names"]
        cluster_n  = int(row["cluster"])
        score      = float(row["score"])
        is_true    = bool(row["is_true_item"])
        cover_url  = row["cover_url"]
        spotify_url = row["spotify_url"]

        card_class = "song-card song-card-positive" if is_true else "song-card"

        with st.container():
            cols = st.columns([1, 4, 2])

            with cols[0]:
                if cover_url:
                    st.image(cover_url, width=120)
                else:
                    st.markdown("🎵", unsafe_allow_html=False)

            with cols[1]:
                if is_true:
                    st.markdown(f"### #{rank} &nbsp; ✅ ITEM POSITIF UJI")
                else:
                    st.markdown(f"### #{rank}")

                st.write(f"**{title}**")
                st.write(f"Artis: {artist}")
                st.write(f"Cluster: {cluster_n} — {cluster_labels.get(cluster_n, 'Cluster')}")

            with cols[2]:
                st.metric("Skor NCF", f"{score:.6f}")
                if spotify_url:
                    st.link_button("🎧 Spotify", spotify_url)

        st.divider()

    # Tabel ringkasan
    st.subheader("📋 Tabel Ringkasan")

    table_view = top10_df[[
        "rank", "item_id_enc", "track_name", "artist_names",
        "cluster", "score", "is_true_item"
    ]].copy()

    table_view.columns = [
        "Rank", "Item ID Enc", "Judul Lagu", "Artis",
        "Cluster", "Skor NCF", "Item Positif Uji"
    ]

    st.dataframe(table_view, use_container_width=True, hide_index=True)

    # Download
    csv_export = top10_df.copy()
    csv_export["status"]       = summary["status"]
    csv_export["true_item_id"] = summary["true_item_id"]
    csv_export["true_rank"]    = summary["true_rank"]
    csv_export["true_title"]   = summary["true_title"]
    csv_export["true_artist"]  = summary["true_artist"]

    st.download_button(
        label="⬇️ Download Hasil Top-10 (CSV)",
        data=csv_export.to_csv(index=False).encode("utf-8"),
        file_name=f"top10_user{selected_user}_seed{seed}.csv",
        mime="text/csv"
    )


# ──────────────────────────────────────────────────
# TAB 4 — PENJELASAN METRIK EVALUASI
# ──────────────────────────────────────────────────

with tab4:
    st.subheader("📊 Penjelasan Metrik Evaluasi")
    st.caption("Halaman ini menjelaskan HR@10 dan NDCG@10 untuk audiens yang belum familiar.")

    # ---- HR@10 ----
    st.markdown("### A. Hit Rate@10 (HR@10)")

    st.info("""
**HR@10 menjawab pertanyaan:**
> *"Dari semua pengguna, berapa persen yang lagu relevannya muncul di Top-10 rekomendasi?"*

**Cara kerja evaluasi (leave-one-out):**
- Untuk setiap pengguna, ambil **1 lagu** yang benar-benar didengar → **item positif test**
- Campurkan dengan **99 lagu acak** yang tidak pernah didengar → **item negatif**
- Minta model meranking 100 lagu tersebut
- Jika lagu yang benar masuk **Top-10** → **HIT** (nilai = 1)
- Jika tidak masuk → **MISS** (nilai = 0)
- **HR@10 = Σ HIT / total pengguna**

**Contoh:** HR@10 = 0,75 artinya **75% pengguna** mendapat rekomendasi yang relevan di Top-10.
    """)

    # Contoh konkret untuk user ini
    true_title = summary["true_title"]
    true_rank  = summary["true_rank"]
    status     = summary["status"]

    if summary["hit"]:
        st.success(
            f"✅ **User *{selected_username}*** mendapat **HIT** — "
            f"lagu *\"{true_title}\"* berada di rank **#{true_rank}** dari 100 kandidat."
        )
    else:
        st.error(
            f"❌ **User *{selected_username}*** mendapat **MISS** — "
            f"lagu *\"{true_title}\"* berada di rank **#{true_rank}** dari 100 kandidat "
            f"(di luar Top-10)."
        )

    st.divider()

    # ---- NDCG@10 ----
    st.markdown("### B. NDCG@10 (Normalized Discounted Cumulative Gain@10)")

    st.info("""
**NDCG@10 menjawab pertanyaan:**
> *"Seberapa tinggi posisi lagu relevan di dalam Top-10?"*

**Perbedaan dengan HR@10:**
- **HR@10** hanya peduli *APAKAH* lagu muncul di Top-10 (ya/tidak)
- **NDCG@10** peduli *DI POSISI MANA* lagu muncul — rank #1 jauh lebih baik dari rank #10

**Formula:**
- Jika lagu relevan ada di rank **1** → NDCG = 1,000 (sempurna)
- Jika lagu relevan ada di rank **5** → NDCG ≈ 0,387
- Jika lagu relevan ada di rank **10** → NDCG ≈ 0,289
- Jika **tidak masuk** Top-10 → NDCG = 0

**NDCG@10 = rata-rata nilai ini di semua pengguna.**
Semakin mendekati 1,0, semakin baik sistem menempatkan lagu relevan di posisi atas.
    """)

    # Hitung NDCG individual
    if summary["hit"]:
        ndcg_individual = 1.0 / math.log2(true_rank + 1)
        st.success(
            f"📈 Untuk user ini, **NDCG individual = {ndcg_individual:.4f}** "
            f"karena lagu relevan berada di rank **#{true_rank}**."
        )
    else:
        ndcg_individual = 0.0
        st.warning(
            f"📉 Untuk user ini, **NDCG individual = 0,0000** "
            f"karena lagu relevan tidak masuk Top-10 (rank #{true_rank})."
        )

    st.divider()

    # ---- Tabel referensi NDCG ----
    st.markdown("### C. Tabel Referensi Nilai NDCG per Rank")

    ndcg_ref = pd.DataFrame({
        "Rank":       [1, 2, 3, 5, 7, 10, ">10"],
        "Nilai NDCG": [
            f"{1/math.log2(2):.4f}",
            f"{1/math.log2(3):.4f}",
            f"{1/math.log2(4):.4f}",
            f"{1/math.log2(6):.4f}",
            f"{1/math.log2(8):.4f}",
            f"{1/math.log2(11):.4f}",
            "0.0000"
        ],
        "Interpretasi": [
            "Sempurna — lagu relevan di posisi pertama",
            "Sangat baik",
            "Baik",
            "Cukup",
            "Rendah",
            "Sangat rendah",
            "Tidak masuk Top-10"
        ]
    })

    st.dataframe(ndcg_ref, use_container_width=True, hide_index=True)

    st.caption(
        f"Rumus: NDCG = 1 / log₂(rank + 1). "
        f"User *{selected_username}* saat ini berada di rank #{true_rank}, "
        f"sehingga NDCG individual = {ndcg_individual:.4f}."
    )


# ──────────────────────────────────────────────────
# TAB 5 — PERBANDINGAN SKENARIO
# ──────────────────────────────────────────────────

with tab5:
    st.subheader("🏆 Perbandingan Skenario Eksperimen")

    # CATATAN: Isi nilai HR10 dan NDCG10 setelah eksperimen Bab 3 selesai.
    # Saat ini dikosongkan karena hasil belum tersedia.
    scenario_data = {
        "Skenario": [
            "Skenario 1 — NeuMF Baseline",
            "Skenario 2 — NeuMF + Cluster",
            "Skenario 3 — NeuMF + TPE",
            "Skenario 4 — NeuMF + Cluster + TPE (Final)"
        ],
        "HR@10":   [None, None, None, None],   # ISI SETELAH EKSPERIMEN SELESAI
        "NDCG@10": [None, None, None, None]    # ISI SETELAH EKSPERIMEN SELESAI
    }

    active_idx = scenario_index_from_name(model_info["name"])

    # Banner informasi
    st.info(
        "⏳ **Hasil eksperimen sedang dalam proses.** "
        "Tabel akan diperbarui setelah seluruh skenario selesai dijalankan."
    )

    # Buat DataFrame untuk tabel
    scenario_df = pd.DataFrame(scenario_data)

    # Tambahkan kolom Status
    scenario_df["Status"] = [
        "✅ Aktif" if i == active_idx else "⏳ Pending"
        for i in range(len(scenario_df))
    ]

    # Tampilkan nilai None sebagai "—"
    scenario_display = scenario_df.copy()
    scenario_display["HR@10"]   = scenario_display["HR@10"].apply(
        lambda v: "—" if v is None else f"{v:.4f}"
    )
    scenario_display["NDCG@10"] = scenario_display["NDCG@10"].apply(
        lambda v: "—" if v is None else f"{v:.4f}"
    )

    # Kolom urutan tampilan
    scenario_display = scenario_display[["Status", "Skenario", "HR@10", "NDCG@10"]]

    st.dataframe(
        scenario_display,
        use_container_width=True,
        hide_index=True
    )

    # Catatan aktif
    active_name = model_display_name(model_info["name"])
    st.caption(f"**Skenario aktif saat ini:** {active_name}")

    # Placeholder chart
    chart_placeholder = st.empty()
    chart_placeholder.info(
        "📊 Bar chart perbandingan HR@10 dan NDCG@10 akan ditampilkan di sini "
        "setelah semua nilai tersedia. Ganti `None` menjadi nilai float di `scenario_data` "
        "pada baris kode yang bertanda komentar **# ISI SETELAH EKSPERIMEN SELESAI**."
    )

    # Jika sudah ada nilai (semua tidak None), tampilkan chart otomatis
    all_hr_available   = all(v is not None for v in scenario_data["HR@10"])
    all_ndcg_available = all(v is not None for v in scenario_data["NDCG@10"])

    if all_hr_available and all_ndcg_available:
        chart_df = pd.DataFrame({
            "Skenario": [s.split(" — ")[0] for s in scenario_data["Skenario"]],
            "HR@10":    scenario_data["HR@10"],
            "NDCG@10":  scenario_data["NDCG@10"]
        }).set_index("Skenario")
        chart_placeholder.bar_chart(chart_df)

    st.divider()

    # Penjelasan tiap skenario
    with st.expander("📖 Apa perbedaan tiap skenario?", expanded=True):
        st.markdown("""
**Skenario 1 — NeuMF Baseline**
Model NeuMF standar tanpa fitur tambahan apapun.
Digunakan sebagai acuan perbandingan (*baseline*).

---

**Skenario 2 — NeuMF + Cluster**
Model NeuMF dengan tambahan embedding cluster dari K-Means.
Fitur audio lagu dikelompokkan menjadi **3 cluster** menggunakan 5 fitur audio terpilih
(Danceability, Energy, Valence, Acousticness, Instrumentalness),
lalu label cluster dimasukkan sebagai input tambahan ke model.

---

**Skenario 3 — NeuMF + TPE**
Model NeuMF Baseline dengan hyperparameter yang dioptimasi menggunakan
**Tree-structured Parzen Estimator (TPE)** dari library Optuna.
Optimasi dilakukan pada: `emb_dim`, `dropout`, `learning rate`, `batch size`.

---

**Skenario 4 — NeuMF + Cluster + TPE (Final)**
Kombinasi lengkap: embedding cluster K-Means + optimasi hyperparameter TPE.
Ini adalah **model final** yang diajukan dalam skripsi sebagai kontribusi utama.
        """)


# ══════════════════════════════════════════════════
# TAB 6 — VISUALISASI CLUSTER
# ══════════════════════════════════════════════════

with tab6:
    st.subheader("🗂️ Visualisasi K-Means Clustering")
    st.caption(
        "Halaman ini menampilkan hasil K-Means Clustering (K=3) pada fitur audio lagu Spotify, "
        "termasuk plot evaluasi cluster dan dominasi fitur audio per cluster."
    )

    # ─── A. Plot Evaluasi Cluster ───────────────────
    st.markdown("### A. Evaluasi Jumlah Cluster")

    _elbow_path     = Path("elbow_plot.png")
    _silhouette_path_root = Path("silhouette_plot.png")
    _silhouette_path_test = Path("testing_code/silhouette_plot.png")

    _col_e, _col_s = st.columns(2)

    with _col_e:
        st.markdown("**Elbow Plot**")
        if _elbow_path.exists():
            st.image(str(_elbow_path), use_container_width=True,
                     caption="Elbow Method — menentukan K optimal")
        else:
            st.info("📁 File `elbow_plot.png` belum tersedia di root folder.")

    with _col_s:
        st.markdown("**Silhouette Plot**")
        _sil_path = _silhouette_path_root if _silhouette_path_root.exists() else _silhouette_path_test
        if _sil_path.exists():
            st.image(str(_sil_path), use_container_width=True,
                     caption="Silhouette Score — validasi kualitas cluster")
        else:
            st.info("📁 File `silhouette_plot.png` belum tersedia.")

    st.caption(
        "Elbow plot menunjukkan penurunan WCSS semakin landai setelah K=3, "
        "dan Silhouette Score tertinggi juga dicapai pada K=3."
    )

    st.divider()

    # ─── B. Scatter Plot Persebaran Cluster (PCA 2D) ───
    st.markdown("### B. Persebaran Cluster (PCA 2D)")
    st.caption(
        "Setiap titik mewakili satu lagu. Warna menunjukkan cluster K-Means. "
        "Sumbu PCA merangkum 5 fitur audio ke dalam 2 dimensi utama."
    )

    _AUDIO_FEATS = ["danceability", "energy", "valence", "acousticness", "tempo"]

    # Load cluster + item data
    _cluster_csv = Path("outputs/item_cluster_kmeans.csv")
    _item_csv_5f = Path("data/item_dataset_5f.csv")
    _item_csv    = Path("data/item_dataset.csv")

    try:
        _cl_df = pd.read_csv(_cluster_csv)
        _it_df = pd.read_csv(_item_csv_5f if _item_csv_5f.exists() else _item_csv)

        _feats_available = [f for f in _AUDIO_FEATS if f in _it_df.columns]
        _it_sub = _it_df[_feats_available].copy().reset_index(drop=True)
        _cl_sub = _cl_df[["item_id", "cluster_id"]].copy()

        _merged = _cl_sub.copy()
        for _f in _feats_available:
            _merged[_f] = _merged["item_id"].map(_it_sub[_f])

        # ── PCA Scatter Plot ────────────────────────────────
        try:
            import pickle
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import MinMaxScaler

            _X_feat = _merged[_feats_available].dropna().values
            _y_feat = _merged.loc[_merged[_feats_available].notna().all(axis=1), "cluster_id"].values

            # Use saved scaler if available, else fit new
            _scaler_path = Path("scaler_5f.pkl")
            if _scaler_path.exists():
                with open(_scaler_path, "rb") as _sf:
                    _scaler = pickle.load(_sf)
                _X_scaled = _scaler.transform(_X_feat)
            else:
                _scaler = MinMaxScaler()
                _X_scaled = _scaler.fit_transform(_X_feat)

            # PCA 2D
            _pca_path = Path("pca_5f.pkl")
            if _pca_path.exists():
                with open(_pca_path, "rb") as _pf:
                    _pca_model = pickle.load(_pf)
                _X_2d = _pca_model.transform(_X_scaled)[:, :2]
            else:
                _pca_model = PCA(n_components=2, random_state=42)
                _X_2d = _pca_model.fit_transform(_X_scaled)

            # Plot
            _SCATTER_COLORS = {0: "#1DB954", 1: "#3498db", 2: "#e67e22"}
            _fig, _ax = plt.subplots(figsize=(9, 5.5))
            _fig.patch.set_facecolor("#0f172a")
            _ax.set_facecolor("#111827")

            for _cn, _clr in _SCATTER_COLORS.items():
                _mask = _y_feat == _cn
                _ax.scatter(
                    _X_2d[_mask, 0], _X_2d[_mask, 1],
                    c=_clr, alpha=0.35, s=8, label=_CL_LABEL.get(_cn, f"Cluster {_cn}"),
                    edgecolors="none"
                )
                # centroid
                _cx, _cy = _X_2d[_mask, 0].mean(), _X_2d[_mask, 1].mean()
                _ax.scatter(_cx, _cy, c=_clr, s=180, marker="*",
                            edgecolors="white", linewidths=0.8, zorder=5)
                _ax.annotate(
                    f"C{_cn}", (_cx, _cy),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=_clr, fontweight="bold"
                )

            # Variance explained
            _var = getattr(_pca_model, "explained_variance_ratio_", None)
            _xlab = f"PC1 ({_var[0]*100:.1f}% varians)" if _var is not None else "PC1"
            _ylab = f"PC2 ({_var[1]*100:.1f}% varians)" if _var is not None else "PC2"

            _ax.set_xlabel(_xlab, color="#9ca3af", fontsize=10)
            _ax.set_ylabel(_ylab, color="#9ca3af", fontsize=10)
            _ax.set_title("Persebaran Lagu per Cluster (PCA 2D)", color="#f1f5f9",
                          fontsize=12, fontweight="bold", pad=10)
            _ax.tick_params(colors="#6b7280", labelsize=8)
            for _sp in _ax.spines.values():
                _sp.set_color("#1e2a3a")

            _leg = _ax.legend(
                loc="upper right", framealpha=0.25,
                facecolor="#1e2a3a", edgecolor="#374151",
                labelcolor="white", fontsize=8.5
            )

            plt.tight_layout()
            st.pyplot(_fig, use_container_width=True)
            plt.close(_fig)

        except Exception as _pca_err:
            st.warning(f"⚠️ Scatter plot tidak bisa ditampilkan: {_pca_err}")

        st.divider()

        # ── Mean per cluster untuk bar chart ───────────
        st.markdown("### C. Karakteristik Fitur Audio per Cluster")
        st.caption(
            "Rata-rata nilai 5 fitur audio di setiap cluster. "
            "Semakin panjang bar, semakin dominan fitur tersebut."
        )

        _means = (
            _merged.groupby("cluster_id")[_feats_available]
            .mean()
            .round(4)
            .reset_index()
        )
        _CL_COLORS = {0: "#1DB954", 1: "#3498db", 2: "#e67e22"}
        _CL_LABEL  = {
            0: "Cluster 0 — Pop & Mainstream",
            1: "Cluster 1 — High Energy & Upbeat",
            2: "Cluster 2 — Akustik & Melankolis"
        }
        _CL_DESC = {
            0: "Danceability sedang-tinggi (0.656), energi moderat (0.596), tempo sedang. Lagu pop mainstream sehari-hari.",
            1: "Energi sangat tinggi (0.799), valence ceria (0.578), tempo cepat. Lagu hype, EDM, pop upbeat.",
            2: "Energi rendah (0.362), acousticness tinggi (0.504), tempo lebih lambat. Lagu akustik, ballad, atau ambient."
        }

        # ── Deskripsi card per cluster ──────────────────────
        _desc_cols = st.columns(3)
        for _cnum in range(3):
            _color = _CL_COLORS[_cnum]
            _label = _CL_LABEL[_cnum]
            _desc  = _CL_DESC[_cnum]
            with _desc_cols[_cnum]:
                st.markdown(f"""
<div style="background:#16213e;border:2px solid {_color};border-radius:10px;
            padding:.9rem 1rem;margin-bottom:.6rem;">
  <div style="color:{_color};font-weight:700;font-size:.9rem;margin-bottom:.4rem;">{_label}</div>
  <div style="color:#cbd5e1;font-size:.8rem;line-height:1.5;">{_desc}</div>
</div>""", unsafe_allow_html=True)

        st.divider()

        # ── Bar chart horizontal per cluster ────────────────
        _cl_cols = st.columns(len(_means))
        for _cidx, _row in _means.iterrows():
            _cnum  = int(_row["cluster_id"])
            _color = _CL_COLORS.get(_cnum, "#888")
            _label = _CL_LABEL.get(_cnum, f"Cluster {_cnum}")

            # Build full HTML string first, then call markdown ONCE
            _bars_html = ""
            for _f in _feats_available:
                _val = float(_row[_f])
                _pct = _val * 100
                _bars_html += (
                    f'<div style="margin:6px 0;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:.74rem;color:#9ca3af;margin-bottom:3px;">'
                    f'<span>{_f}</span><span>{_val:.3f}</span></div>'
                    f'<div style="background:#1e2a3a;border-radius:4px;height:16px;overflow:hidden;">'
                    f'<div style="width:{_pct:.1f}%;background:{_color};height:100%;'
                    f'border-radius:4px;"></div></div></div>'
                )

            _card_html = (
                f'<div style="background:#16213e;border:1px solid {_color};'
                f'border-radius:10px;padding:1rem;">'
                f'<div style="color:{_color};font-weight:700;font-size:.88rem;'
                f'margin-bottom:.8rem;">{_label}</div>'
                f'{_bars_html}'
                f'</div>'
            )

            with _cl_cols[_cidx]:
                st.markdown(_card_html, unsafe_allow_html=True)

        st.divider()

        # ─── D. Jumlah Lagu per Cluster ──────────────
        st.markdown("### D. Distribusi Lagu per Cluster")

        _counts = _cl_df.groupby("cluster_id").size().reset_index(name="jumlah_lagu")
        _total  = _counts["jumlah_lagu"].sum()

        _cnt_cols = st.columns(len(_counts))
        for _cidx, _crow in _counts.iterrows():
            _cnum  = int(_crow["cluster_id"])
            _color = _CL_COLORS.get(_cnum, "#888")
            _label = _CL_LABEL.get(_cnum, f"Cluster {_cnum}")
            _pct   = _crow["jumlah_lagu"] / _total * 100
            _card  = (
                f'<div style="background:#16213e;border:1px solid {_color};'
                f'border-radius:10px;padding:1rem;text-align:center;">'
                f'<div style="color:{_color};font-weight:700;font-size:.85rem;'
                f'margin-bottom:.4rem;">{_label}</div>'
                f'<div style="font-size:2rem;font-weight:800;color:#fff;">{int(_crow["jumlah_lagu"])}</div>'
                f'<div style="color:#9ca3af;font-size:.78rem;">lagu ({_pct:.1f}%)</div>'
                f'</div>'
            )
            _cnt_cols[_cidx].markdown(_card, unsafe_allow_html=True)

        st.divider()

        # ─── E. Tabel Rata-rata Fitur per Cluster ────
        st.markdown("### E. Tabel Rata-rata Fitur Audio")
        _display_means = _means.copy()
        _display_means["cluster_id"] = _display_means["cluster_id"].map(_CL_LABEL)
        _display_means = _display_means.rename(columns={"cluster_id": "Cluster"})
        for _f in _feats_available:
            _display_means[_f] = _display_means[_f].map("{:.4f}".format)
        st.dataframe(_display_means, use_container_width=True, hide_index=True)

        st.caption(
            "💡 Perbedaan utama: Cluster 1 memiliki **energy** tertinggi (0.799 vs 0.596), "
            "sementara Cluster 0 lebih balance. Cluster 2 menonjol di **acousticness** (0.504)."
        )

        st.divider()

        # ─── F. Contoh Lagu per Cluster ──────────────
        st.markdown("### F. Contoh Lagu per Cluster")
        _sample_cols = st.columns(len(_CL_LABEL))
        for _cnum, _clabel in _CL_LABEL.items():
            _color = _CL_COLORS.get(_cnum, "#888")
            _songs = _cl_df[_cl_df["cluster_id"] == _cnum][["track_name", "artist_names"]].head(5)
            with _sample_cols[_cnum]:
                _header = (
                    f'<div style="background:#16213e;border:1px solid {_color};'
                    f'border-radius:10px;padding:.7rem;margin-bottom:.5rem;">'
                    f'<div style="color:{_color};font-weight:700;font-size:.84rem;">{_clabel}</div>'
                    f'</div>'
                )
                st.markdown(_header, unsafe_allow_html=True)
                for _, _s in _songs.iterrows():
                    _t = str(_s["track_name"]).title()[:30]
                    _a = str(_s["artist_names"]).title()[:26]
                    _row_html = (
                        f'<div style="padding:5px 0;border-bottom:1px solid #1e2a3a;">'
                        f'<div style="font-size:.8rem;color:#e2e8f0;">🎵 {_t}</div>'
                        f'<div style="font-size:.7rem;color:#9ca3af;">{_a}</div>'
                        f'</div>'
                    )
                    st.markdown(_row_html, unsafe_allow_html=True)

    except Exception as _e:
        st.error(f"❌ Gagal memuat data cluster: {_e}")
        st.info("Pastikan file `outputs/item_cluster_kmeans.csv` dan `data/item_dataset.csv` tersedia.")

