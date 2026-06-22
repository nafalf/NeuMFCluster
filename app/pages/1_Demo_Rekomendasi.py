import pandas as pd
import streamlit as st

from core.components import (
    hit_banner,
    inject_css,
    miss_banner,
    render_sidebar,
    select_demo_user,
    song_card,
)
from core.data_loader import load_all
from core.inference import get_item_row, run_single_user_hit_miss


inject_css()

try:
    with st.spinner("Memuat data..."):
        data = load_all()
except Exception as e:
    st.error(f"Gagal memuat aplikasi: {e}")
    st.stop()

user_df = data["user_df"]
train_df = data["train_df"]
item_df = data["item_df"]
cluster_labels = data["cluster_labels"]

if user_df.empty:
    st.error("Tidak ada user valid.")
    st.stop()

sidebar_data = render_sidebar(data)

st.title("Demo Rekomendasi Musik")
st.caption("Pilih satu user, lalu klik Generate untuk menampilkan Top-10 rekomendasi.")

selected_user, selected_username = select_demo_user(data, key="demo_page_user")
n_history = st.slider("Tampilkan Jumlah Riwayat", min_value=0, max_value=20, value=10)
seed = sidebar_data["seed"]

if selected_user is None:
    st.error("Tidak ada user valid.")
    st.stop()

current_key = f"{selected_user}-{n_history}-{seed}"

if st.button("Generate Rekomendasi", type="primary", use_container_width=True):
    st.session_state["demo_generated_key"] = current_key

if st.session_state.get("demo_generated_key") != current_key:
    st.info("Klik Generate Rekomendasi untuk menghitung skor NCF dan menampilkan Top-10 lagu.")
    st.stop()


def format_cluster(cluster_id):
    cluster_id = int(cluster_id)
    label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
    prefix = f"Cluster {cluster_id} - "
    if label.startswith(prefix):
        return f"{cluster_id} - {label[len(prefix):]}"
    return str(cluster_id)


def get_cluster_counts(user_history):
    if user_history.empty:
        return {}, 0

    hist_cols = ["item_id_enc", "cluster"]
    item_cluster_lookup = item_df[hist_cols].drop_duplicates("item_id_enc")
    merged_cluster = user_history.merge(item_cluster_lookup, on="item_id_enc", how="left")
    counts = (
        merged_cluster.dropna(subset=["cluster"])
        .assign(cluster=lambda df: df["cluster"].astype(int))
        .groupby("cluster")
        .size()
        .to_dict()
    )

    total = sum(counts.values())
    return counts, total


def render_cluster_summary(user_history):
    counts, total = get_cluster_counts(user_history)
    if total == 0:
        return

    cards = []
    for cluster_id in [0, 1, 2]:
        count = counts.get(cluster_id, 0)
        pct = count / total * 100
        cards.append(
            f'<div class="cluster-summary-card cluster-{cluster_id}">'
            f'<strong>{format_cluster(cluster_id)}</strong>'
            f'<span>{count} lagu riwayat ({pct:.1f}%)</span>'
            f'</div>'
        )

    st.markdown(
        '<div class="cluster-summary-grid">' + ''.join(cards) + '</div>',
        unsafe_allow_html=True,
    )


summary, top10_df = run_single_user_hit_miss(
    user_id_enc=selected_user,
    data=data,
    seed=seed,
    num_neg=99,
    top_k=10,
)

if summary is None:
    st.error("User ini tidak punya data evaluasi valid. Silakan pilih user lain.")
    st.stop()

st.subheader(f"Profil Pengguna: {selected_username}")

user_history = (
    train_df[(train_df["user_id_enc"] == selected_user) & (train_df["label"] == 1)]
    .copy()
)
user_train = user_history.tail(n_history).copy()

st.subheader("Ringkasan Cluster Riwayat")
render_cluster_summary(user_history)

st.subheader("Riwayat Lagu yang Didengar")

if user_train.empty:
    st.warning("Tidak ada riwayat untuk user ini.")
else:
    hist_cols = [
        "item_id_enc",
        "track_name",
        "artist_names",
        "cluster",
        "cover_url",
        "cover_url_md",
        "spotify_url",
    ]
    existing_cols = [c for c in hist_cols if c in item_df.columns]
    merged = user_train.merge(item_df[existing_cols], on="item_id_enc", how="left").reset_index(drop=True)

    for i in range(0, len(merged), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(merged):
                row = merged.iloc[i + j]
                with cols[j]:
                    cover = (
                        row["cover_url_md"]
                        if ("cover_url_md" in row and pd.notna(row["cover_url_md"]))
                        else row.get("cover_url")
                    )
                    song_card(
                        rank=None,
                        title=row["track_name"],
                        artist=row["artist_names"],
                        score=None,
                        is_true_item=False,
                        cover_url=cover,
                        spotify_url=row.get("spotify_url"),
                        display_style="grid",
                    )

st.divider()

st.subheader("Lagu Uji (Item Positif Test)")
true_row = get_item_row(summary["true_item_id"], item_df)
if true_row is not None:
    cover = (
        true_row.get("cover_url_md")
        if pd.notna(true_row.get("cover_url_md", None))
        else true_row.get("cover_url")
    )
    song_card(
        rank=None,
        title=summary["true_title"],
        artist=summary["true_artist"],
        score=summary["true_score"],
        is_true_item=True,
        cover_url=cover,
        spotify_url=true_row.get("spotify_url"),
        cluster_info=format_cluster(summary["true_cluster"]),
        display_style="list",
    )
else:
    st.info(f"Detail lagu tidak ditemukan (ID: {summary['true_item_id']})")

st.divider()

st.subheader(f"Top-10 Rekomendasi untuk {selected_username}")

if summary["hit"]:
    hit_banner(summary["true_rank"])
else:
    miss_banner(summary["true_rank"])

for _, row in top10_df.iterrows():
    cluster_info = format_cluster(row["cluster"])
    song_card(
        rank=int(row["rank"]),
        title=row["track_name"],
        artist=row["artist_names"],
        score=float(row["score"]),
        is_true_item=bool(row["is_true_item"]),
        cover_url=row["cover_url"],
        spotify_url=row["spotify_url"],
        cluster_info=cluster_info,
        display_style="list",
    )
