import streamlit as st

from core.components import inject_css, render_sidebar
from core.data_loader import load_all


st.set_page_config(
    page_title="Dashboard Dataset - NeuMF + K-Means",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

try:
    with st.spinner("Memuat data..."):
        data = load_all()
except Exception as e:
    st.error(f"Gagal memuat aplikasi: {e}")
    st.stop()

render_sidebar(data)

train_df = data["train_df"]
test_df = data["test_df"]
n_users = int(data["n_users"])
n_items = int(data["n_items"])
n_clusters = int(data["n_clusters"])

train_positive = int((train_df["label"] == 1).sum())
test_positive = int((test_df["label"] == 1).sum())
n_interactions = train_positive + test_positive
total_possible = n_users * n_items
density = n_interactions / total_possible if total_possible else 0
sparsity = 1 - density

st.title("Dashboard Dataset")
st.caption("Ringkasan data yang digunakan pada demo rekomendasi NeuMF + K-Means.")

metric_cols = st.columns(6)
metric_cols[0].metric("Jumlah User", f"{n_users:,}")
metric_cols[1].metric("Jumlah Item", f"{n_items:,}")
metric_cols[2].metric("Interaksi Positif", f"{n_interactions:,}")
metric_cols[3].metric("Density", f"{density * 100:.4f}%")
metric_cols[4].metric("Sparsity", f"{sparsity * 100:.4f}%")
metric_cols[5].metric("Jumlah Cluster", f"{n_clusters:,}")

st.markdown(
    """
<div class="dashboard-note">
  <strong>Dataset</strong>
  <span>Data lagu berasal dari Spotify, sedangkan pola interaksi pengguna berasal dari Last.fm. Data kemudian dipakai untuk membentuk pasangan user-item, label interaksi, dan informasi cluster lagu.</span>
</div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Perhitungan Density dan Sparsity")
st.markdown(
    f"""
<div class="dashboard-note">
  <strong>Rumus</strong>
  <span>Total kemungkinan pasangan user-item = <code>{n_users:,}</code> user x <code>{n_items:,}</code> item = <code>{total_possible:,}</code> pasangan.</span>
  <span>Density = interaksi positif / total kemungkinan pasangan = <code>{n_interactions:,}</code> / <code>{total_possible:,}</code> = <code>{density * 100:.4f}%</code>.</span>
  <span>Sparsity = 1 - density = <code>{sparsity * 100:.4f}%</code>. Artinya hanya sebagian kecil pasangan user-item yang benar-benar memiliki interaksi.</span>
</div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Komposisi Data Evaluasi")
split_cols = st.columns(4)
split_cols[0].metric("Train Positif", f"{train_positive:,}")
split_cols[1].metric("Test Positif", f"{test_positive:,}")
split_cols[2].metric("Train Baris", f"{len(train_df):,}")
split_cols[3].metric("Test Baris", f"{len(test_df):,}")

st.markdown(
    """
<div class="dashboard-note">
  <strong>Negative Sampling</strong>
  <span>Data train memakai rasio <code>1 positif : 4 negatif</code> untuk setiap interaksi train.</span>
  <span>Data testing memakai rasio <code>1 positif : 99 negatif</code> untuk setiap user, sehingga evaluasi dilakukan pada 100 kandidat lagu per user.</span>
</div>
    """,
    unsafe_allow_html=True,
)
