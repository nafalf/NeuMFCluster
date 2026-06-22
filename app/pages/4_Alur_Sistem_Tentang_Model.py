import streamlit as st

from core.components import inject_css, render_sidebar
from core.data_loader import load_all


inject_css()

try:
    with st.spinner("Memuat data..."):
        data = load_all()
except Exception as e:
    st.error(f"Gagal memuat aplikasi: {e}")
    st.stop()

render_sidebar(data)

st.title("Alur Sistem / Tentang Model")
st.caption("Journey data dari dataset mentah sampai menjadi skor NCF dan Top-10 rekomendasi.")

flow_steps = [
    {
        "title": "Data Spotify + Last.fm",
        "detail": "Spotify: 6.513 lagu mentah dengan metadata dan fitur audio. Last.fm: 730.071 interaksi awal dari 21.476 user dan 6.225 item.",
    },
    {
        "title": "Preprocessing Item",
        "detail": "Data Spotify dibersihkan dari missing value dan duplikasi track-artist, lalu menjadi 5.494 item unik pada item_dataset.csv.",
    },
    {
        "title": "Preprocessing User 10-Core",
        "detail": "Interaksi Last.fm difilter minimal 10 interaksi per user dan 10 interaksi per item. Hasil awal: 12.044 user, 4.882 item, 687.744 interaksi.",
    },
    {
        "title": "Filter Item dengan Audio Features",
        "detail": "Item user disesuaikan dengan item Spotify yang memiliki fitur audio. Item turun dari 4.882 menjadi 4.144.",
    },
    {
        "title": "Re-10-Core Final",
        "detail": "10-core dijalankan ulang setelah filter item. Data final: 9.529 user, 4.144 item, 561.227 interaksi positif.",
    },
    {
        "title": "Encoding User dan Item",
        "detail": "User dan track_id diubah menjadi indeks numerik. Range akhir: user_id_enc 0-9.528 dan item_id_enc 0-4.143.",
    },
    {
        "title": "LOO Split dan Negative Sampling",
        "detail": "Setiap user punya 1 item positif test + 99 negatif. Train memakai sisa positif, masing-masing ditambah 4 negatif.",
    },
    {
        "title": "Train/Test Dataset",
        "detail": "Train: 2.758.490 baris. Test: 952.900 baris. Angka ini sudah termasuk label positif dan negatif hasil sampling.",
    },
    {
        "title": "PCA untuk Clustering",
        "detail": "5 fitur audio dipakai: danceability, energy, valence, acousticness, tempo. PCA 2 komponen mempertahankan 66,91 persen varians.",
    },
    {
        "title": "K-Means Clustering",
        "detail": "Konfigurasi terbaik: PCA n=2 dan K=3 dengan Silhouette 0,4175. Distribusi awal: 2.263, 2.231, dan 1.000 lagu.",
    },
    {
        "title": "Cluster Embedding",
        "detail": "Cluster final untuk 4.144 item disimpan pada item_cluster_4144.csv. Tidak ada missing cluster dan cluster unik adalah 0, 1, 2.",
    },
    {
        "title": "NeuMF + Cluster + TPE",
        "detail": "Model final memakai emb_dim 64, cluster_dim 16, dropout 0,0808, learning rate 0,000614, batch size 2.048, dan 30 epoch.",
    },
    {
        "title": "Logit",
        "detail": "Untuk setiap pasangan user-item-cluster, model menghasilkan raw output atau logit.",
    },
    {
        "title": "Sigmoid",
        "detail": "Skor NCF dihitung dengan sigmoid(logit), sehingga nilainya berada pada rentang 0 sampai 1.",
    },
    {
        "title": "Top-10 Recommendation",
        "detail": "Pada demo, 100 kandidat lagu untuk satu user diberi skor NCF, lalu diurutkan dari skor tertinggi menjadi Top-10.",
    },
]

flow_html = ['<div class="system-flow">']
for idx, step in enumerate(flow_steps):
    flow_html.append(
        f'<div class="system-flow-node">'
        f'<strong>{step["title"]}</strong>'
        f'<span>{step["detail"]}</span>'
        f'</div>'
    )
    if idx < len(flow_steps) - 1:
        flow_html.append('<div class="system-flow-arrow">&darr;</div>')
flow_html.append('</div>')

st.subheader("A. Journey Data dan Model")
st.markdown("".join(flow_html), unsafe_allow_html=True)

st.divider()

st.subheader("B. Contoh Encoding")

st.markdown(
    """
<div class="encoding-grid">
  <div class="encoding-card">
    <strong>User Encoding</strong>
    <span>Contoh: user_id <code>--mopsi--</code> menjadi <code>user_id_enc = 0</code>. Total user final adalah <code>9.529</code>.</span>
  </div>
  <div class="encoding-card">
    <strong>Item Encoding</strong>
    <span>Contoh: track_id <code>05mAIVLkIWc2d1UBYZBCp8</code> menjadi <code>item_id_enc = 70</code>. Range item final adalah <code>0-4.143</code>.</span>
  </div>
  <div class="encoding-card">
    <strong>Cluster Encoding</strong>
    <span>Contoh: track_id <code>000xQL6tZNLJzIrtIgxqSl</code> memiliki <code>item_id = 0</code> dan <code>cluster = 0</code>.</span>
  </div>
  <div class="encoding-card">
    <strong>Label Interaksi</strong>
    <span>Interaksi asli Last.fm diberi <code>label = 1</code>. Item negatif hasil sampling diberi <code>label = 0</code>.</span>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.subheader("C. Tentang Model")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
<div class="model-info-card">
  <strong>Data dan Cluster</strong>
  <span>Fitur lagu dari Spotify diproses, direduksi dengan PCA, lalu dikelompokkan menggunakan K-Means. Hasil cluster menjadi informasi tambahan untuk item lagu.</span>
</div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
<div class="model-info-card">
  <strong>Skor NCF</strong>
  <span>NeuMF menghasilkan logit untuk pasangan user dan lagu. Logit tersebut dilewatkan ke sigmoid sehingga menjadi skor NCF, lalu kandidat lagu diurutkan dari skor tertinggi.</span>
</div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="model-info-card model-info-wide">
  <strong>Peran Cluster Embedding</strong>
  <span>Cluster digunakan pada cabang MLP sebagai representasi karakteristik lagu. Tujuannya bukan menggantikan pola interaksi user-item, tetapi memperkaya input item agar model memiliki konteks tambahan tentang tipe lagu.</span>
</div>
    """,
    unsafe_allow_html=True,
)

st.subheader("D. Dari Logit ke Skor Sigmoid")

st.markdown(
    """
<div class="model-info-card model-info-wide">
  <strong>Perhitungan Skor NCF</strong>
  <span>Pada tahap inference, model menerima satu user dan 100 kandidat lagu. Untuk setiap kandidat, model menghasilkan <code>logit</code>. Nilai ini dihitung menjadi skor dengan rumus <code>skor = 1 / (1 + exp(-logit))</code>. Skor inilah yang ditampilkan sebagai Skor NCF dan dipakai untuk mengurutkan Top-10 rekomendasi.</span>
</div>
    """,
    unsafe_allow_html=True,
)

eval_cols = st.columns(3)
eval_cols[0].metric("Best Trial HR@10", "0.8787")
eval_cols[1].metric("Final HR@10", "0.8449")
eval_cols[2].metric("Final NDCG@10", "0.6151")
