import pickle
import streamlit as st
import pandas as pd

from core.components import inject_css
from core.config import (
    DATA_DIR, AUDIO_FEATURES,
    ELBOW_PLOT_PATH, SILHOUETTE_PLOT_ROOT_PATH
)

from core.data_loader import load_all
from core.components import inject_css, render_sidebar

inject_css()

try:
    with st.spinner("Memuat data..."):
        data = load_all()
except Exception as e:
    st.error(f"Gagal memuat aplikasi: {e}")
    st.stop()

render_sidebar(data)

st.title("Visualisasi K-Means Clustering")
st.caption("Hasil K-Means Clustering (K=3) pada 5 fitur audio lagu Spotify.")

#  Tentukan path data cluster 
# Cluster data ada di kmeans_cluster/ sesuai hasil pelatihan
CLUSTER_CSV = "kmeans_cluster/item_cluster.csv"
ITEM_CSV_5F = DATA_DIR / "item_dataset_5f.csv"
ITEM_CSV    = DATA_DIR / "item_dataset.csv"
SCALER_PKL  = "kmeans_cluster/scaler_cluster.pkl"
PCA_PKL     = "kmeans_cluster/pca_5f.pkl"
SCATTER_PNG = "kmeans_cluster/scatter_cluster_2d.png"
ELBOW_SIL   = "kmeans_cluster/elbow_silhouette_plot.png"
FITUR_PNG   = "kmeans_cluster/fitur_per_cluster.png"

from pathlib import Path

_CL_COLORS = {0: "#1DB954", 1: "#3498db", 2: "#e67e22"}
_CL_LABEL  = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2"
}

#  A. Elbow & Silhouette 
st.markdown("### A. Evaluasi Jumlah Cluster")
_ea, _sb = st.columns(2)
with _ea:
    p = Path(ELBOW_SIL)
    if p.exists():
        st.image(str(p), use_container_width=True, caption="Elbow & Silhouette Plot")
    else:
        st.info(f"`{ELBOW_SIL}` belum tersedia.")
with _sb:
    p2 = Path(SILHOUETTE_PLOT_ROOT_PATH)
    p3 = Path("kmeans_cluster/silhouette_plot.png")
    if p2.exists():
        st.image(str(p2), use_container_width=True, caption="Silhouette Score per K")
    elif p3.exists():
        st.image(str(p3), use_container_width=True, caption="Silhouette Score per K")
    else:
        st.info("Silhouette plot belum tersedia.")

st.divider()

#  B. Scatter Plot (Saved PNG lebih hemat RAM) 
st.markdown("### B. Persebaran Cluster (PCA 2D)")

_scatter = Path(SCATTER_PNG)
if _scatter.exists():
    st.image(str(_scatter), use_container_width=True,
             caption="Scatter Plot PCA 2D - tiap titik = 1 lagu, warna = cluster")
else:
    st.info(f"`{SCATTER_PNG}` belum tersedia. Generate dengan menjalankan clustering notebook.")

st.divider()

#  C. Fitur Per Cluster (Saved PNG) 
st.markdown("### C. Karakteristik Fitur Audio per Cluster")

_fitur = Path(FITUR_PNG)
if _fitur.exists():
    st.image(str(_fitur), use_container_width=True,
             caption="Rata-rata 5 fitur audio per cluster")
else:
    # Fallback: hitung dari CSV
    try:
        _cl_df = pd.read_csv(CLUSTER_CSV)
        _it_df = pd.read_csv(str(ITEM_CSV_5F) if ITEM_CSV_5F.exists() else str(ITEM_CSV))
        
        # Kolom cluster: gunakan cluster_5f jika ada
        if "cluster_5f" in _cl_df.columns:
            _cl_df["cluster"] = _cl_df["cluster_5f"]
        
        _feats = [f for f in AUDIO_FEATURES if f in _it_df.columns]
        
        # Gabungkan via track_id / item_id
        id_col = "track_id" if "track_id" in _cl_df.columns else _cl_df.columns[0]
        _merged = _cl_df[["cluster", id_col]].copy()
        for f in _feats:
            _merged[f] = _merged[id_col].map(dict(zip(_it_df[id_col], _it_df[f]))) if id_col in _it_df.columns else None
        
        _means = _merged.groupby("cluster")[_feats].mean().round(4).reset_index()
        
        _cl_cols = st.columns(len(_means))
        for _cidx, _row in _means.iterrows():
            _cn   = int(_row["cluster"])
            _color = _CL_COLORS.get(_cn, "#888")
            _label = _CL_LABEL.get(_cn, f"Cluster {_cn}")
            _bars  = ""
            for f in _feats:
                _v   = float(_row[f]) if not pd.isna(_row[f]) else 0
                _pct = _v * 100
                _bars += (
                    f'<div style="margin:5px 0">'
                    f'<div style="display:flex;justify-content:space-between;font-size:.73rem;color:#b3b3b3;margin-bottom:2px">'
                    f'<span>{f}</span><span>{_v:.3f}</span></div>'
                    f'<div style="background:#282828;border-radius:4px;height:14px;overflow:hidden">'
                    f'<div style="width:{_pct:.1f}%;background:{_color};height:100%;border-radius:4px"></div>'
                    f'</div></div>'
                )
            with _cl_cols[_cidx]:
                st.markdown(
                    f'<div style="background:#181818;border:1px solid #282828;border-radius:10px;padding:1rem">'
                    f'<div style="color:{_color};font-weight:700;font-size:.88rem;margin-bottom:.8rem">{_label}</div>'
                    f'{_bars}</div>',
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.error(f"Gagal memuat data fitur: {e}")

st.markdown(
    '<div class="cluster-interpretation-grid">'
    '<div class="cluster-interpretation-card cluster-0">'
    '<strong>Cluster 0 - Energik dan populer</strong>'
    '<p>Memiliki danceability, energy, dan valence yang relatif tinggi. Cluster ini merepresentasikan lagu yang cenderung mudah dinikmati, upbeat, dan cocok untuk rekomendasi bernuansa pop atau dance.</p>'
    '</div>'
    '<div class="cluster-interpretation-card cluster-1">'
    '<strong>Cluster 1 - Seimbang dan moderat</strong>'
    '<p>Nilai energy dan danceability berada di tengah, tetapi valence dan acousticness lebih rendah. Cluster ini cocok dibaca sebagai kelompok lagu yang lebih netral, tidak terlalu akustik, dan tidak terlalu ceria.</p>'
    '</div>'
    '<div class="cluster-interpretation-card cluster-2">'
    '<strong>Cluster 2 - Akustik dan mellow</strong>'
    '<p>Acousticness paling menonjol, sementara energy, danceability, dan valence lebih rendah. Cluster ini mengarah ke lagu yang lebih kalem, akustik, melankolis, atau santai.</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

#  D. Distribusi Lagu 
st.markdown("### D. Distribusi Lagu per Cluster")
try:
    _cl_df = pd.read_csv(CLUSTER_CSV)
    if "cluster_5f" in _cl_df.columns:
        _cl_df["cluster"] = _cl_df["cluster_5f"]
    
    _counts = _cl_df.groupby("cluster").size().reset_index(name="n")
    _total  = int(_counts["n"].sum())
    
    _dc = st.columns(len(_counts))
    for _i, _crow in _counts.iterrows():
        _cn    = int(_crow["cluster"])
        _color = _CL_COLORS.get(_cn, "#888")
        _label = _CL_LABEL.get(_cn, f"Cluster {_cn}")
        _pct   = _crow["n"] / _total * 100
        _dc[_i].markdown(
            f'<div style="background:#181818;border:1px solid #282828;border-radius:10px;'
            f'padding:1rem;text-align:center">'
            f'<div style="color:{_color};font-weight:700;font-size:.85rem;margin-bottom:.4rem">{_label}</div>'
            f'<div style="font-size:2rem;font-weight:800;color:#fff">{int(_crow["n"])}</div>'
            f'<div style="color:#b3b3b3;font-size:.78rem">lagu ({_pct:.1f}%)</div>'
            f'</div>',
            unsafe_allow_html=True
        )
except Exception as e:
    st.error(f"Gagal memuat distribusi cluster: {e}")
