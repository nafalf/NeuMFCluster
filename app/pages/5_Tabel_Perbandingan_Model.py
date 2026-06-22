import pandas as pd
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

st.title("Tabel Perbandingan Skenario Eksperimen")
st.caption("Kolom metrik dikosongkan agar dapat diisi sesuai hasil akhir yang ingin ditampilkan.")

scenario_df = pd.DataFrame(
    {
        "Skenario": [
            "Skenario 1 - NeuMF Baseline",
            "Skenario 2 - NeuMF + K-Means Cluster",
            "Skenario 3 - NeuMF + TPE Hyperparameter",
            "Skenario 4 - NeuMF + Cluster + TPE (Final)",
        ],
        "HR@10": ["", "", "", ""],
        "NDCG@10": ["", "", "", ""],
    }
)

st.dataframe(
    scenario_df[["Skenario", "HR@10", "NDCG@10"]],
    use_container_width=True,
    hide_index=True,
)
