from pathlib import Path

# ==================================================
# PATH CONFIG
# ==================================================

# Menggunakan path relatif terhadap root project
DATA_DIR    = Path("data")
MODEL_DIR   = Path("src/models")    # model .pt ada di src/models/
ENCODER_DIR = DATA_DIR / "pkl"

# Data loader mengambil data cluster dari kmeans_cluster sesuai src
CLUSTER_PATH = Path("kmeans_cluster/item_dataset_5f.csv")

# Path tambahan untuk Tab KMeans Cluster
ELBOW_PLOT_PATH = Path("elbow_plot.png")
SILHOUETTE_PLOT_ROOT_PATH = Path("silhouette_plot.png")
SILHOUETTE_PLOT_TEST_PATH = Path("testing_code/silhouette_plot.png")
ITEM_CLUSTER_KMEANS_PATH = Path("outputs/item_cluster_kmeans.csv")
SCALER_PATH = Path("scaler_5f.pkl")
PCA_PATH = Path("pca_5f.pkl")

# The audio features selected for this project
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness"
]

# Model candidates ordered by priority (path, uses_cluster)
# Hanya skenario 4 ncf_cluster_tpe sesuai permintaan
MODEL_CANDIDATES = [
    (MODEL_DIR / "skenario4_cluster_5f_tpe_2layer.pt", True),  # Model utama skripsi
]
