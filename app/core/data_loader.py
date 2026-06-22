import pickle
import pandas as pd
import streamlit as st
import torch
from .config import (
    ENCODER_DIR, DATA_DIR, CLUSTER_PATH, MODEL_CANDIDATES, AUDIO_FEATURES
)
from .model import NeuMFCluster, NeuMFBaseline

def load_encoders():
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
        
    return user_enc, item_enc

def load_dataframes():
    train_cluster_path = DATA_DIR / "train_dataset_cluster_5f.csv"
    test_cluster_path  = DATA_DIR / "test_dataset_cluster_5f.csv"
    train_df_path = train_cluster_path if train_cluster_path.exists() else DATA_DIR / "train_dataset.csv"
    test_df_path  = test_cluster_path if test_cluster_path.exists() else DATA_DIR / "test_dataset.csv"

    if not train_df_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {train_df_path}")
    if not test_df_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {test_df_path}")

    train_cols = ["user_id_enc", "item_id_enc", "label"]
    train_dtypes = {"user_id_enc": "int32", "item_id_enc": "int32", "label": "int8"}
    if train_cluster_path.exists() and test_cluster_path.exists():
        train_cols.append("cluster_5f")
        train_dtypes["cluster_5f"] = "int8"

    train_df = pd.read_csv(train_df_path, usecols=train_cols, dtype=train_dtypes)
    test_df  = pd.read_csv(test_df_path, usecols=train_cols, dtype=train_dtypes)

    item_5f_path      = DATA_DIR / "item_dataset_5f.csv"
    item_default_path = DATA_DIR / "item_dataset.csv"

    if item_5f_path.exists():
        item_df = pd.read_csv(item_5f_path)
    elif item_default_path.exists():
        item_df = pd.read_csv(item_default_path)
    else:
        raise FileNotFoundError("item_dataset_5f.csv atau item_dataset.csv tidak ditemukan.")

    cluster_4144_path = DATA_DIR / "item_cluster_4144.csv"
    cluster_path = cluster_4144_path if cluster_4144_path.exists() else CLUSTER_PATH

    if not cluster_path.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {cluster_path}")

    cluster_df = pd.read_csv(cluster_path)
    # Sesuaikan dengan logic di folder src (new_ncf_cluster_tpe.ipynb)
    if 'cluster_5f' in cluster_df.columns:
        cluster_df['cluster'] = cluster_df['cluster_5f']
    
    return train_df, test_df, item_df, cluster_df

def load_model(n_users, n_items, n_clusters):
    model = None
    model_info = {
        "name":         None,
        "uses_cluster": True,
        "uses_tpe":     False,
        "hr10":         None,
        "ndcg10":       None,
        "mlp_layer":    None,
        "best_params":  {}
    }

    for model_path, uses_cluster in MODEL_CANDIDATES:
        if not model_path.exists():
            continue

        ckpt   = torch.load(model_path, map_location="cpu")
        params = ckpt.get("best_params", {})
        mlp_layer = ckpt.get("mlp_layer", None)

        if uses_cluster:
            model = NeuMFCluster(
                n_users=n_users,
                n_items=n_items,
                n_clusters=n_clusters,
                emb_dim=params.get("emb_dim", 64),
                cluster_dim=params.get("cluster_dim", 16),
                dropout=params.get("dropout", 0.2),
                mlp_layer=mlp_layer
            )
        else:
            model = NeuMFBaseline(
                n_users=n_users,
                n_items=n_items,
                emb_dim=params.get("emb_dim", 64),
                dropout=params.get("dropout", 0.2),
                mlp_layer=mlp_layer
            )

        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        model_info["name"]         = model_path.stem
        model_info["uses_cluster"] = uses_cluster
        model_info["uses_tpe"]     = ckpt.get("tpe", False)
        model_info["hr10"]         = ckpt.get("HR@10_mean", None)
        model_info["ndcg10"]       = ckpt.get("NDCG@10_mean", None)
        model_info["mlp_layer"]    = mlp_layer
        model_info["best_params"]  = params

        break

    if model is None:
        raise FileNotFoundError("Tidak ada file model .pt yang cocok di folder models.")
        
    return model, model_info

def build_user_table(train_df, test_df, user_enc):
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
    return user_df

@st.cache_resource(show_spinner=False)
def load_all():
    user_enc, item_enc = load_encoders()
    train_df, test_df, item_df, cluster_df = load_dataframes()

    track_to_enc = dict(
        zip(
            item_enc.classes_,
            item_enc.transform(item_enc.classes_)
        )
    )

    if "item_id" in cluster_df.columns:
        cluster_df["item_id_enc"] = cluster_df["item_id"]
    else:
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

    if "item_id" in item_df.columns:
        item_df["item_id_enc"] = item_df["item_id"].astype(int)
    else:
        item_df["item_id_enc"] = item_df["track_id"].map(track_to_enc)
    item_df["cluster"]     = item_df["item_id_enc"].map(item_cluster)

    if "cover_url_md" not in item_df.columns:
        if "cover_url" in item_df.columns:
            item_df["cover_url_md"] = item_df["cover_url"]
        else:
            item_df["cover_url_md"] = None

    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)

    model, model_info = load_model(n_users, n_items, n_clusters)

    train_positive = (
        train_df[train_df["label"] == 1]
        .groupby("user_id_enc")["item_id_enc"]
        .apply(set)
        .to_dict()
    )

    user_df = build_user_table(train_df, test_df, user_enc)

    cluster_labels = {
        0: "Cluster 0 - Energik & Populer",
        1: "Cluster 1 - Seimbang & Moderat",
        2: "Cluster 2 - Akustik & Mellow"
    }

    # Determine which audio features actually exist in item_df
    available_audio_features = [f for f in AUDIO_FEATURES if f in item_df.columns]

    return {
        "model":                   model,
        "model_info":              model_info,
        "user_enc":                user_enc,
        "item_enc":                item_enc,
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
