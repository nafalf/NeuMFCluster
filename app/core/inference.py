import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ==================================================
# DATA HELPERS
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


# ==================================================
# INFERENCE
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

@torch.no_grad()
def explain_forward(user_id_enc, item_id_enc, cluster_id, model, model_info):
    """
    Menjalankan forward pass untuk SATU pasang (user, item, cluster) 
    dan merekam output intermediate menggunakan forward hooks.
    """
    # Siapkan tensor
    user_t = torch.tensor([user_id_enc], dtype=torch.long)
    item_t = torch.tensor([item_id_enc], dtype=torch.long)
    
    uses_cluster = model_info["uses_cluster"]
    cluster_t = torch.tensor([cluster_id], dtype=torch.long) if uses_cluster else None

    # Struktur untuk menampung intermediate values
    intermediate = {}

    # Ekstrak manual GMF dan embedding awal
    user_gmf_emb = model.user_gmf(user_t)
    item_gmf_emb = model.item_gmf(item_t)
    gmf_output = user_gmf_emb * item_gmf_emb

    intermediate["gmf_user_emb"] = user_gmf_emb.squeeze().cpu().numpy()
    intermediate["gmf_item_emb"] = item_gmf_emb.squeeze().cpu().numpy()
    intermediate["gmf_product"] = gmf_output.squeeze().cpu().numpy()

    user_mlp_emb = model.user_mlp(user_t)
    item_mlp_emb = model.item_mlp(item_t)
    intermediate["mlp_user_emb"] = user_mlp_emb.squeeze().cpu().numpy()
    intermediate["mlp_item_emb"] = item_mlp_emb.squeeze().cpu().numpy()

    if uses_cluster:
        cluster_emb = model.cluster_emb(cluster_t)
        intermediate["mlp_cluster_emb"] = cluster_emb.squeeze().cpu().numpy()
        mlp_in = torch.cat([user_mlp_emb, item_mlp_emb, cluster_emb], dim=-1)
    else:
        mlp_in = torch.cat([user_mlp_emb, item_mlp_emb], dim=-1)

    intermediate["mlp_concat_input"] = mlp_in.squeeze().cpu().numpy()

    # Hook untuk merekam output setiap nn.Linear di dalam model.mlp
    hook_handles = []
    layer_outputs = {}

    def get_hook(layer_idx):
        def hook(module, input, output):
            layer_outputs[layer_idx] = output.squeeze().cpu().numpy()
        return hook

    linear_idx = 0
    for layer in model.mlp:
        if isinstance(layer, nn.Linear):
            handle = layer.register_forward_hook(get_hook(linear_idx))
            hook_handles.append(handle)
            linear_idx += 1

    # Lakukan forward pass penuh
    if uses_cluster:
        logit = model(user_t, item_t, cluster_t)
    else:
        logit = model(user_t, item_t)

    # Hapus semua hook agar tidak mengganggu pass berikutnya
    for handle in hook_handles:
        handle.remove()

    # Simpan output MLP
    intermediate["mlp_layers_output"] = layer_outputs

    # Ambil output layer terakhir dari MLP (dari layer_outputs index terakhir)
    if layer_outputs:
        last_mlp_idx = max(layer_outputs.keys())
        last_mlp_output = layer_outputs[last_mlp_idx]
    else:
        last_mlp_output = mlp_in.squeeze().cpu().numpy() # fallback if no linear layers

    # Concat GMF dan MLP terakhir (sebelum self.output layer)
    concat_final = np.concatenate([intermediate["gmf_product"], last_mlp_output])
    intermediate["final_concat"] = concat_final

    # Hasil akhir
    intermediate["logit"] = logit.item()
    intermediate["score"] = torch.sigmoid(logit).item()

    return intermediate
