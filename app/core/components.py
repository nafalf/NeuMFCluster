from html import escape
from pathlib import Path

import streamlit as st


def inject_css():
    """Membaca dan menyuntikkan style.css ke dalam aplikasi Streamlit."""
    css_path = Path(__file__).parent.parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""<style>
        .main-header{background:linear-gradient(135deg,#1DB954,#121212);padding:2rem;border-radius:12px;color:#fff;margin-bottom:1.5rem}
        .hit-banner{background:#1DB954;color:#121212;padding:1rem;border-radius:50px;font-weight:700;text-align:center;margin-bottom:1rem}
        .miss-banner{background:#e74c3c;color:#fff;padding:1rem;border-radius:50px;font-weight:700;text-align:center;margin-bottom:1rem}
        </style>""", unsafe_allow_html=True)


def hit_banner(rank):
    st.markdown(
        f'<div class="hit-banner">HIT - Item positif muncul di posisi #{rank} dari 100 kandidat.</div>',
        unsafe_allow_html=True
    )


def miss_banner(rank):
    st.markdown(
        f'<div class="miss-banner">MISS - Item positif tidak muncul di Top-10 (rank #{rank} dari 100 kandidat).</div>',
        unsafe_allow_html=True
    )


def song_card(rank, title, artist, score, is_true_item, cover_url=None, spotify_url=None, cluster_info=None, display_style="list"):
    """
    Menampilkan kartu lagu sebagai satu blok HTML utuh.
    display_style bisa 'list' (Top-10) atau 'grid' (history / item uji).
    """
    title = escape(str(title or "Untitled"))
    artist = escape(str(artist or ""))
    cluster_info = escape(str(cluster_info or ""))
    cover_url = escape(str(cover_url)) if cover_url else ""
    spotify_url = escape(str(spotify_url)) if spotify_url else ""

    cover_html = (
        f'<img class="song-cover" src="{cover_url}" alt="Album cover">'
        if cover_url
        else '<div class="song-cover song-cover-empty">No cover</div>'
    )
    spotify_html = (
        f'<a class="song-link" href="{spotify_url}" target="_blank">Spotify</a>'
        if spotify_url
        else ""
    )

    if display_style == "list":
        card_class = "song-card song-card-positive" if is_true_item else "song-card"
        rank_text = f"#{rank}" if rank is not None else ""
        label_html = '<span class="song-badge">ITEM POSITIF UJI</span>' if is_true_item else ""
        score_html = (
            f'<div class="song-score"><span>Skor NCF</span><strong>{float(score):.6f}</strong></div>'
            if score is not None
            else ""
        )
        cluster_html = (
            f'<div class="song-meta"><span class="song-label">Cluster:</span> {cluster_info}</div>'
            if cluster_info
            else ""
        )
        html = (
            f'<div class="{card_class}">'
            f'{cover_html}'
            f'<div class="song-body">'
            f'<div class="song-rank">{rank_text} {label_html}</div>'
            f'<div class="song-title"><span class="song-label">Judul:</span> {title}</div>'
            f'<div class="song-meta"><span class="song-label">Artist:</span> {artist}</div>'
            f'{cluster_html}'
            f'</div>'
            f'<div class="song-actions">{score_html}{spotify_html}</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
        return

    if display_style == "grid":
        card_class = "song-card-grid song-card-positive" if is_true_item else "song-card-grid"
        label_html = '<div class="song-badge">ITEM POSITIF UJI</div>' if is_true_item else ""
        score_html = f'<div class="song-grid-score">Skor NCF: {float(score):.6f}</div>' if score is not None else ""
        cluster_html = (
            f'<div class="song-meta"><span class="song-label">Cluster:</span> {cluster_info}</div>'
            if cluster_info
            else ""
        )
        html = (
            f'<div class="{card_class}">'
            f'{cover_html}'
            f'<div class="song-grid-body">'
            f'{label_html}'
            f'<div class="song-title"><span class="song-label">Judul:</span> {title}</div>'
            f'<div class="song-meta"><span class="song-label">Artist:</span> {artist}</div>'
            f'{cluster_html}'
            f'{score_html}'
            f'{spotify_html}'
            f'</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)


def flowchart_pipeline():
    st.markdown(
        """
<div class="score-flow">
  <div class="score-flow-row score-flow-inputs">
    <div class="score-node input">User ID</div>
    <div class="score-node input">Item ID Lagu</div>
    <div class="score-node input">Cluster Lagu</div>
  </div>
  <div class="score-flow-row score-flow-arrows">
    <div class="score-arrow-down">v</div>
    <div class="score-arrow-down">v</div>
    <div class="score-arrow-down">v</div>
  </div>
  <div class="score-flow-row">
    <div class="score-node">User GMF Embedding</div>
    <div class="score-node">Item GMF Embedding</div>
    <div class="score-node">User + Item + Cluster MLP Embedding</div>
  </div>
  <div class="score-flow-row">
    <div class="score-note center">Embedding pertama dan kedua masuk ke GMF; gabungan user, item, dan cluster masuk ke MLP.</div>
  </div>
  <div class="score-flow-row score-flow-arrows split">
    <div class="score-arrow-down">v</div>
    <div class="score-arrow-down">v</div>
  </div>
  <div class="score-flow-branches">
    <div class="score-branch">
      <div class="score-branch-title">Cabang GMF</div>
      <div class="score-node">Element-wise Product</div>
      <div class="score-note">Menangkap kedekatan user dan lagu secara langsung.</div>
    </div>
    <div class="score-branch">
      <div class="score-branch-title">Cabang MLP</div>
      <div class="score-node">Concat Embedding</div>
      <div class="score-node">Linear + ReLU + Dropout</div>
      <div class="score-note">Mempelajari pola non-linear dari user, lagu, dan cluster.</div>
    </div>
  </div>
  <div class="score-flow-row score-flow-arrows">
    <div class="score-arrow-down">v</div>
  </div>
  <div class="score-flow-row">
    <div class="score-node wide">Final Concat: output GMF + output MLP</div>
  </div>
  <div class="score-flow-row score-flow-arrows">
    <div class="score-arrow-down">v</div>
  </div>
  <div class="score-flow-row">
    <div class="score-node">Linear Output</div>
    <div class="score-arrow">-&gt;</div>
    <div class="score-node">Logit</div>
    <div class="score-arrow">-&gt;</div>
    <div class="score-node final">Sigmoid(logit) = Skor NCF</div>
  </div>
</div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar(data):
    with st.sidebar:
        st.markdown("## MusicRec Demo")

    return {
        "seed": 42
    }


def select_demo_user(data, key="demo_user"):
    user_df = data["user_df"]
    user_options = {
        f"{row['username']} | {int(row['n_interactions'])} interaksi": int(row["user_id_enc"])
        for _, row in user_df.head(1000).iterrows()
    }

    labels = list(user_options.keys())
    if not labels:
        return None, ""

    selected_label = st.selectbox("Demo Rekomendasi", labels, index=0, key=key)
    selected_user = user_options[selected_label]
    selected_username = selected_label.split(" | ")[0]
    return selected_user, selected_username
