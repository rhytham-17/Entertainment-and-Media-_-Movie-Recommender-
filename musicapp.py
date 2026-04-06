"""
Fast Music Recommender - Optimized for large datasets
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")

# Dataset path finder
BASE_DIR = Path(__file__).resolve().parent
CANDIDATE_PATHS = [
    BASE_DIR / "tcc_ceds_music.csv",
    BASE_DIR / "Data" / "tcc_ceds_music.csv",
]

MUSIC_CSV_LOCAL = next((p for p in CANDIDATE_PATHS if p.exists()), None)

if MUSIC_CSV_LOCAL is None:
    st.error("Place tcc_ceds_music.csv in project folder or Data/")
    st.stop()

FEATURE_COLS = [
    "danceability", "loudness", "acousticness", "instrumentalness", 
    "valence", "energy", "age", "romantic", "sadness", "music"
]

@st.cache_data
def load_sample_data(max_songs=2000):
    """Load only a fast subset for instant recommendations"""
    df = pd.read_csv(MUSIC_CSV_LOCAL, nrows=max_songs * 2)
    
    # Keep only required columns to save memory
    keep_cols = ["artist_name", "track_name", "genre"] + FEATURE_COLS
    available_cols = [col for col in keep_cols if col in df.columns]
    
    df = df[available_cols].dropna(subset=["artist_name", "track_name"])
    df["song_label"] = df["track_name"] + " - " + df["artist_name"]
    df = df.drop_duplicates("song_label").head(max_songs)
    
    # Features only
    feature_cols = [col for col in FEATURE_COLS if col in df.columns]
    df[feature_cols] = df[feature_cols].fillna(0).astype(np.float32)
    
    X = df[feature_cols].values
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sim_matrix = (X / norms) @ (X / norms).T
    
    return df, sim_matrix

def recommend(df, sim_matrix, song_label, top_k=10):
    idx = df[df["song_label"] == song_label].index[0]
    scores = sim_matrix[idx]
    top_indices = np.argsort(-scores)[:top_k + 1]
    rec_idx = [i for i in top_indices if i != idx][:top_k]
    
    recs = df.iloc[rec_idx][["track_name", "artist_name", "genre"]].copy()
    recs["score"] = scores[rec_idx].round(3)
    return recs

# Main app
st.title("🎵 Fast Music Recommender")
st.caption("Lightweight version - loads 2000 songs instantly")

col1, col2 = st.columns([1, 3])
with col1:
    max_songs = st.slider("Songs to use", 500, 5000, 2000, 500)
    top_k = st.slider("Recommendations", 5, 15, 10)

with st.spinner("Loading fast dataset..."):
    df, sim_matrix = load_sample_data(max_songs)

st.success(f" Loaded {len(df):,} songs using {len([c for c in FEATURE_COLS if c in df.columns])} features")

songs = sorted(df["song_label"].tolist())
selected = st.selectbox("Pick a song:", songs)

if selected:
    # Selected song info
    row = df[df["song_label"] == selected].iloc[0]
    st.subheader("🎧 Selected")
    st.write(f"**{row['track_name']}** by **{row['artist_name']}**")
    st.caption(row.get("genre", "N/A"))
    
    # Recommendations
    st.subheader("🔥 Similar Songs")
    recs = recommend(df, sim_matrix, selected, top_k)
    st.dataframe(recs, use_container_width=True)

st.caption("💡 Lag fixed! Uses sample data + optimized matrix computation")
