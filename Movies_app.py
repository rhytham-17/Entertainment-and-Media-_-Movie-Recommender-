import os
import zipfile
import urllib.request
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Data", "MovieLens")

MOVIES_CSV_LOCAL = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV_LOCAL = os.path.join(DATA_DIR, "ratings.csv")

# This is now your renamed filtered dataset
FILTERED_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "movies_data.csv")

ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_LATEST_SMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")


def _ensure_data_downloaded() -> Tuple[str, str]:
    if os.path.exists(MOVIES_CSV_LOCAL) and os.path.exists(RATINGS_CSV_LOCAL):
        return MOVIES_CSV_LOCAL, RATINGS_CSV_LOCAL

    os.makedirs(DATA_DIR, exist_ok=True)

    extracted_movies = os.path.join(ML_LATEST_SMALL_DIR, "movies.csv")
    extracted_ratings = os.path.join(ML_LATEST_SMALL_DIR, "ratings.csv")
    if os.path.exists(extracted_movies) and os.path.exists(extracted_ratings):
        return extracted_movies, extracted_ratings

    zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")
    with st.spinner("Downloading MovieLens (ml-latest-small) ..."):
        urllib.request.urlretrieve(ML_LATEST_SMALL_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    if not (os.path.exists(extracted_movies) and os.path.exists(extracted_ratings)):
        raise FileNotFoundError("MovieLens download completed, but CSVs were not found.")

    return extracted_movies, extracted_ratings


@st.cache_data(show_spinner=False)
def load_and_prepare_data(
    top_n_movies: int,
    top_n_users: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:

    # Use your renamed filtered file directly
    if os.path.exists(FILTERED_DATA_CSV_LOCAL):
        df = pd.read_csv(FILTERED_DATA_CSV_LOCAL)

        if not {"userId", "title", "rating"}.issubset(df.columns):
            raise ValueError(
                "movies_data.csv must contain at least userId, title, and rating."
            )

        genres_map = {}
        if "genres" in df.columns:
            genres_map = (
                df[["title", "genres"]]
                .drop_duplicates(subset=["title"])
                .set_index("title")["genres"]
                .to_dict()
            )

        user_movie_matrix = (
            df.pivot_table(index="userId", columns="title", values="rating", aggfunc="mean")
            .fillna(0)
        )

        movies = df[["title"]].drop_duplicates()
        return df, movies, user_movie_matrix, genres_map

    # Fallback: build from raw MovieLens files
    movies_path, ratings_path = _ensure_data_downloaded()
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    df = pd.merge(ratings, movies, on="movieId")
    df.dropna(inplace=True)

    top_movies = df["title"].value_counts().head(top_n_movies).index
    df = df[df["title"].isin(top_movies)]

    active_users = df["userId"].value_counts().head(top_n_users).index
    df = df[df["userId"].isin(active_users)]

    genres_map = (
        movies[["title", "genres"]]
        .drop_duplicates(subset=["title"])
        .set_index("title")["genres"]
        .to_dict()
    )

    user_movie_matrix = (
        df.pivot_table(index="userId", columns="title", values="rating", aggfunc="mean")
        .fillna(0)
    )

    return df, movies, user_movie_matrix, genres_map


@st.cache_resource(show_spinner=True)
def build_recommender(
    top_n_movies: int,
    top_n_users: int,
) -> Tuple[np.ndarray, Dict[str, str], pd.Index, Dict[str, int]]:

    _, _, user_movie_matrix, genres_map = load_and_prepare_data(
        top_n_movies=top_n_movies, top_n_users=top_n_users
    )

    X = user_movie_matrix.to_numpy(dtype=np.float32)
    col_norms = np.linalg.norm(X, axis=0)
    col_norms[col_norms == 0] = 1e-8
    Xn = X / col_norms
    sim_matrix = Xn.T @ Xn

    movie_titles = user_movie_matrix.columns
    title_to_index = {title: i for i, title in enumerate(movie_titles)}
    return sim_matrix, genres_map, movie_titles, title_to_index


def recommend_from_similarity(
    sim_matrix: np.ndarray,
    movie_titles: pd.Index,
    genres_map: Dict[str, str],
    title_to_index: Dict[str, int],
    movie_name: str,
    top_k: int,
) -> pd.DataFrame:
    if movie_name not in movie_titles:
        return pd.DataFrame([{"movie": movie_name, "score": 0.0, "genres": genres_map.get(movie_name, "")}])

    movie_index = title_to_index[movie_name]
    scores = sim_matrix[movie_index]

    order = np.argsort(-scores)
    top_indices = [i for i in order.tolist() if i != movie_index][:top_k]

    rec_titles = movie_titles[top_indices]
    rec_scores = scores[top_indices]
    rec_df = pd.DataFrame({"movie": rec_titles, "score": rec_scores})
    rec_df["genres"] = rec_df["movie"].map(lambda t: genres_map.get(t, ""))
    return rec_df


def main() -> None:
    st.set_page_config(page_title="RecoMind", page_icon="🎬", layout="centered")
    st.title("RecoMind - Movie Recommendations")
    st.caption("Collaborative filtering using cosine similarity over user ratings.")

    with st.sidebar:
        st.header("Recommendation Settings")
        if os.path.exists(FILTERED_DATA_CSV_LOCAL):
            st.markdown(
                "**Speed mode**: Found `movies_data.csv` in `Data/MovieLens` and will load it directly."
            )
        else:
            st.markdown(
                "**Speed mode**: If you create `Data/MovieLens/movies_data.csv`, model building becomes much faster."
            )

        with st.form("train_form"):
            top_n_movies = st.slider(
                "Top Movies (training filter)",
                min_value=50,
                max_value=300,
                value=100,
                step=10,
            )
            top_n_users = st.slider(
                "Active Users (training filter)",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
            )
            build_submit = st.form_submit_button("Build/Refresh model")

        top_k = st.slider("Recommendations to show", min_value=5, max_value=30, value=10, step=1)

    if "last_built_params" not in st.session_state or build_submit:
        st.session_state["last_built_params"] = {"top_n_movies": top_n_movies, "top_n_users": top_n_users}

    last = st.session_state.get("last_built_params", {"top_n_movies": 100, "top_n_users": 500})
    sim_matrix, genres_map, movie_titles, title_to_index = build_recommender(
        top_n_movies=last["top_n_movies"],
        top_n_users=last["top_n_users"],
    )

    selected_movie = st.selectbox("Pick a movie", options=sorted(movie_titles.tolist()))

    st.divider()
    st.subheader("Top Recommendations")

    recs = recommend_from_similarity(
        sim_matrix=sim_matrix,
        movie_titles=movie_titles,
        movie_name=selected_movie,
        genres_map=genres_map,
        title_to_index=title_to_index,
        top_k=top_k,
    )

    st.dataframe(recs, use_container_width=True)
    if selected_movie in genres_map:
        st.info(f"Selected movie genres: {genres_map.get(selected_movie, '')}")


if __name__ == "__main__":
    main()