import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer

# ---------------------
# Load data + model (with caching)
# ---------------------

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    movies = df[df["type"] == "Movie"].copy().reset_index(drop=True)

    # fill missing text stuff
    for col in ["description", "listed_in"]:
        movies[col] = movies[col].fillna("")

    # primary genre (first label)
    movies["primary_genre"] = movies["listed_in"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) and x.strip() else "Unknown"
    )

    # combined text for embeddings
    movies["combined"] = movies["description"] + " " + movies["listed_in"]

    return movies


@st.cache_resource
def load_model():
    # small, fast sentence transformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def compute_embeddings(texts):
    model = load_model()
    embeddings = model.encode(
        texts.tolist(),
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


movies = load_data()
embeddings = compute_embeddings(movies["combined"])
title_to_index = {movies.loc[i, "title"].lower(): i for i in movies.index}


# ---------------------
# helper for year score
# ---------------------
def year_score(query_year, candidate_year, max_diff=25):
    if pd.isna(query_year) or pd.isna(candidate_year):
        return 0.0
    diff = abs(candidate_year - candidate_year)
    if diff >= max_diff:
        return 0.0
    return 1.0 - (diff / max_diff)


# ---------------------
# recommender using embeddings
# ---------------------
def recommend(movie_title, num_recommendations=5):
    key = movie_title.strip().lower()

    # exact or fuzzy match
    if key not in title_to_index:
        close = get_close_matches(key, title_to_index.keys(), n=1, cutoff=0.4)
        if close:
            key = close[0]
        else:
            return None, f"❌ Could not find: {movie_title}"

    idx = title_to_index[key]

    query_vec = embeddings[idx].reshape(1, -1)
    all_sims = cosine_similarity(query_vec, embeddings)[0]

    query_genre = movies.loc[idx, "primary_genre"]
    query_year = movies.loc[idx, "release_year"]

    candidates = []

    for i in range(len(movies)):
        if i == idx:
            continue

        base_sim = all_sims[i]
        same_genre = 1.0 if movies.loc[i, "primary_genre"] == query_genre else 0.0
        y_score = year_score(query_year, movies.loc[i, "release_year"])

        # You can tune these weights:
        total = 0.7 * base_sim + 0.25 * same_genre + 0.05 * y_score

        candidates.append((i, total, base_sim, same_genre, y_score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    top = candidates[:num_recommendations]
    indices = [i for i, _, _, _, _ in top]

    return movies.iloc[indices], None


# ---------------------
# Streamlit UI
# ---------------------
st.title("🎬 Netflix Movie Recommender – Semantic Version")

st.write(
    "Uses SentenceTransformer embeddings (all-MiniLM-L6-v2) on description + genres, "
    "plus genre & year awareness for smarter recommendations."
)

user_input = st.text_input("Enter a movie title")

num_recs = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please enter a movie title")
    else:
        result, error = recommend(user_input, num_recs)

        if error:
            st.error(error)
        else:
            key = user_input.strip().lower()
            idx = title_to_index.get(key)
            if idx is not None:
                st.info(
                    f"Detected primary genre: **{movies.loc[idx, 'primary_genre']}**, "
                    f"Year: **{movies.loc[idx, 'release_year']}**"
                )

            st.success(f"Because you watched **{user_input}**:")

            for _, row in result.iterrows():
                st.markdown(f"### 🎞 {row['title']} ({row['release_year']})")
                st.write(f"**Primary genre:** {row['primary_genre']}")
                st.write(f"**All genres:** {row['listed_in']}")
                st.write(f"**Rating:** {row['rating']}")
                st.write(row["description"])
                st.write("---")
