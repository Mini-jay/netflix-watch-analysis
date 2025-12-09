import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ---------------------
# Load data
# ---------------------
# CSV is in the same folder as app.py on Streamlit Cloud
df = pd.read_csv("netflix_titles.csv")

# Only movies (you can change this later)
movies = df[df["type"] == "Movie"].copy().reset_index(drop=True)

# Fill missing fields
for col in ["description", "listed_in"]:
    movies[col] = movies[col].fillna("")

# Primary genre = first in the comma-separated list
movies["primary_genre"] = movies["listed_in"].apply(
    lambda x: x.split(",")[0].strip() if isinstance(x, str) and x.strip() else "Unknown"
)

# Combined text = description + all genres
movies["combined"] = movies["description"] + " " + movies["listed_in"]

# ---------------------
# TF-IDF vectorizer
# ---------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined"])

cosine_sim = cosine_similarity(tfidf_matrix)

# Lookup: title (lowercase) -> index
title_to_index = {movies.loc[i, "title"].lower(): i for i in movies.index}


# ---------------------
# Recommender (genre-aware)
# ---------------------
def recommend(movie_title, num_recommendations=5):
    key = movie_title.strip().lower()

    # exact match or fuzzy match
    if key not in title_to_index:
        close = get_close_matches(key, title_to_index.keys(), n=1, cutoff=0.4)
        if close:
            key = close[0]
        else:
            return None, f"❌ Could not find: {movie_title}"

    idx = title_to_index[key]

    # base similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # query genre
    query_genre = movies.loc[idx, "primary_genre"]

    # separate candidates: same-genre and others
    same_genre = []
    other_genre = []

    for i, score in sim_scores:
        if i == idx:
            continue  # skip itself
        if movies.loc[i, "primary_genre"] == query_genre:
            same_genre.append((i, score))
        else:
            other_genre.append((i, score))

    # sort each list by similarity score (desc)
    same_genre.sort(key=lambda x: x[1], reverse=True)
    other_genre.sort(key=lambda x: x[1], reverse=True)

    # take from same genre first
    chosen = same_genre[:num_recommendations]

    # if not enough, fill from others
    if len(chosen) < num_recommendations:
        remaining = num_recommendations - len(chosen)
        chosen += other_genre[:remaining]

    indices = [i for i, _ in chosen]
    return movies.iloc[indices], None


# ---------------------
# Streamlit UI
# ---------------------
st.title("🎬 Netflix Movie Recommender")

st.write(
    "Recommendations based on description + genres, "
    "with a preference for movies in the same main genre."
)

user_input = st.text_input("Enter a movie title")

num_recs = st.slider("Number of recommendations", 1, 10, 5, 5)

if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please enter a movie title")
    else:
        result, error = recommend(user_input, num_recs)

        if error:
            st.error(error)
        else:
            query_idx = title_to_index.get(user_input.strip().lower())
            if query_idx is not None:
                st.info(f"Detected primary genre: **{movies.loc[query_idx, 'primary_genre']}**")

            st.success(f"Because you watched **{user_input}**:")

            for _, row in result.iterrows():
                st.markdown(f"### 🎞 {row['title']} ({row['release_year']})")
                st.write(f"**Primary genre:** {row['primary_genre']}")
                st.write(f"**All genres:** {row['listed_in']}")
                st.write(f"**Rating:** {row['rating']}")
                st.write(row["description"])
                st.write("---")
