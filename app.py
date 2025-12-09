import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# ---------------------
# Load data
# ---------------------
df = pd.read_csv("netflix_titles.csv")

# Only movies (you can change to TV Show or both later)
movies = df[df["type"] == "Movie"].copy().reset_index(drop=True)

# Fill missing text fields
for col in ["description", "listed_in", "cast", "director"]:
    movies[col] = movies[col].fillna("")

# Normalize genres as list
movies["genres_list"] = movies["listed_in"].str.split(", ").fillna([])

# Combined text: description + genres + cast + director
movies["combined"] = (
    movies["description"]
    + " "
    + movies["listed_in"]
    + " "
    + movies["cast"]
    + " "
    + movies["director"]
)

# ---------------------
# TF-IDF vectorizer
# ---------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined"])

cosine_sim = cosine_similarity(tfidf_matrix)

# index lookup
title_to_index = {movies.loc[i, "title"].lower(): i for i in movies.index}


# ---------------------
# recommender function (with genre-aware reranking)
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

    # get query genres
    query_genres = set(movies.loc[idx, "genres_list"])

    ranked = []
    for i, score in sim_scores:
        if i == idx:
            continue  # skip itself
        target_genres = set(movies.loc[i, "genres_list"])
        genre_overlap = len(query_genres & target_genres)
        ranked.append((i, score, genre_overlap))

    # 1) sort by: genre overlap (desc), then similarity score (desc)
    ranked.sort(key=lambda x: (x[2], x[1]), reverse=True)

    # if everything has 0 genre overlap, fall back to pure similarity
    if all(r[2] == 0 for r in ranked):
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

    # take top N
    top = ranked[:num_recommendations]
    indices = [i for i, _, _ in top]

    return movies.iloc[indices], None


# ---------------------
# Streamlit UI
# ---------------------
st.title("🎬 Netflix Movie Recommender (v2.0)")

st.write(
    "Content-based recommendations using description, genres, cast & director, "
    "with genre-aware reranking."
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
            st.success(f"Because you watched **{user_input}**:")

            for _, row in result.iterrows():
                st.markdown(f"### 🎞 {row['title']} ({row['release_year']})")
                st.write(f"**Genres:** {row['listed_in']}")
                st.write(f"**Rating:** {row['rating']}")
                st.write(row["description"])
                st.write("---")

