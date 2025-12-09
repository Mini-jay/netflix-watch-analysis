import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches


# ---------------------
# Load data
# ---------------------
df = pd.read_csv("netflix_titles.csv")

movies = df[df["type"] == "Movie"].copy().reset_index(drop=True)

# combine text fields (description + genre)
movies["combined"] = (
    movies["description"].fillna("") + " " +
    movies["listed_in"].fillna("")
)


# ---------------------
# TF-IDF vectorizer
# ---------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined"])

cosine_sim = cosine_similarity(tfidf_matrix)

# index lookup
title_to_index = {
    movies.loc[i, "title"].lower(): i for i in movies.index
}


# ---------------------
# recommender function
# ---------------------
def recommend(movie_title, num_recommendations=5):

    key = movie_title.strip().lower()

    # exact match?
    if key not in title_to_index:
        close = get_close_matches(key, title_to_index.keys(), n=1, cutoff=0.4)

        if close:
            key = close[0]
        else:
            return None, f"❌ Could not find: {movie_title}"

    idx = title_to_index[key]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # skip itself
    sim_scores = sim_scores[1:num_recommendations+1]

    indices = [i for i, _ in sim_scores]

    return movies.iloc[indices], None


# ---------------------
# Streamlit UI
# ---------------------
st.title("🎬 Netflix Movie Recommender")

st.write("Search for any movie and get similar recommendations!")


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
                st.write(row['description'])
                st.write("---")
