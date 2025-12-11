import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer

# Set Streamlit page config
st.set_page_config(
    page_title="Cinematic Matchmaker",
    layout="wide", # Uses the full width
    initial_sidebar_state="collapsed"
)

# --- Define Optimal Weights ---
OPTIMAL_WEIGHTS = {
    'semantic': 0.80, # High priority for enhanced semantic vector
    'genre': 0.15,   # Moderate priority for Multi-Genre Overlap (Jaccard)
    'year': 0.05     # Low priority for Year Proximity
}
TOTAL_W = sum(OPTIMAL_WEIGHTS.values())
WEIGHTS = {k: v / TOTAL_W for k, v in OPTIMAL_WEIGHTS.items()}


# ---------------------
# 1. Load data + model (with caching)
# ---------------------

@st.cache_data
def load_data():
    """Loads and preprocesses the Netflix data, creating a rich combined text column."""
    try:
        df = pd.read_csv("netflix_titles.csv")
    except FileNotFoundError:
        st.error("Fatal Error: 'netflix_titles.csv' not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()

    movies = df[df["type"] == "Movie"].copy().reset_index(drop=True)

    # CRITICAL FIX: Convert year to integer, coercing errors to NaN and filling with a placeholder
    movies["release_year"] = pd.to_numeric(movies["release_year"], errors='coerce').fillna(2000).astype(int)

    for col in ["description", "listed_in", "director", "cast", "country"]:
        movies[col] = movies[col].fillna("")

    # --- Feature Engineering for Better Embeddings ---
    def clean_and_limit_cast(cast_str):
        # Cleans and takes top 3 cast members
        return " ".join(cast_str.split(",")[:3]).replace(",", " ").strip()

    movies["all_genres"] = movies["listed_in"].apply(lambda x: x.replace(",", " "))
    
    # Create the enhanced combined text feature
    movies["combined"] = (
        movies["all_genres"] + " " + movies["all_genres"] + " " + movies["all_genres"] + " " +
        movies["description"] + " " +
        movies["director"].apply(lambda x: x.replace(", ", " ").strip()) + " " +
        movies["country"].apply(lambda x: x.replace(", ", " ").strip()) + " " +
        movies["cast"].apply(clean_and_limit_cast)
    )
    # --- End Feature Engineering ---

    return movies


@st.cache_resource
def load_model():
    """Loads the Sentence Transformer model (cached for performance)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def compute_embeddings(texts):
    """Computes and analyzes the embeddings (cached)."""
    model = load_model()
    embeddings = model.encode(
        texts.tolist(),
        show_progress_bar=False,
        normalize_embeddings=True 
    )
    return embeddings


movies = load_data()

# Check for empty data before proceeding
if movies.empty:
    st.error("Cannot proceed. The data file was not loaded correctly.")
    st.stop()
    
# Proceed only if data is successfully loaded
embeddings = compute_embeddings(movies["combined"])
title_to_index = {movies.loc[i, "title"].lower(): i for i in movies.index}
movie_titles = list(title_to_index.keys()) 

# ---------------------
# 2. Helper for year score (Exponential decay)
# ---------------------
def year_score(query_year, candidate_year, decay_rate=0.15):
    """Calculates a score based on year proximity using exponential decay."""
    diff = abs(query_year - candidate_year)
    return np.exp(-decay_rate * diff)


# ---------------------
# 3. The Hybrid Recommender function
# ---------------------
def recommend(movie_title, num_recommendations, weights):
    """
    Generates hybrid recommendations. 
    Handles movie titles (match or fuzzy match) or treats the input as a raw query.
    """
    key = movie_title.strip().lower()
    query_title = movie_title
    query_genres = set()
    query_year = None
    query_source = "Raw Query"
    exclude_idx = -1
    
    # 1. Determine the query vector and source
    if key in title_to_index:
        idx = title_to_index[key]
        query_vec = embeddings[idx].reshape(1, -1)
        query_title = movies.loc[idx, 'title']
        query_genres_str = movies.loc[idx, "listed_in"]
        query_genres = {g.strip() for g in query_genres_str.split(',') if g.strip()}
        query_year = movies.loc[idx, "release_year"]
        query_source = f"Match: {query_title}"
        exclude_idx = idx

    else:
        # Fuzzy Matching attempt
        close = get_close_matches(key, movie_titles, n=1, cutoff=0.6) 
        if close:
            key = close[0]
            idx = title_to_index[key]
            query_vec = embeddings[idx].reshape(1, -1)
            query_title = movies.loc[idx, 'title']
            query_genres_str = movies.loc[idx, "listed_in"]
            query_genres = {g.strip() for g in query_genres_str.split(',') if g.strip()}
            query_year = movies.loc[idx, "release_year"]
            query_source = f"Fuzzy Match: {query_title}"
            exclude_idx = idx
            st.info(f"🤔 We think you mean: **{query_title}**. Generating hybrid recommendations based on this title.")
        
        else:
            # Raw Query Path
            st.warning(f"💡 Movie **'{movie_title}'** not found. Generating recommendations based on your description/query.")
            model = load_model()
            query_vec = model.encode([movie_title.strip()], normalize_embeddings=True).reshape(1, -1)
            query_source = f"Raw Query: '{movie_title.strip()}'"

    # 2. Score Calculation
    all_sims = cosine_similarity(query_vec, embeddings)[0]
    
    candidates = []

    for i in range(len(movies)):
        if i == exclude_idx:
            continue

        base_sim = all_sims[i]
        
        # Jaccard/Year Scoring is only applied if the query was a found movie (Hybrid Mode)
        if exclude_idx != -1: 
            candidate_genres_str = movies.loc[i, "listed_in"]
            candidate_genres = {g.strip() for g in candidate_genres_str.split(',') if g.strip()}
            
            intersection = len(query_genres.intersection(candidate_genres))
            union = len(query_genres.union(candidate_genres))
            genre_overlap_score = intersection / union if union > 0 else 0.0
            
            y_score = year_score(query_year, movies.loc[i, "release_year"])
        else:
            # If Raw Query, these factors are ignored (score is 0)
            genre_overlap_score = 0.0
            y_score = 0.0

        # Hybrid Score: using defined optimal weights
        total_score = (
            weights['semantic'] * base_sim + 
            weights['genre'] * genre_overlap_score +
            weights['year'] * y_score
        )
        
        candidates.append((i, total_score, base_sim, genre_overlap_score, y_score)) 

    # 3. Sorting and Filtering
    candidates.sort(key=lambda x: x[1], reverse=True)

    top = candidates[:num_recommendations]
    indices = [i for i, _, _, _, _ in top]

    return movies.iloc[indices], top, query_title, query_source

# ---------------------
# 4. Streamlit UI (Fun & Stacked Version)
# ---------------------

# Centering Fix: Use columns to create a central content area (4/6ths width)
col_left, col_center, col_right = st.columns([1, 4, 1])

with col_center:
    st.title("🍿 Your Cinematic Matchmaker")

    st.markdown(
        """
        We dive deep into the plot, cast, and style of what you love to find your next obsession.
        **Enter a movie title or even a descriptive phrase** (e.g., "A gritty action film from the 2000s").
        """
    )

    # --- Input Section ---
    user_input = st.text_input(
        "Tell us a movie you're obsessed with...", 
        placeholder="Type a movie title or a descriptive query here..."
    )

    # Use a nested column set for the slider and button, keeping them within the center column
    col_slider, col_button = st.columns([2, 1])

    with col_slider:
        num_recs = st.slider("How many suggestions do you want?", 1, 15, 7)

    # Handle the button click logic
    with col_button:
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("Find My Next Watch 🎬"):
            if not user_input.strip():
                st.warning("Please enter a movie title or a query to get started.")
            
            else:
                # Run the recommender
                result_df, scores, query_title, query_source = recommend(user_input, num_recs, WEIGHTS)

                st.success(f"**Recommendations based on:** {query_source}")
                
                # --- Results Display (STACKED LAYOUT) ---
                for rank, ((_, row), (_, total_score, sim, genre_sc, year_sc)) in enumerate(zip(result_df.iterrows(), scores)):
                    
                    st.markdown(f"## {rank+1}. {row['title']} ({row['release_year']})")
                    
                    # Metadata section
                    st.write(f"**Rating:** {row['rating']} | **Director:** {row['director'].split(',')[0] if row['director'] else 'N/A'} | **Top Cast:** {row['cast'].split(',')[0]}... | **Genres:** {row['listed_in']}")

                    # Description
                    st.markdown(f"**Summary:** _{row['description']}_")

                    # Score Feedback section (simplified)
                    st.markdown(f"**Relevance Score: {total_score:.3f}**")
                    st.progress(total_score)
                    
                    # Determine the main driver of the high score
                    driver_scores = {
                        'Semantic Match': sim,
                        'Multi-Genre Match': genre_sc,
                        'Year Proximity': year_sc
                    }
                    
                    # Check if we are in Hybrid Mode (i.e., a movie was found)
                    if 'Match' in query_source:
                        best_match_key = max(driver_scores, key=driver_scores.get)
                        st.caption(f"Strongest Factor: **{best_match_key}** (Semantic: {sim:.3f}, Multi-Genre: {genre_sc:.2f})")
                    else:
                        st.caption(f"Search Type: **Semantic Deep Dive** (Match Score: {sim:.3f})")

                    st.markdown("---")
                    
                # --- Explainer Section ---
                with st.expander("🔬 How does this work? (The Secret Formula)"):
                    st.markdown(
                        """
                        Our Matchmaker uses a sophisticated **Hybrid Filtering** system, falling back to a powerful **Semantic Search** if a specific movie is not found.
                        
                        **Hybrid Mode (Movie Match/Fuzzy Match):**
                        1. **Semantic Deep Dive (80% Weight):** Uses a language model to find similar *themes* and *styles*.
                        2. **Multi-Genre Check (15% Weight):** Checks the Jaccard overlap of all genres.
                        3. **Year Vibe (5% Weight):** Gently favors nearby release years.
                        
                        **Semantic Search Mode (Raw Query):**
                        If your input is a general phrase or a movie not in the dataset, it uses **Semantic Deep Dive (1)** alone for the best conceptual match.
                        """
                    )
