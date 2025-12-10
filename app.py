import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer

# Set Streamlit page config
st.set_page_config(
    page_title="Cinematic Matchmaker",
    layout="wide",
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
        # NOTE: Assumes netflix_titles.csv is present
        df = pd.read_csv("netflix_titles.csv") 
    except FileNotFoundError:
        st.error("Error: 'netflix_titles.csv' not found. Please ensure the file is in the same directory.")
        return pd.DataFrame() 

    movies = df[df["type"] == "Movie"].copy().reset_index(drop=True)

    for col in ["description", "listed_in", "director", "cast", "country"]:
        movies[col] = movies[col].fillna("")

    # --- Feature Engineering for Better Embeddings ---
    def clean_and_limit_cast(cast_str):
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
    # 
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

if not movies.empty:
    embeddings = compute_embeddings(movies["combined"])
    title_to_index = {movies.loc[i, "title"].lower(): i for i in movies.index}
    movie_titles = list(title_to_index.keys()) 
else:
    st.stop() 

# ---------------------
# 2. Helper for year score (Exponential decay)
# ---------------------
def year_score(query_year, candidate_year, decay_rate=0.15):
    """Calculates a score based on year proximity using exponential decay."""
    if pd.isna(query_year) or pd.isna(candidate_year):
        return 0.0
    diff = abs(query_year - candidate_year)
    # The exponential decay function: score = e^(-decay_rate * difference)
    return np.exp(-decay_rate * diff)
    # 

[Image of Exponential Decay Function Plot]



# ---------------------
# 3. The Hybrid Recommender function (with New Query Path)
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
    query_source = "Movie Not Found"
    
    # 1. Try to find the movie title (with fuzzy matching)
    if key in title_to_index:
        idx = title_to_index[key]
        query_vec = embeddings[idx].reshape(1, -1)
        query_title = movies.loc[idx, 'title']
        query_genres_str = movies.loc[idx, "listed_in"]
        query_genres = {g.strip() for g in query_genres_str.split(',') if g.strip()}
        query_year = movies.loc[idx, "release_year"]
        query_source = f"Match: {query_title}"

    else:
        # Fuzzy Matching (Cutoff changed from 0.7 to 0.6 for better robustness)
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
            st.info(f"🤔 We think you mean: **{query_title}**")
        
        else:
            # --- NEW PATH: Treat input as a raw query ---
            st.warning(f"💡 Movie **{movie_title}** not found. Generating recommendations based on your description/query.")
            model = load_model()
            # Compute the embedding for the raw query text
            query_vec = model.encode([movie_title.strip()], normalize_embeddings=True).reshape(1, -1)
            query_source = f"Raw Query: '{movie_title.strip()}'"

    # 2. Score Calculation
    all_sims = cosine_similarity(query_vec, embeddings)[0]
    
    candidates = []
    
    # Check if a specific movie was found to exclude it from recommendations
    exclude_idx = title_to_index.get(key) if 'Match' in query_source else -1

    for i in range(len(movies)):
        if i == exclude_idx: # Skip the movie itself if it was found
            continue

        base_sim = all_sims[i]
        
        # --- Multi-Genre Overlap Scoring (Jaccard Similarity) ---
        candidate_genres_str = movies.loc[i, "listed_in"]
        candidate_genres = {g.strip() for g in candidate_genres_str.split(',') if g.strip()}
        
        # Jaccard only applies if the query was a found movie. Otherwise, genre is 0.
        if query_genres: 
            intersection = len(query_genres.intersection(candidate_genres))
            union = len(query_genres.union(candidate_genres))
            genre_overlap_score = intersection / union if union > 0 else 0.0
        else:
            genre_overlap_score = 0.0
        
        # Year Proximity only applies if the query was a found movie. Otherwise, year is 0.
        y_score = year_score(query_year, movies.loc[i, "release_year"]) if query_year else 0.0

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

col_slider, col_button = st.columns([2, 1])

with col_slider:
    num_recs = st.slider("How many suggestions do you want?", 1, 15, 7)

# Handle the button click logic
with col_button:
    st.markdown("<br>", unsafe_allow_html=True) 
    if st.button("Find My Next Watch 🎬"):
        if not user_input.strip():
            st.warning("Please enter a movie title or a query to get started.")
            # st.stop() # Removed st.stop() for cleaner user experience
            # No need to stop, the app will continue below the button block
        
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
                # Check if genre and year contributed (they won't for raw query)
                is_hybrid = True if (genre_sc > 0 or year_sc > 0) else False
                
                if is_hybrid:
                    best_match_key = max(driver_scores, key=driver_scores.get)
                    st.caption(f"Strongest Factor: **{best_match_key}** (Semantic: {sim:.3f}, Multi-Genre: {genre_sc:.2f})")
                else:
                    st.caption(f"Strongest Factor: **Semantic Match** (Raw query search: {sim:.3f})")

                st.markdown("---")
                
            # --- Explainer Section ---
            with st.expander("🔬 How does this work? (The Secret Formula)"):
                st.markdown(
                    """
                    Our Matchmaker uses a sophisticated **Hybrid Filtering** system, falling back to a **Semantic Search** if a specific movie is not found.
                    
                    1. **Semantic Deep Dive (80% Weight):** Uses a language model to find similar *themes* and *styles* based on plot, director, and cast.
                    2. **Multi-Genre Check (15% Weight):** Checks the overlap of **all** listed genres to ensure basic category alignment (Only used when a movie is found).
                    3. **Year Vibe (5% Weight):** Gently favors movies released close to the original movie's year (Only used when a movie is found).
                    
                    **If your input is a general phrase or a movie not in the dataset, it uses Semantic Deep Dive (1) alone.**
                    """
                )

Would you like to know more about the **Sentence Transformer** model used for the semantic deep dive?
