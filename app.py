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
    'genre': 0.15,     # Moderate priority for Multi-Genre Overlap (Jaccard)
    'year': 0.05       # Low priority for Year Proximity
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
    return np.exp(-decay_rate * diff)


# ---------------------
# 3. The Hybrid Recommender function (with Multi-Genre Scoring)
# ---------------------
def recommend(movie_title, num_recommendations, weights):
    """
    Generates hybrid recommendations based on enhanced features and fixed optimal weights.
    """
    key = movie_title.strip().lower()
    
    # 1. Fuzzy Matching for resilient input
    if key not in title_to_index:
        close = get_close_matches(key, movie_titles, n=1, cutoff=0.7) 
        if close:
            key = close[0]
            st.info(f"🤔 We think you mean: **{key.title()}**")
        else:
            return None, f"🚫 Movie not found: **{movie_title}**. Please check your spelling."

    idx = title_to_index[key]

    query_vec = embeddings[idx].reshape(1, -1)
    all_sims = cosine_similarity(query_vec, embeddings)[0]

    query_genres_str = movies.loc[idx, "listed_in"]
    query_genres = {g.strip() for g in query_genres_str.split(',') if g.strip()}
    query_year = movies.loc[idx, "release_year"]
    
    candidates = []

    # 2. Score Calculation (using Multi-Genre Jaccard Index)
    for i in range(len(movies)):
        if i == idx:
            continue

        base_sim = all_sims[i]
        
        # --- Multi-Genre Overlap Scoring (Jaccard Similarity) ---
        candidate_genres_str = movies.loc[i, "listed_in"]
        candidate_genres = {g.strip() for g in candidate_genres_str.split(',') if g.strip()}
        
        intersection = len(query_genres.intersection(candidate_genres))
        union = len(query_genres.union(candidate_genres))
        genre_overlap_score = intersection / union if union > 0 else 0.0
        # --- End Multi-Genre Scoring ---
        
        y_score = year_score(query_year, movies.loc[i, "release_year"])

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

    return movies.iloc[indices], top, movies.loc[idx, 'title']


# ---------------------
# 4. Streamlit UI (Fun Version)
# ---------------------

st.title("🍿 Your Cinematic Matchmaker")

st.markdown(
    """
    We dive deep into the plot, cast, and style of what you love to find your next obsession.
    No simple genre tags here—just smart, nuanced recommendations.
    """
)

# --- Input Section ---
user_input = st.text_input(
    "Tell us a movie you're obsessed with...", 
    placeholder="Type a movie title here..."
)

col_slider, col_button = st.columns([2, 1])

with col_slider:
    num_recs = st.slider("How many suggestions do you want?", 1, 15, 7)

with col_button:
    st.markdown("<br>", unsafe_allow_html=True) # Adds a little vertical spacing for alignment
    if st.button("Find My Next Watch 🎬"):
        if not user_input.strip():
            st.warning("Please enter a movie title to get started.")
            st.stop()
        
        # Run the recommender
        result_df, scores, query_title = recommend(user_input, num_recs, WEIGHTS)

        if isinstance(result_df, pd.DataFrame):
            
            st.success(f"Perfect Match! Because you love **{query_title}**, try these:")
            
            # --- Results Display ---
            for rank, ((_, row), (_, total_score, sim, genre_sc, year_sc)) in enumerate(zip(result_df.iterrows(), scores)):
                
                st.markdown(f"## {rank+1}. {row['title']} ({row['release_year']})")
                
                col_meta, col_desc, col_score = st.columns([1.5, 3, 1.5])
                
                with col_meta:
                    st.write(f"**Rating:** {row['rating']}")
                    st.write(f"**Director:** {row['director'].split(',')[0] if row['director'] else 'N/A'}")
                    st.write(f"**Top Cast:** {row['cast'].split(',')[0]}...")
                    st.write(f"**Genres:** {row['listed_in']}")
                    
                with col_desc:
                    st.markdown(f"_{row['description']}_")
                    
                with col_score:
                    # Score Feedback: Focus on the overall match and the top driver
                    st.markdown(f"**Match Score:**")
                    st.progress(total_score)
                    
                    # Determine the main driver of the high score
                    driver_scores = {
                        'Semantic Match': sim,
                        'Multi-Genre Match': genre_sc,
                        'Year Proximity': year_sc
                    }
                    best_match_key = max(driver_scores, key=driver_scores.get)
                    
                    st.caption(f"Strongest Factor: **{best_match_key}**")

                st.markdown("---")
                
            # --- Explainer Section ---
            with st.expander("🔬 How does this work? (The Secret Formula)"):
                st.markdown(
                    """
                    Our Matchmaker uses a sophisticated **Hybrid Filtering** system:
                    
                    1. **Semantic Deep Dive (80% Weight):** We use a language model to understand the *meaning* of the plot, director's style, and cast synergy. This ensures we match on *vibe*, not just genre.
                    2. **Multi-Genre Check (15% Weight):** We check the overlap of **all** listed genres (using Jaccard Similarity) to ensure basic category alignment.
                    3. **Year Vibe (5% Weight):** A slight preference is given to movies released around the same time, accounting for general production trends.
                    """
                )

        else:
            st.error(scores) # Display the error message
