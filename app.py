import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer

# Set Streamlit page config
st.set_page_config(
    page_title="Intelligent Hybrid Recommender",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapse the sidebar by default
)

# --- Define Optimal Weights ---
# We use optimized weights that prioritize the sophisticated semantic vector.
# This makes the app "just work" well for complex movies like Interstellar.
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
    
    # Create the enhanced combined text feature: 
    # Repeat genres for emphasis, include Director/Cast/Country.
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
    movie_titles = list(title_to_index.keys()) # For suggestions
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
        close = get_close_matches(key, movie_titles, n=1, cutoff=0.7) # Slightly stricter cutoff for better suggestions
        if close:
            key = close[0]
            # Inform user of the match
            st.info(f"🔍 Found a close match: **{key.title()}**")
        else:
            return None, f"❌ Could not find a close match for: **{movie_title}**. Try checking your spelling or selecting from the list."

    idx = title_to_index[key]

    query_vec = embeddings[idx].reshape(1, -1)
    all_sims = cosine_similarity(query_vec, embeddings)[0]

    # Get genres for the query movie
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

        # Store the component scores for display
        candidates.append((i, total_score, base_sim, genre_overlap_score, y_score)) 

    # 3. Sorting and Filtering
    candidates.sort(key=lambda x: x[1], reverse=True)

    top = candidates[:num_recommendations]
    indices = [i for i, _, _, _, _ in top]

    return movies.iloc[indices], top, movies.loc[idx, 'title']


# ---------------------
# 4. Streamlit UI (Simplified)
# ---------------------

st.title("🎬 Smart Netflix Movie Recommender")

st.write(
    "Enter a movie you enjoyed, and our system will find highly related titles by combining **semantic deep learning** "
    "on features like **director, cast, and plot** with **multi-genre matching**."
)

st.header("1. Find a Movie")

# Use st.text_input for free-form typing, as requested
user_input = st.text_input(
    "Enter a movie title (e.g., Interstellar, Roma, The Dark Knight)", 
    placeholder="Type a movie title here..."
)

# Add a simple slider for the number of recommendations
num_recs = st.slider("Number of recommendations", 1, 15, 7)

if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please enter a movie title to get started.")
        st.stop()
    
    # Run the recommender with the fixed, optimal weights
    result_df, scores, query_title = recommend(user_input, num_recs, WEIGHTS)

    if isinstance(result_df, pd.DataFrame):
        
        # Find the index of the *matched* query movie 
        key = query_title.strip().lower()
        idx = title_to_index[key]
        
        st.success(f"Because you liked **{query_title}**, we recommend:")
        
        # Display results with detailed scores
        for rank, ((_, row), (_, total_score, sim, genre_sc, year_sc)) in enumerate(zip(result_df.iterrows(), scores)):
            
            st.markdown(f"## {rank+1}. {row['title']} ({row['release_year']})")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Primary Metadata
                st.write(f"**Rating:** {row['rating']}")
                st.write(f"**Director:** {row['director'].split(',')[0] if row['director'] else 'N/A'}")
                st.write(f"**Cast:** {row['cast'].split(',')[0]}...")
                st.write(f"**Genres:** {row['listed_in']}")
                
            with col2:
                # Simplified Score Feedback (Shows the user the scores, but not the weights)
                st.markdown(f"**Relevance Breakdown (Total Score: {total_score:.3f}):**")
                
                # Show normalized weighted scores
                weighted_sim = sim * WEIGHTS['semantic']
                weighted_genre = genre_sc * WEIGHTS['genre']
                weighted_year = year_sc * WEIGHTS['year']
                
                st.caption(
                    f"Semantic Match: **{sim:.3f}** | "
                    f"Multi-Genre Overlap: **{genre_sc:.2f}** | "
                    f"Year Proximity: **{year_sc:.2f}**"
                )
                
                # Use st.progress for a simple visual representation of the final score
                st.progress(total_score)
                
            st.markdown(f"_{row['description']}_")
            st.markdown("---")
            
        # Add a note about the underlying engine in a collapsible expander
        with st.expander("How these recommendations were calculated:"):
            st.markdown(
                f"""
                The final score is a weighted sum using optimal weights:
                * **Semantic Similarity ({WEIGHTS['semantic']:.2f})**: Uses deep learning (Sentence Transformers) on the movie's plot, director, and top cast to find similar *themes* and *styles*.
                * **Multi-Genre Overlap ({WEIGHTS['genre']:.2f})**: Uses Jaccard similarity to measure how many of the movie's total genres match the suggested movie.
                * **Year Proximity ({WEIGHTS['year']:.2f})**: Gently favors movies released close to the original movie's year.
                """
            )


    else:
        st.error(scores) # Display the error message
