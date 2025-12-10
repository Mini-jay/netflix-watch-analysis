import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer

# Set Streamlit page config
st.set_page_config(
    page_title="Advanced Netflix Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        return pd.DataFrame() # Return empty DataFrame on failure

    movies = df[df["type"] == "Movie"].copy().reset_index(drop=True)

    # 1. Fill missing values for text features
    for col in ["description", "listed_in", "director", "cast", "country"]:
        movies[col] = movies[col].fillna("")

    # 2. Extract primary genre (first label)
    movies["primary_genre"] = movies["listed_in"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) and x.strip() else "Unknown"
    )

    # 3. Enhanced 'combined' text for embeddings (Including Director, Top 3 Cast, and Country)
    def clean_and_limit_cast(cast_str):
        # Take the top 3 actors and join with spaces
        return " ".join(cast_str.split(",")[:3]).replace(",", " ").strip()

    movies["combined"] = (
        movies["description"] + " " +
        movies["listed_in"] + " " +
        movies["director"].apply(lambda x: x.replace(", ", " ").strip()) + " " + # Director names as keywords
        movies["country"].apply(lambda x: x.replace(", ", " ").strip()) + " " + # Country names as keywords
        movies["cast"].apply(clean_and_limit_cast)
    )

    return movies


@st.cache_resource
def load_model():
    """Loads the Sentence Transformer model (cached for performance)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def compute_embeddings(texts):
    """Computes and normalizes the embeddings (cached)."""
    model = load_model()
    embeddings = model.encode(
        texts.tolist(),
        show_progress_bar=False, # Streamlit hides console progress, so set to False
        normalize_embeddings=True # Normalizing is crucial for cosine similarity
    )
    return embeddings


movies = load_data()

if not movies.empty:
    embeddings = compute_embeddings(movies["combined"])
    # Create the title map AFTER loading data
    title_to_index = {movies.loc[i, "title"].lower(): i for i in movies.index}
else:
    st.stop() # Stop execution if data failed to load

# ---------------------
# 2. Helper for year score (Using non-linear decay)
# ---------------------
def year_score(query_year, candidate_year, decay_rate=0.15):
    """Calculates a score based on year proximity using exponential decay."""
    if pd.isna(query_year) or pd.isna(candidate_year):
        return 0.0
    # Use 0.15 for a moderate decay. Lower is flatter, higher is steeper.
    diff = abs(query_year - candidate_year)
    return np.exp(-decay_rate * diff)


# ---------------------
# 3. The Hybrid Recommender function
# ---------------------
def recommend(movie_title, num_recommendations, weights):
    """
    Generates hybrid recommendations based on semantic similarity, genre, and year.
    
    Args:
        movie_title (str): The movie the user liked.
        num_recommendations (int): Number of recommendations to return.
        weights (dict): A dictionary of weights for the hybrid score components.
    """
    key = movie_title.strip().lower()

    # 1. Fuzzy Matching
    if key not in title_to_index:
        close = get_close_matches(key, title_to_index.keys(), n=1, cutoff=0.6) # Increased cutoff
        if close:
            key = close[0]
        else:
            return None, f"❌ Could not find a close match for: **{movie_title}**"

    idx = title_to_index[key]

    # Get the embedding for the query movie
    query_vec = embeddings[idx].reshape(1, -1)
    
    # Calculate cosine similarity with all other movies
    all_sims = cosine_similarity(query_vec, embeddings)[0]

    query_genre = movies.loc[idx, "primary_genre"]
    query_year = movies.loc[idx, "release_year"]
    
    candidates = []

    # 2. Score Calculation and Hybrid Weighting
    for i in range(len(movies)):
        if i == idx:
            continue

        base_sim = all_sims[i]
        
        # Binary genre score (1.0 if primary genre matches, 0.0 otherwise)
        same_genre = 1.0 if movies.loc[i, "primary_genre"] == query_genre else 0.0
        
        # Year score (using the exponential decay function)
        y_score = year_score(query_year, movies.loc[i, "release_year"])

        # Hybrid Score: using user-defined weights
        total_score = (
            weights['semantic'] * base_sim + 
            weights['genre'] * same_genre + 
            weights['year'] * y_score
        )

        candidates.append((i, total_score, base_sim, same_genre, y_score))

    # 3. Sorting and Filtering
    candidates.sort(key=lambda x: x[1], reverse=True)

    top = candidates[:num_recommendations]
    indices = [i for i, _, _, _, _ in top]

    # Return the results dataframe and the scores for detailed feedback
    return movies.iloc[indices], top, movies.loc[idx, 'title']


# ---------------------
# 4. Streamlit UI
# ---------------------

st.title("🎬 Advanced Hybrid Movie Recommender")

st.write(
    "This system uses **Hybrid Filtering** combining semantic similarity (on description, genres, cast, director) "
    "with explicit scoring for primary genre match and year proximity."
)

st.header("1. Settings")

with st.sidebar:
    st.header("Tune the Hybrid Weights")
    st.write("Adjust how much influence each factor has on the final score. The total should ideally sum to 1.0.")
    
    # Tunable Weights
    semantic_weight = st.slider("Semantic Similarity Weight", 0.0, 1.0, 0.70, 0.05, key='w_sem')
    genre_weight = st.slider("Primary Genre Match Weight", 0.0, 1.0, 0.25, 0.05, key='w_genre')
    year_weight = st.slider("Year Proximity Weight", 0.0, 1.0, 0.05, 0.05, key='w_year')
    
    total_w = semantic_weight + genre_weight + year_weight
    st.info(f"Total Weight Sum: **{total_w:.2f}**")
    
    weights_dict = {
        'semantic': semantic_weight / total_w, 
        'genre': genre_weight / total_w, 
        'year': year_weight / total_w
    }
    
    st.write("---")
    st.header("Recommendation Parameters")
    num_recs = st.slider("Number of recommendations", 1, 15, 7)
    
# --- Main UI ---

st.header("2. Input")

# Using a selectbox for better user experience
movie_list = sorted(movies['title'].tolist())
user_input = st.selectbox(
    "Select a movie you enjoyed:",
    options=[''] + movie_list,
    index=0 # Starts with an empty selection
)

if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please select a movie title to get started.")
        st.stop()
    
    # Run the recommender
    result_df, scores, query_title = recommend(user_input, num_recs, weights_dict)

    if isinstance(result_df, pd.DataFrame):
        
        # Display key info about the query movie
        key = query_title.strip().lower()
        idx = title_to_index[key]
        st.info(
            f"Query Movie: **{query_title}** ({movies.loc[idx, 'release_year']}). "
            f"Primary Genre: **{movies.loc[idx, 'primary_genre']}**."
        )

        st.success(f"Because you liked **{query_title}**, we recommend:")
        
        # Display results with detailed scores (from Section 3.1: Detailed Score Feedback)
        for rank, ((_, row), (_, total_score, sim, genre_sc, year_sc)) in enumerate(zip(result_df.iterrows(), scores)):
            
            st.markdown(f"## {rank+1}. {row['title']} ({row['release_year']})")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Rating:** {row['rating']}")
                st.write(f"**Director:** {row['director'].split(',')[0] if row['director'] else 'N/A'}")
                st.write(f"**Cast:** {row['cast'].split(',')[0]}...")
                st.write(f"**Primary Genre:** {row['primary_genre']}")
                
            with col2:
                st.markdown(f"**Score Breakdown (Total: {total_score:.3f}):**")
                
                # Show weighted scores based on the sidebar settings
                weighted_sim = sim * weights_dict['semantic']
                weighted_genre = genre_sc * weights_dict['genre']
                weighted_year = year_sc * weights_dict['year']

                st.write(
                    f"* Semantic Match: **{sim:.3f}** (Contributed **{weighted_sim:.3f}**)"
                )
                st.write(
                    f"* Genre Match: **{genre_sc:.2f}** (Contributed **{weighted_genre:.3f}**)"
                )
                st.write(
                    f"* Year Proximity: **{year_sc:.2f}** (Contributed **{weighted_year:.3f}**)"
                )
                
            st.markdown(f"_{row['description']}_")
            st.markdown("---")

    else:
        st.error(scores) # Display the error message
