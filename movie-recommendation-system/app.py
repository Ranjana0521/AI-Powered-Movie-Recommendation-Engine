import streamlit as st
from model.recommender import MovieRecommender
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎬 Hybrid Movie Recommender",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return MovieRecommender("data/movies.csv", "data/ratings.csv")

recommender = load_model()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: rgba(0,0,0,0.25);
}

.movie-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 20px;
    transition: 0.3s;
}

.movie-card:hover {
    transform: scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<h1 style='text-align:center;'>🎬 Hybrid Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Content-Based + Collaborative Filtering</p>", unsafe_allow_html=True)

st.markdown("---")

# =========================
# SIDEBAR (Only Movie Selection)
# =========================
st.sidebar.header("🎥 Select Your Favorite Movie")

selected_movie = st.sidebar.selectbox(
    "Choose a Movie",
    sorted(recommender.movies['title'].values)
)

# =========================
# BUTTON
# =========================
if st.sidebar.button("Get Recommendations"):

    with st.spinner("Finding best movies for you... 🎬"):

        # We now pass dummy user_id = 1 internally
        recommendations = recommender.hybrid_recommend(
            1,  # default user (hidden from UI)
            selected_movie,
            10
        )

    if recommendations:

        st.subheader("✨ Recommended Movies For You")

        cols = st.columns(2)

        for i, movie in enumerate(recommendations):

            movie_info = recommender.movies[
                recommender.movies['title'] == movie
            ].iloc[0]

            ratings = recommender.data[
                recommender.data['title'] == movie
            ]['rating']

            avg_rating = round(ratings.mean(), 1) if len(ratings) > 0 else None

            year = None
            if 'year' in movie_info and pd.notna(movie_info['year']):
                year = int(movie_info['year'])

            meta = ""
            if avg_rating:
                meta += f"⭐ {avg_rating}"
            if year:
                if meta:
                    meta += " | "
                meta += f"📅 {year}"

            with cols[i % 2]:
                st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                st.markdown(f"### {movie}")
                if meta:
                    st.caption(meta)
                st.caption("Based on similar movies & user patterns")
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("No recommendations found")

# =========================
# ABOUT SECTION
# =========================
with st.expander("About This Project"):

    st.write("""
This is a Hybrid Movie Recommendation System combining:

• Collaborative Filtering  
• Content-Based Filtering  

Built using:

• Python  
• Pandas  
• Scikit-learn  
• Streamlit  

Designed as a portfolio-ready Machine Learning project.
""")