import os
import sys
import warnings
import pandas as pd
from utilities import get_recommendations,create_cosinus_df
from streamlit_searchbox import st_searchbox
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st



warnings.filterwarnings("ignore")

# Page setting
st.set_page_config(
    layout="centered", page_title="Movies recommandation", page_icon=":film_projector:"
)

st.markdown(
    "<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True
)
st.title(":film_projector: Movie recommandation")

@st.cache_data
def load_data():
    data = pd.read_parquet("data/movies.parquet")
    cosine = create_cosinus_df(data)
    cosine_matrix = cosine_similarity(cosine, cosine)
    return data, cosine_matrix


# Create a text element and let the reader know the data is loading.
data_load_state = st.text("Loading data...")

df, cosine_matrix = load_data()
#cosine_sim = load_cosinus()

# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data...done!")


movies_dict = df.set_index("title")["id"].to_dict()


def search_movies(search_term):
    """
    Search for movies that contain the given search term and limit the results to 5.
    Returns a list of movie titles.

    Args:
    search_term (str): The term to search for within movie titles.

    Returns:
    list: A list of up to 5 movie titles that contain the search term.
    """
    try:
        # Filter movies and return only titles
        filtered_movies = [title for title in movies_dict if search_term in title]
        return filtered_movies[:5]
    except Exception as e:
        print(e)
        return []


with st.container():
    selected_movie_title = st_searchbox(
        label="Select movies",
        search_function=search_movies,
        key="search",
        placeholder="Search for a movie",
    )

    # Use the title to get the ID from the dictionary, then fetch the movie details
    if selected_movie_title:
        selected_movie_id = movies_dict[selected_movie_title]
        selected_movie = df[df["id"] == selected_movie_id].iloc[0]

        if not selected_movie.empty:
            st.markdown(
                f"<h2 style='text-align: center; color: white;'>{selected_movie['title']}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<img src='https://image.tmdb.org/t/p/original/{selected_movie['backdrop_path']}' alt='Film Poster' width='350' height='auto' style='display:block; margin-left:auto; margin-right:auto; padding-bottom:25px'>",
                unsafe_allow_html=True,
            )
            st.markdown(selected_movie["overview"])
        else:
            st.subheader("No movie selected or movie not found")
    else:
        st.subheader("No movie selected or movie not found")

with st.container():
    if selected_movie_title:
        st.write(get_recommendations(df, selected_movie_id, cosine_matrix))
