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
    layout="centered", page_title="Movies Recommandation", page_icon=":film_projector:"
)

st.markdown(
    "<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True
)


def render_title():
    """
    Render the title of the page.
    """
    st.title(":film_projector: Movie Recommendation")


@st.cache_data
def load_data():
    """
    Load the movies data and the cosine matrix.
    """
    data = pd.read_parquet("data/movies.parquet")
    cosine = create_cosinus_df(data)
    cosine_matrix = cosine_similarity(cosine, cosine)
    return data, cosine_matrix


# Create a text element and let the reader know the data is loading.
data_load_state = st.text("Loading data...")

# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data...done!")

def create_movie_dict(df)-> dict:
    """
    Create a dictionary with the movie title as the key and the movie id as the value.
    """
    return df.set_index("title")["id"].to_dict()


def search_movies(search_term, movies_dict):
    """
    Search for movies that contain the given search term and limit the results to 5.
    Returns a list of movie titles.
    """
    try:
        filtered_movies = [title for title in movies_dict if search_term.lower() in title.lower()]
        return filtered_movies[:5]
    except Exception as e:
        print(e)
        return []


def display_movie_details(selected_movie):
    """
    Display the details of the selected movie.
    """
    st.markdown(
        f"<h2 style='text-align: center; color: white;'>{selected_movie['title']}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<img src='https://image.tmdb.org/t/p/original/{selected_movie['backdrop_path']}' alt='Film Poster' width='350' height='auto' style='display:block; margin-left:auto; margin-right:auto; padding-bottom:25px'>",
        unsafe_allow_html=True,
    )
    st.markdown(selected_movie["overview"])

def handle_movie_selection(selected_movie_title, movies_dict, df):
    """
    Handle the selection of a movie and display its details or an error message.
    """
    if selected_movie_title:
        selected_movie_id = movies_dict.get(selected_movie_title)
        if selected_movie_id:
            selected_movie = df[df["id"] == selected_movie_id].iloc[0]
            if not selected_movie.empty:
                display_movie_details(selected_movie)
            else:
                st.subheader("No movie selected or movie not found")
        else:
            st.subheader("No movie selected or movie not found")
    else:
        st.subheader("No movie selected or movie not found")

def display_recommendations(selected_movie_title, df, selected_movie_id, cosine_matrix):
    """
    Display recommendations based on the selected movie.
    """
    if selected_movie_title:
        st.write(get_recommendations(df, selected_movie_id, cosine_matrix))


def main():
    df, cosine_matrix = load_data()
    render_title()
    movies_dict = create_movie_dict(df)

    with st.container():
        selected_movie_title = st_searchbox(
            label="Select movies",
            search_function=search_movies,
            key="search",
            placeholder="Search for a movie",
        )

    handle_movie_selection(selected_movie_title, movies_dict, df)

    with st.container():
        display_recommendations(selected_movie_title, df, movies_dict[selected_movie_title], cosine_matrix)

if __name__ == "__main__":
    main()
