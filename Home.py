import os
import warnings
import duckdb
import pandas as pd
from utilities import (
    get_recommendations,
    find_nearest_neighbors,
    get_movies_db,
)
from streamlit_searchbox import st_searchbox
from streamlit_carousel import carousel
from dotenv import load_dotenv
import streamlit as st


# Load the .env file
load_dotenv()

TABLE_NAME = os.getenv("TABLE_NAME")
DB_NAME = os.getenv("DB_NAME")

warnings.filterwarnings("ignore")

# Page setting
st.set_page_config(
    layout="wide", page_title="Movies Recommandation", page_icon=":film_projector:"
)

st.markdown(
    "<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True
)


st.title(":film_projector: Movie Recommendation")

classifier_name = st.sidebar.selectbox("Select classifier", ("KNN", "Cosinus"), index=1)

# Create a text element and let the reader know the data is loading.
data_load_state = st.text("Loading data...")
df = get_movies_db()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data...done!")

movies_dict = df.set_index("title")["id"].to_dict()


def create_carousel(recom_dict: list) -> list:
    """
    Create a carousel of movies
    """
    carousel_liste = []
    for movie in recom_dict:
        carousel_liste.append(
            {
                "title": movie["title"],
                "text": movie["tagline"],
                "img": f"https://image.tmdb.org/t/p/original/{movie['backdrop_path']}",
                "link": f"https://www.imdb.com/title/{movie['imdb_id']}",
            }
        )

    return carousel_liste


def search_movies(search_term, movies_dict):
    """
    Search for movies that contain the given search term and limit the results to 5.
    Returns a list of movie titles.
    """
    try:
        filtered_movies = [title for title in movies_dict if search_term in title]
        return filtered_movies[:5]
    except Exception as e:
        print(e)
        return []


with st.container():
    coll, colm, colr = st.columns((1, 2, 1))
    with coll:
        """"""
    with colm:
        selected_movie_title = st_searchbox(
            label="Select movies",
            search_function=lambda search_term: search_movies(search_term, movies_dict),
            key="search",
            placeholder="Search for a movie",
            default="Avatar",
            clear_on_submit=True,
        )
    with colr:
        """"""
if selected_movie_title:

    selected_movie_id = movies_dict.get(selected_movie_title)
    if selected_movie_id:
        selected_movie = df[df["id"] == selected_movie_id].iloc[0]
        if not selected_movie.empty:

            st.markdown(
                f"<h2 style='text-align: center; color: white;'>{selected_movie['title']}</h2>",
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                col1, col2 = st.columns((1, 3), gap="large")

                with col1:
                    st.markdown(
                        f"<img src='https://image.tmdb.org/t/p/original/{selected_movie['backdrop_path']}' alt='Film Poster' width='350' height='auto' style='display:block; margin-left:auto; margin-right:auto; padding-bottom:25px'>",
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(selected_movie["overview"])
                    st.text(
                        f"Rating: {selected_movie['vote_average']}    Imdb-Score: {round(selected_movie['imdb_score'],2)}"
                    )

                    with st.popover(
                        label=":film_frames: Trailer",
                        help="Play the trailer of the movie",
                    ):
                        video_keys = selected_movie["video_key"].strip("[]").split(", ")
                        if video_keys:
                            st.video(
                                f"https://www.youtube.com/watch?v={video_keys[0].strip()}"
                            )
                        else:
                            st.text("No trailer available.")

        else:
            st.subheader("No movie selected or movie not found")
    else:
        st.subheader("No movie selected or movie not found")
else:
    st.subheader("No movie selected or movie not found")


def add_parameter_ui(classifier_name):
    """
    Add a selectbox to the sidebar to select the metric for the KNN model.

    Args:
    classifier_name (str): The name of the classifier.

    Returns:
    dict: A dictionary with the metric selected by the user.
    """
    params = dict()
    if classifier_name == "KNN":
        metric = st.sidebar.selectbox(
            "Select metric",
            ("minkowski", "cosine", "euclidean", "manhattan", "l1", "l2", "cityblock"),
        )
        params["metric"] = metric
    elif classifier_name == "Cosinus":
        metric = "minkowski"
        params["metric"] = metric
    return params


if selected_movie_title:
    params = add_parameter_ui(classifier_name)
    cosinus_dict = get_recommendations(df, selected_movie_id, params)

    knn_dict = find_nearest_neighbors(selected_movie_id, df, params)
    cosinus_carousel = create_carousel(cosinus_dict)
    knn_carousel = create_carousel(knn_dict)

    # Display the carousel based on the classifier selected
    if classifier_name == "Cosinus":
        st.markdown(
            f"<h2 style='text-align: center; color: white;'>Cosinus Model</h2>",
            unsafe_allow_html=True,
        )
        carousel(items=cosinus_carousel, width=0.5)
    elif classifier_name == "KNN":
        st.markdown(
            f"<h2 style='text-align: center; color: white;'>KNN Model</h2>",
            unsafe_allow_html=True,
        )
        carousel(items=knn_carousel, width=0.5)
