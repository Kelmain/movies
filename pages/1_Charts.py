import os
import warnings
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from dotenv import load_dotenv
from utilities import (
    get_movies_db,
)



# Load the .env file
load_dotenv()

TABLE_NAME = os.getenv("TABLE_NAME")
DB_NAME = os.getenv("DB_NAME")

nltk.data.path.append(
    r"c:\Users\Work\Desktop\projects\project2\project2-movie-env\Lib\site-packages\nltk"
)
from nltk.corpus import stopwords

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")
stop_words = set(stopwords.words("french"))
stop_words.update(",", ";", "!", "?", ".", "(", ")", "$", "#", "+", ":", "...", " ", "")
# Page setting
st.set_page_config(layout="wide", page_title="Charts", page_icon=":bar_chart:")
st.markdown(
    "<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True
)

st.title(":bar_chart: Charts")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text("Loading data...")
# Load 10,000 rows of data into the dataframe.
df = get_movies_db()

movies2 = df[
    [
        "title",
        "genres",
        "actors",
        "directors",
        "keywords",
        "imdb_score",
        "vote_average",
        "vote_count",
    ]
]
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data...done!")
movies2["genres"] = (
    movies2["genres"].str.strip("[]").str.replace(" ", "").str.replace("'", "")
)
movies2["genres"] = movies2["genres"].str.split(",")
movies2["actors"] = (
    movies2["actors"].str.strip("[]").str.replace("'", "").str.replace('"', "")
)
movies2["actors"] = movies2["actors"].str.split(",")
movies2["directors"] = (
    movies2["directors"].str.strip("[]").str.replace("'", "").str.replace('"', "")
)
movies2["directors"] = movies2["directors"].str.split(",")

col1, col2 = st.columns((2))
# Define custom color palette

with col1:
    st.subheader("Top genres")
    plt.subplots(figsize=(14, 12))
    list1 = []
    for i in movies2["genres"]:
        list1.extend(i)
    ax = (
        pd.Series(list1)
        .value_counts()[:10]
        .sort_values(ascending=True)
        .plot.barh(width=0.9, color=sns.color_palette("hls", 10))
    )
    for i, v in enumerate(
        pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values
    ):
        ax.text(0.8, i, v, fontsize=12, color="black", weight="bold")
    plt.title("Top Genres")

    st.pyplot(plt)


with col2:
    st.subheader("Actors with highest appearance")
    plt.subplots(figsize=(14, 12))
    list1 = []
    for i in movies2["actors"]:
        list1.extend(i)
    ax = (
        pd.Series(list1)
        .value_counts()[:10]
        .sort_values(ascending=True)
        .plot.barh(width=0.9, color=sns.color_palette("hls", 15))
    )
    for i, v in enumerate(
        pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values
    ):
        ax.text(0.8, i, v, fontsize=12, color="black", weight="bold")
    plt.title("Actors with highest appearance")
    st.pyplot(plt)

col3, col4 = st.columns((2))

with col3:
    st.subheader("Top directors")
    plt.subplots(figsize=(14, 12))
    filtered_movies = movies2[
        (movies2["directors"] != "")
        & (~movies2["directors"].apply(lambda x: "Unknown" in x))
    ]
    ax = (
        filtered_movies.explode("directors")["directors"]
        .value_counts()[:10]
        .sort_values(ascending=True)
        .plot.barh(width=0.9, color=sns.color_palette("hls", 40))
    )
    for i, v in enumerate(
        filtered_movies.explode("directors")["directors"]
        .value_counts()[:10]
        .sort_values(ascending=True)
        .values
    ):
        ax.text(0.5, i, v, fontsize=12, color="black", weight="bold")
    plt.title("Directors with most movies")

    st.pyplot(plt)

with col4:
    st.subheader("Keywords")
    plt.subplots(figsize=(14, 12))
    words = movies2["keywords"].dropna().apply(nltk.word_tokenize)
    word = []
    for i in words:
        word.extend(i)
    word = pd.Series(word)
    word = [i for i in word.str.lower() if i not in stop_words]
    wc = WordCloud(
        background_color="black",
        max_words=1000,
        stopwords=stop_words,
        max_font_size=60,
        width=1000,
        height=1000,
    )
    wc.generate(" ".join(word))
    plt.imshow(wc)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(10, 10)

    st.pyplot(plt)
