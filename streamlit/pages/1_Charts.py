import streamlit as st
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk

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


def load_data():
    data = pd.read_parquet("data/movies.parquet")
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text("Loading data...")
# Load 10,000 rows of data into the dataframe.
df = load_data()
movies2 = df[["title", "genres", "actors", "directors", "keywords", 'imbd_score', 'vote_average', 'vote_count']]
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data...done!")
movies2["genres"] = (
    movies2["genres"].str.strip("[]").str.replace(" ", "").str.replace("'", "")
)
movies2["genres"] = movies2["genres"].str.split(",")
movies2['actors'] = movies2['actors'].str.strip('[]').str.replace("'",'').str.replace('"','')
movies2['actors'] = movies2['actors'].str.split(',')

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
        ax.text(0.8, i, v, fontsize=12, color="white", weight="bold")
    plt.title("Top Genres")

    st.pyplot(plt)


with col2:
    st.subheader("Actors with highest appearance")
    plt.subplots(figsize=(14,12))
    list1=[]
    for i in movies2['actors']:
        list1.extend(i)
    ax=pd.Series(list1).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('muted',40))
    for i, v in enumerate(pd.Series(list1).value_counts()[:15].sort_values(ascending=True).values): 
        ax.text(.8, i, v,fontsize=10,color='white',weight='bold')
    plt.title('Actors with highest appearance')
    st.pyplot(plt)