import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import cache
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.data.path.append(
    r"c:\Users\Work\Desktop\projects\project2\project2-movie-env\Lib\site-packages\nltk"
)
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')
stop_words = set(stopwords.words("french"))
stop_words.update(",", ";", "!", "?", ".", "(", ")", "$", "#", "+", ":", "...", " ", "")


def get_value_from_df_column(df: pd.DataFrame, column_name: str) -> str:
    value = df[column_name].iloc[0]
    return value


def weighted_rating(df: pd.DataFrame, m: int, c: float) -> float:

    v = df["vote_count"]
    r = df["vote_average"]

    return (v / (v + m)) * r + (m / (v + m)) * c


def add_imdb_score(df: pd.DataFrame) -> pd.DataFrame:
    m = 150
    c = df["vote_average"].mean()
    df["imdb_score"] = df.apply(lambda x: weighted_rating(x, m, c), axis=1)
    return df


def create_description_column(df: pd.DataFrame) -> pd.DataFrame:
    df["tagline"] = df["tagline"].fillna("")
    df["description"] = df["overview"] + " " + df["tagline"]
    return df


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""


def create_mix(x):
    return (
        " ".join(x["keywords"])
        + " "
        + " ".join(x["actors"])
        + " "
        + x["directors"]
        + " "
        + " ".join(x["genres"])
        + " "
        + " ".join(x["production_company"])
    )


def create_mixed_column(df: pd.DataFrame) -> pd.DataFrame:
    features = ["actors", "keywords", "directors", "genres", "production_company"]

    for feature in features:
        df[feature] = df[feature].apply(clean_data)
    df["mixed"] = df.apply(create_mix, axis=1)
    return df


def read_db() -> pd.DataFrame:
    con = duckdb.connect("movies.db")
    df = con.table("movies").df()
    con.close()
    return df


def creatre_df() -> pd.DataFrame:
    # print("Current working directory:", os.getcwd())  # This will show the directory from which the script is run

    df = read_db()
    df = create_mixed_column(df)
    df = add_imdb_score(df)
    df = create_description_column(df)

    # Save the DataFrame to a Parquet file
    df.to_parquet("streamlit/data/movies.parquet")
    return df


def create_tifidf(df):
    print("Creating TF-IDF matrix")
    tfidf = TfidfVectorizer(stop_words=list(stop_words), lowercase=True)
    tfidf_matrix = tfidf.fit_transform(df["description"])
    tifidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out()
    )
    print("TF-IDF matrix created")
    return tifidf_df


def create_count(df):
    print("Creating count matrix")
    count_vectorizer = CountVectorizer(stop_words=list(stop_words), lowercase=True)
    count_matrix = count_vectorizer.fit_transform(df["mixed"])
    count_df = pd.DataFrame(
        count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out()
    )
    print("Count matrix created")
    return count_df


def create_standard_scaler(df):
    print("Creating standard scaler")
    scaler = StandardScaler()
    numerical_columns = ["imdb_score", "vote_average"]
    numerical_features = scaler.fit_transform(df[numerical_columns])
    scaled_df = pd.DataFrame(numerical_features, columns=numerical_columns)
    print("Standard scaler created")
    return scaled_df




import pandas as pd
import os

def create_cosinus_df(df):
   
    try:
        
        tfidf_df = create_tifidf(df)
        count_df = create_count(df)
        scaled_df = create_standard_scaler(df)
        if tfidf_df.shape[0] == count_df.shape[0] == scaled_df.shape[0]:
            final_features = pd.concat([tfidf_df, count_df, scaled_df], axis=1)
            cosine_matrix = cosine_similarity(final_features, final_features)
            return cosine_matrix
        else:
            print("Error: DataFrames do not have the same number of rows")
    except Exception as e:
        print(f"An error occurred: {e}")





def get_recommendations(
    df: pd.DataFrame, id: int, cosine_sim: np.ndarray) -> list:
    """
    Get recommendations based on a movie ID and a precomputed cosine similarity matrix.

    Args:
    df (pd.DataFrame): DataFrame containing movie data.
    id (int): The ID of the movie for which recommendations are sought.
    cosine_sim (np.ndarray): Precomputed cosine similarity matrix.

    Returns:
    list: A list of dictionaries, each containing all the information of the top 10 most similar movies.
    """
    # Get the index of the movie that matches the ID
    idx = df.index[df["id"] == id][0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]  # Exclude the first item since it's the movie itself

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the list of dictionaries for the top 10 most similar movies
    return df.iloc[movie_indices].to_dict(orient="records")


if __name__ == "__main__":
    # creatre_df()
    create_cosinus_df()
