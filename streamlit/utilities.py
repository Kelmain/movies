import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import cache
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
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
    df["tagline"] = df["tagline"].replace("", "Missing")
    df["overview"] = df["overview"].replace("", "Missing")
    df["description"] = df["overview"] + " " + df["tagline"]
    return df


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
        "".join(x["keywords"])
        + " "
        + "".join(x["actors"])
        + " "
        + x["directors"]
        + " "
        + "".join(x["genres"])
        + " "
        + "".join(x["production_company"])
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


def create_tifidf(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating TF-IDF matrix")
    tfidf = TfidfVectorizer(stop_words=list(stop_words), lowercase=True)
    tfidf_matrix = tfidf.fit_transform(df["description"])
    tifidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out()
    )
    print("TF-IDF matrix created")
    return tifidf_df


def create_count(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating count matrix")
    count_vectorizer = CountVectorizer(stop_words=list(stop_words), lowercase=True)
    count_matrix = count_vectorizer.fit_transform(df["mixed"])
    count_df = pd.DataFrame(
        count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out()
    )
    print("Count matrix created")
    return count_df


def create_standard_scaler(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating standard scaler")
    scaler = StandardScaler()
    numerical_columns = ["imdb_score", "vote_average"]
    numerical_features = scaler.fit_transform(df[numerical_columns])
    scaled_df = pd.DataFrame(numerical_features, columns=numerical_columns)
    print("Standard scaler created")
    return scaled_df


def create_preprocessor() -> ColumnTransformer:
    columns_tfidf_to_encode = "description"
    columns_to_countvectorize = "mixed"
    columns_to_std = ["imdb_score", "vote_average"]
    tfidf_transformers = [
        (
            "tfidf",
            TfidfVectorizer(stop_words=list(stop_words), lowercase=True),
            columns_tfidf_to_encode,
        )
    ]
    count_transformers = [
        (
            "count",
            CountVectorizer(stop_words=list(stop_words), lowercase=True),
            columns_to_countvectorize,
        )
    ]
    std_scale_transformers = [("std_scale", StandardScaler(), columns_to_std)]
    all_transformers = count_transformers + std_scale_transformers + tfidf_transformers
    preprocessor = ColumnTransformer(all_transformers)
    return preprocessor


def create_pipeline_knn(df: pd.DataFrame,params: dict) -> Pipeline:
    
    # Define the KNN model
    knn_model = NearestNeighbors(n_neighbors=11, algorithm="auto", metric=params['metric'])
    preprocessor = create_preprocessor()
    # Combine preprocessing and KNN model using Pipeline
    pipeline_with_knn = Pipeline(
        steps=[("preprocessor", preprocessor), ("knn", knn_model)]
    )
    pipeline_with_knn.fit(df)
    return pipeline_with_knn


def find_nearest_neighbors(
    id: int, movies: pd.DataFrame, params: dict
) -> list:
    """
    Find and return the nearest neighbors of a given movie ID using a pre-trained KNN pipeline.

    Args:
    id (int): The ID of the movie for which neighbors are sought.
    pipeline_with_knn (Pipeline): A scikit-learn Pipeline object containing preprocessing steps and a KNN model.
    movies (pd.DataFrame): DataFrame containing movie data.

    Returns:
    list: A list of dictionaries, each containing all the information of the nearest neighbor movies.
    """
    df = movies[["mixed", "imdb_score", "vote_average", "id", "description"]]
    pipeline_with_knn = create_pipeline_knn(movies, params)
    # Filter the DataFrame to get the features of the specified movie
    query_product_features = df[df["id"] == id].drop(columns=["id"])
    # Use the pipeline to preprocess the query movie features
    query_product_features_processed = pipeline_with_knn.named_steps[
        "preprocessor"
    ].transform(query_product_features)

    # Use the KNN model to find the nearest neighbors for the query movie
    nearest_neighbors_indices = pipeline_with_knn.named_steps["knn"].kneighbors(
        query_product_features_processed, return_distance=False 
    )[0]

    # Get the nearest neighbors' Movie IDs
    nearest_neighbors_movie_ids = df.iloc[nearest_neighbors_indices]["id"]

    # Create a DataFrame containing the nearest neighbors' information
    nearest_neighbors_df = movies[movies["id"].isin(nearest_neighbors_movie_ids)]

    # Exclude the query movie itself from the results and convert to a list of dictionaries
    nearest_neighbors_list = nearest_neighbors_df[
        nearest_neighbors_df["id"] != id
    ].to_dict(orient="records")

    return nearest_neighbors_list


def create_cosinus_matrix(df: pd.DataFrame, params: dict) -> np.ndarray:
    pipeline_with_knn = create_pipeline_knn(df, params)
    similarity_matrix = cosine_similarity(
        pipeline_with_knn.named_steps["preprocessor"].transform(df)
    )
    return similarity_matrix


def get_recommendations(df: pd.DataFrame, id: int, params: dict) -> list:
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
    cosine_sim = create_cosinus_matrix(df, params)
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
    creatre_df()
