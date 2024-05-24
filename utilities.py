"""
This module provides utility functions for data processing and machine learning tasks in a Streamlit application.
It includes functions for database operations, data cleaning, feature engineering, and model pipeline creation.
"""

import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import skops.io as sio
from nltk.stem import SnowballStemmer
from dotenv import load_dotenv
from functools import cache


# Load the .env file
load_dotenv()
nltk.data.path.append(os.getenv("NLTK_PATH"))
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
stop_words = set(stopwords.words("french"))
stop_words.update(",", ";", "!", "?", ".", "(", ")", "$", "#", "+", ":", "...", "Unknown", "Missing")

TABLE_NAME = os.getenv("TABLE_NAME")
DB_NAME = os.getenv("DB_NAME")

@cache
def get_movies_db()-> pd.DataFrame:
    """
    Get the movies data from the database.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the movies data.
    """
    con = duckdb.connect(DB_NAME)
    try:
        movies = con.table(TABLE_NAME).df()
        return movies
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()

def add_imdb_score_db(table_name: str):
    """
    Add an IMDb score column to the database table based on the weighted rating formula.

    Args:
    table_name (str): The name of the table in the database to update.
    """
    con = duckdb.connect(DB_NAME)
    try:
        # Calculate the mean vote average across the dataset
        c = con.execute(f"SELECT AVG(vote_average) FROM {table_name}").fetchone()[0]
        m = 150  # The minimum number of votes required to be considered

        # Add the IMDb score column
        con.execute(
            f"""
            ALTER TABLE {table_name} ADD COLUMN imdb_score DOUBLE;
        """
        )
        con.execute(
            f"""
            UPDATE {table_name}
            SET imdb_score = (vote_count / (vote_count + {m})) * vote_average + ({m} / (vote_count + {m})) * {c}
        """
        )

    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()


def create_mixed_column_db(table_name: str):
    """
    Create a mixed column in the database table that includes the actors, keywords, directors, genres, and production company.

    Args:
    table_name (str): The name of the table in the database to update.
    """
    con = duckdb.connect(DB_NAME)
    try:
        # Add a new 'mixed' column
        con.execute(
            f"""
            ALTER TABLE {table_name} ADD COLUMN mixed VARCHAR;
            """
        )

        # Populate the 'mixed' column with cleaned and combined data
        con.execute(
            f"""
            UPDATE {table_name}
            SET mixed = LOWER(REPLACE(keywords, ' ', '') || ' ' || 
                             REPLACE(actors, ' ', '') || ' ' || 
                             REPLACE(directors, ' ', '') || ' ' || 
                             REPLACE(genres, ' ', '') || ' ' || 
                             REPLACE(production_company, ' ', ''))
            """
        )

    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()


def create_description_column_db(table_name: str):
    """
    Update the database table to include a 'description' column by combining 'tagline' and 'overview',
    and applying text processing using SQL in duckdb.

    Args:
    table_name (str): The name of the table in the database to update.

    """
    con = duckdb.connect(DB_NAME)
    try:
        # Replace empty taglines and overviews with 'Missing'
        con.execute(
            f"""
            UPDATE {table_name}
            SET tagline = 'Missing'
            WHERE tagline = '';
        """
        )
        con.execute(
            f"""
            UPDATE {table_name}
            SET overview = 'Missing'
            WHERE overview = '';
        """
        )

        # Add a new column 'description' that combines 'overview' and 'tagline'
        con.execute(
            f"""
            ALTER TABLE {table_name} ADD COLUMN description VARCHAR;
        """
        )
        con.execute(
            f"""
            UPDATE {table_name}
            SET description = LOWER(overview || ' ' || tagline);
        """
        )

    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()



def stem_text(text):
    """
    Stem the input text using NLTK's SnowballStemmer.

    Args:
    text (str): The text to stem.

    Returns:
    str: The stemmed text.
    """
    if text is None:
        return None
    stemmer = SnowballStemmer("french")  # Assuming the text is in English
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    return " ".join(stemmed_words).lower()

def update_text_columns_with_stemming(table_name: str):
    """
    Update the 'keywords' and 'description' columns in the database table using the stem_text UDF.

    Args:
    table_name (str): The name of the table to update.
    """
    con = duckdb.connect(DB_NAME)
    try:
        # Update the 'keywords' and 'description' columns using the stem_text UDF
        con.execute(f"""
            UPDATE {table_name}
            SET keywords = stem_text(keywords);
        """)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()

def add_ml_columns_db():
    """
    Add the 'description' and 'mixed' columns to the database table using the lemmatize_text UDF.
    """
    # Connect to DuckDB
    con = duckdb.connect(DB_NAME)

    # Register the UDF
    con.create_function(
        "stem_text", stem_text, parameters=["varchar"], return_type="varchar"
    )
    # create description column
    create_description_column_db(TABLE_NAME)
    update_text_columns_with_stemming(TABLE_NAME)
    # create mixed column
    create_mixed_column_db(TABLE_NAME)
    # create imdb score column
    add_imdb_score_db(TABLE_NAME)

    show = con.sql(f"SELECT * FROM {TABLE_NAME}")
    print(show)
    con.close()


def create_preprocessor() -> ColumnTransformer:
    """
    Create a preprocessor that includes TF-IDF and count vectorization.

    Returns:
    ColumnTransformer: A scikit-learn ColumnTransformer object with TF-IDF and count vectorization.
    """
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


def create_pipeline_knn(df: pd.DataFrame, params: dict) -> Pipeline:
    """
    Create a pipeline that includes preprocessing and a KNN model.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    params (dict): Parameters for the KNN model, including 'metric'.

    Returns:
    Pipeline: A scikit-learn Pipeline object with preprocessing and KNN model.
    """
    # Define the KNN model
    knn_model = NearestNeighbors(
        n_neighbors=11, algorithm="auto", metric=params["metric"]
    )
    preprocessor = create_preprocessor()
    # Combine preprocessing and KNN model using Pipeline
    pipeline_with_knn = Pipeline(
        steps=[("preprocessor", preprocessor), ("knn", knn_model)]
    )
    pipeline_with_knn.fit(df)
    sio.dump(pipeline_with_knn, f"pipeline_{params["metric"]}.skops")
    return pipeline_with_knn


def find_nearest_neighbors(movie_id: int, movies: pd.DataFrame, params: dict) -> list:
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
    query_product_features = df[df["id"] == movie_id].drop(columns=["id"])
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
        nearest_neighbors_df["id"] != movie_id
    ].to_dict(orient="records")

    return nearest_neighbors_list


def create_cosinus_matrix(df: pd.DataFrame, params: dict) -> np.ndarray:
    """
    Create a cosine similarity matrix.

    Args:
    df (pd.DataFrame): DataFrame containing movie data.
    params (dict): Parameters for the KNN model, including 'metric'.

    Returns:
    np.ndarray: A numpy array containing the cosine similarity matrix.
    """
    pipeline_with_knn = create_pipeline_knn(df, params)
    similarity_matrix = cosine_similarity(
        pipeline_with_knn.named_steps["preprocessor"].transform(df)
    )
    return similarity_matrix


def get_recommendations(df: pd.DataFrame, movie_id: int, params: dict) -> list:
    """
    Get recommendations based on a movie ID and a precomputed cosine similarity matrix.
    """
    # Get the index of the movie that matches the movie_id
    idx = df.index[df["id"] == movie_id][0]
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
    add_ml_columns_db()
