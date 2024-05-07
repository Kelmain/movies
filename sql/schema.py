"""
This module handles the creation, modification, and deletion of tables in a DuckDB database using environment variables for configuration.
"""
import os
import duckdb
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

TABLE_NAME = os.getenv("TABLE_NAME")
DB_NAME = os.getenv("DB_NAME")
SCHEMA = "backdrop_path VARCHAR(255),id INT PRIMARY KEY,imdb_id VARCHAR(255),original_title VARCHAR(255),overview TEXT,popularity DECIMAL(10,2),poster_path VARCHAR(255),release_date DATE,runtime INT,status VARCHAR(255),tagline VARCHAR(255),title VARCHAR(255),vote_average DECIMAL(3,2),vote_count INT,genres VARCHAR(255),actors VARCHAR(255),actors_id VARCHAR(255),directors VARCHAR(255), directors_id VARCHAR(255),video_name VARCHAR(255),video_key VARCHAR(255),keywords VARCHAR(255),production_company VARCHAR(255)"



def create_db(db_name: str) -> None:
    """
    Connect to the DuckDB database at the specified path and create it if it doesn't exist.

    :param db_name: The name of the database to be created or connected to.
    :type db_name: str
    :return: None
    """
    con = duckdb.connect(db_name)
    con.close()


def create_table(db_name: str, table_name: str, schema: str) -> None:
    """
    Connect to the DuckDB database at the specified path and create a table with the given schema.

    :param db_name: The name of the database to be connected to.
    :type db_name: str
    :param table_name: The name of the table to be created.
    :type table_name: str
    :param schema: The schema of the table to be created.
    :type schema: str
    :return: None
    """
    con = duckdb.connect(db_name)
    # create a table and load data into it
    con.sql(f"CREATE TABLE {table_name} ({schema})")
    con.close()


def drop_table(db_name: str, table_name: str) -> None:
    """
    Connect to the DuckDB database at the specified path and drop the table with the given name.

    :param db_name: The name of the database to be connected to.
    :type db_name: str
    :param table_name: The name of the table to be dropped.
    :type table_name: str
    :return: None
    """
    con = duckdb.connect(db_name)
    # drop the table
    con.sql(f"DROP TABLE {table_name}")
    con.close()


if __name__ == "__main__":
    #create_db(DB_NAME)
    create_table(DB_NAME, TABLE_NAME, SCHEMA)
    #drop_table(DB_NAME, TABLE_NAME)
