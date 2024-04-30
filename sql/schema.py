import duckdb



TABLE_NAME = "movies"
SCHEMA = "backdrop_path VARCHAR(255),id INT PRIMARY KEY,imdb_id VARCHAR(255),original_title VARCHAR(255),overview TEXT,popularity DECIMAL(10,2),poster_path VARCHAR(255),release_date DATE,runtime INT,status VARCHAR(255),tagline VARCHAR(255),title VARCHAR(255),vote_average DECIMAL(3,2),vote_count INT,genres VARCHAR(255),actors VARCHAR(255),actors_id VARCHAR(255),directors VARCHAR(255), directors_id VARCHAR(255),video_name VARCHAR(255),video_key VARCHAR(255),keywords VARCHAR(255)"
DB_NAME = "movies.db"


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

"""
Maybe i will try this later...
CREATE TABLE Movies (
    backdrop_path VARCHAR(255),
    id INT PRIMARY KEY,
    imdb_id VARCHAR(255),
    original_title VARCHAR(255),
    overview TEXT,
    popularity DECIMAL(5,2),
    poster_path VARCHAR(255),
    release_date DATE,
    runtime INT,
    status VARCHAR(255),
    tagline VARCHAR(255),
    title VARCHAR(255),
    vote_average DECIMAL(3,2),
    vote_count INT
);

CREATE TABLE Genres (
    id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT,
    genre_name VARCHAR(255),
    FOREIGN KEY (movie_id) REFERENCES Movies(id)
);

CREATE TABLE Cast (
    id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT,
    actor_name VARCHAR(255),
    known_for_department VARCHAR(255),
    cast_id INT,
    FOREIGN KEY (movie_id) REFERENCES Movies(id)
);

CREATE TABLE Directors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT,
    director_name VARCHAR(255),
    known_for_department VARCHAR(255),
    direction_id INT,
    FOREIGN KEY (movie_id) REFERENCES Movies(id)
);

CREATE TABLE Videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT,
    video_name VARCHAR(255),
    video_key VARCHAR(255),
    FOREIGN KEY (movie_id) REFERENCES Movies(id)
);

CREATE TABLE Keywords (
    id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT,
    keyword VARCHAR(255),
    FOREIGN KEY (movie_id) REFERENCES Movies(id)
);
"""
