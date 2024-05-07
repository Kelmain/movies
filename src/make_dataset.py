import os
import duckdb
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

TABLE_NAME = os.getenv("TABLE_NAME")
DB_NAME = os.getenv("DB_NAME")


class MovieData:
    """
    Class for handling the transformation of movie data items into structured format suitable for database insertion.
    """
    def __init__(self, item):
        self.backdrop_path = item.get("backdrop_path")
        self.id = item.get("id")
        self.imdb_id = item.get("imdb_id")
        self.original_title = item.get("original_title")
        self.overview = item.get("overview")
        self.popularity = round(item.get("popularity", 0), 2)
        self.poster_path = item.get("poster_path")
        self.release_date = item.get("release_date")
        self.runtime = item.get("runtime")
        self.status = item.get("status")
        self.tagline = item.get("tagline")
        self.title = item.get("title")
        self.vote_average = item.get("vote_average")
        self.vote_count = item.get("vote_count")
        self.genre_name = [genre.get("name") for genre in item.get("genres", ['Unknown'])]
        self.actors, self.actors_id = self.extract_people(item, "cast", "Acting", 2)
        self.directors, self.directors_id = self.extract_directors(item, "crew", "Directing")
        self.video_name, self.video_key = self.extract_videos(item)
        self.keywords = [keyword.get("name") for keyword in item.get("keywords", {}).get("keywords", ['Unknown'])[:3]]
        self.production_company = [company.get("name") for company in item.get("production_companies", ['Unknown'][:1])]

    @staticmethod
    def extract_people(item, role_type:str, department:str, max_people:int = None)->tuple:
        """
        Extract people from the item.
        """
        people = [(person.get("name"), person.get("id")) for person in item.get("credits", {}).get(role_type, ['Unknown'])
                  if person.get("known_for_department") == department and (max_people is None or person.get('order') <= max_people)]
        names, ids = zip(*people) if people else (['Unknown'], ['Unknown'])
        return names, ids
    
    @staticmethod
    def extract_directors(item, role_type:str, department:str,job: str = "Director")->tuple:
        """
        Extract directors from the item.
        """
        people = [(person.get("name"), person.get("id")) for person in item.get("credits", {}).get(role_type, ['Unknown'])
                  if person.get("known_for_department") == department and person.get('job') ==job]
        names, ids = zip(*people) if people else (['Unknown'], ['Unknown'])
        return names, ids

    @staticmethod
    def extract_videos(item)->tuple:
        """
        Extract videos from the item.
        """
        videos = item.get("videos", {}).get("results", ['Unknown'])
        names = [video.get("name") for video in videos]
        keys = [video.get("key") for video in videos]
        return names, keys

def insert_data(data: list, db_name: str = DB_NAME, table_name: str = TABLE_NAME) -> None:
    """
    Inserts a list of movie data into the specified database and table.

    Args:
    data (list): A list of dictionaries, each representing movie data.
    db_name (str): The name of the database to connect to.
    table_name (str): The name of the table where data will be inserted.
    """
    con = duckdb.connect(db_name)
    cur = con.cursor()
    try:
        for item in data:
            if item.get("id") is None:
                print("Skipping item due to missing ID:", item)
                continue  # Skip this item if the ID is missing

            movie_data = MovieData(item)
            values = (
                movie_data.backdrop_path, movie_data.id, movie_data.imdb_id, movie_data.original_title,
                movie_data.overview, movie_data.popularity, movie_data.poster_path, movie_data.release_date,
                movie_data.runtime, movie_data.status, movie_data.tagline, movie_data.title, movie_data.vote_average,
                movie_data.vote_count, movie_data.genre_name, movie_data.actors, movie_data.actors_id,
                movie_data.directors, movie_data.directors_id, movie_data.video_name, movie_data.video_key,
                movie_data.keywords,movie_data.production_company,
            )
            cur.execute(
                f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
    except duckdb.Error as db_err:
        print(f"Database error: {db_err}")
    except ValueError as val_err:
        print(f"Value error: {val_err}")
    con.commit()
    cur.close()
    con.close()

if __name__ == "__main__":
    test_data()




