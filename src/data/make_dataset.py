import duckdb


TABLE_NAME = "movies"
DB_NAME = "movies.db"

# ['backdrop_path', 'genres[list of names]', 'id','imdb_id', 'original_title', 'overview','popularity',
#  'poster_path', 'release_date', 'runtime', 'status', 'tagline', 'title','vote_average', 'vote_count',
#  'cast[list of names based on credits{cast[name]}] get actor by known_for_department',cast_id[],
# directors[list of names based on credits{crew[name]}] get director by known_for_department', direction_id[],
# 'video_name', 'video_key', 'keywords'[list of keywords[keywords]]]
# "backdrop_path VARCHAR(255),id INT PRIMARY KEY,imdb_id VARCHAR(255),original_title VARCHAR(255),overview TEXT,popularity DECIMAL(5,2),
# poster_path VARCHAR(255),release_date DATE,runtime INT,status VARCHAR(255),tagline VARCHAR(255),title VARCHAR(255),vote_average DECIMAL(3,2),
# vote_count INT,genres VARCHAR(255),actors VARCHAR(255),actors_id INT,directors VARCHAR(255), directors_id INT,video_name VARCHAR(255),
# video_key VARCHAR(255),keywords VARCHAR(255)"


def test_data()-> None:
    with duckdb.connect(DB_NAME) as con:
        con.table(TABLE_NAME).show()



class MovieData:
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
        self.genre_name = [genre.get("name") for genre in item.get("genres", [])]
        self.actors, self.actors_id = self.extract_people(item, "cast", "Acting", 5)
        self.directors, self.directors_id = self.extract_people(item, "crew", "Directing")
        self.video_name, self.video_key = self.extract_videos(item)
        self.keywords = [keyword.get("name") for keyword in item.get("keywords", {}).get("keywords", [])]

    @staticmethod
    def extract_people(item, role_type, department, max_people=None):
        people = [(person.get("name"), person.get("id")) for person in item.get("credits", {}).get(role_type, [])
                  if person.get("known_for_department") == department and (max_people is None or person.get('order') <= max_people)]
        names, ids = zip(*people) if people else ([], [])
        return names, ids

    @staticmethod
    def extract_videos(item):
        videos = item.get("videos", {}).get("results", [])
        names = [video.get("name") for video in videos]
        keys = [video.get("key") for video in videos]
        return names, keys

def insert_data(data: list, db_name: str = DB_NAME, table_name: str = TABLE_NAME) -> None:
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
                movie_data.keywords,
            )
            cur.execute(
                f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
    except Exception as e:
        print(f"insert db: {e}")
    con.commit()
    cur.close()
    con.close()

if __name__ == "__main__":
    test_data()




