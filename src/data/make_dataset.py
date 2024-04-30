import duckdb


TABLE_NAME = "movies"
DB_NAME = "movies.db

# ['backdrop_path', 'genres[list of names]', 'id','imdb_id', 'original_title', 'overview','popularity',
#  'poster_path', 'release_date', 'runtime', 'status', 'tagline', 'title','vote_average', 'vote_count',
#  'cast[list of names based on credits{cast[name]}] get actor by known_for_department',cast_id[],
# directors[list of names based on credits{crew[name]}] get director by known_for_department', direction_id[],
# 'video_name', 'video_key', 'keywords'[list of keywords[keywords]]]
# "backdrop_path VARCHAR(255),id INT PRIMARY KEY,imdb_id VARCHAR(255),original_title VARCHAR(255),overview TEXT,popularity DECIMAL(5,2),
# poster_path VARCHAR(255),release_date DATE,runtime INT,status VARCHAR(255),tagline VARCHAR(255),title VARCHAR(255),vote_average DECIMAL(3,2),
# vote_count INT,genres VARCHAR(255),actors VARCHAR(255),actors_id INT,directors VARCHAR(255), directors_id INT,video_name VARCHAR(255),
# video_key VARCHAR(255),keywords VARCHAR(255)"
def insert_data(
    data: list, db_name: str = DB_NAME, table_name: str = TABLE_NAME
) -> None:
    con = duckdb.connect(db_name)
    cur = con.cursor()
    try:
        for item in data:
            genre_name = [genre.get("name") for genre in item.get("genres", [])]
            actors = [
                actor.get("name")
                for actor in item.get("credits", {}).get("cast", [])
                if actor.get("known_for_department") == "Acting"
            ]
            actors_id = [
                actor.get("id")
                for actor in item.get("credits", {}).get("cast", [])
                if actor.get("known_for_department") == "Acting"
            ]
            directors = [
                director.get("name")
                for director in item.get("credits", {}).get("crew", [])
                if director.get("known_for_department") == "Directing"
            ]
            directors_id = [
                director.get("id")
                for director in item.get("credits", {}).get("crew", [])
                if director.get("known_for_department") == "Directing"
            ]
            keywords = [keyword.get("keywords") for keyword in item.get("keywords", [])]

            values = (
                item.get("backdrop_path"),
                item.get("id"),
                item.get("imdb_id"),
                item.get("original_title"),
                item.get("overview"),
                item.get("popularity"),
                item.get("poster_path"),
                item.get("release_date"),
                item.get("runtime"),
                item.get("status"),
                item.get("tagline"),
                item.get("title"),
                item.get("vote_average"),
                item.get("vote_count"),
                genre_name,
                actors,
                actors_id,
                directors,
                directors_id,
                item.get("video_name"),
                item.get("video_key"),
                keywords,
            )

            cur.execute(
                f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )

    except Exception as e:
        print(f"Error: {e}")

    con.commit()
    cur.close()
    con.close()
