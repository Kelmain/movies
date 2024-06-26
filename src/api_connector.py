import os
import sys
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dotenv import load_dotenv



from make_dataset import insert_data

# * important
# ? question
# ! attention
# TODO: Implement api here

# Load the .env file
load_dotenv()

# Get environment variables
API_TOKEN = os.getenv("BEARER_TOKEN")
URL_DISCOVER_MOVIE = os.getenv("URL_DISCOVER_MOVIE")
URL_DETAILS_MOVIE = os.getenv("URL_DETAILS_MOVIE")
HEADERS = {"accept": "application/json", "Authorization": API_TOKEN}
LANGUAGE = os.getenv("LANGUAGE")

async def fetch_page(session, page_count: int) -> dict:
    """
    Fetches movie data for the specified page.

    Parameters:
    - session (aiohttp.ClientSession): An asynchronous HTTP client session.
    - page_count (int): The page number of the movie data to fetch.

    Returns:
    - dict: A dictionary containing the movie data for the specified page.

    Raises:
    - Exception: If an error occurs during the HTTP request.
    """
    # set start and end date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20 * 365)

    params = {
        "language": LANGUAGE,
        "page": page_count,
        "include_adult": "false",
        "include_video": "false",
        "sort_by": "primary_release_date.desc",
        "primary_release_date.gte": start_date.strftime("%Y-%m-%d"),
        "primary_release_date.lte": end_date.strftime("%Y-%m-%d"),
        "vote_average.gte": 5,
        "vote_count.gte": 150,
        "with_runtime.gte": 80,
        "with_runtime.lte": 240,
        "without_genres": "Documentaire",
    }
    # **important** get movies page by page
    try:
        async with session.get(
            URL_DISCOVER_MOVIE, params=params, headers=HEADERS
        ) as response:
            if response.status != 200:
                print(f"Error: {response.status}")
                return {}
            return await response.json()

    except aiohttp.ClientError as e:
        print(f"Error: {e}")
        return {}


async def get_movie_ids(session) -> list:
    """
    Fetches movie IDs for the specified page.

    Parameters:
    - session (aiohttp.ClientSession): An asynchronous HTTP client session.

    Returns:
    - list: A list of movie IDs.
    """
    answer = await fetch_page(session, 1)
    total_pages = answer.get("total_pages", 1)
    #total_pages = 2
    tasks = [
        fetch_page(session, page_count) for page_count in range(2, total_pages + 1)
    ]
    # **important** An asterisk * denotes iterable unpacking. Its operand must be an iterable. The iterable is expanded into a sequence of items, which are included in the new tuple, list, or set, at the site of the unpacking.
    results = await asyncio.gather(*tasks)
    movies_id = set()
    for movies in [answer] + results:
        for movie in movies.get("results", []):
            movies_id.add(movie["id"])

    print(len(movies_id))
    return list(movies_id)


async def get_movie_data(session, movie_id: int) -> dict:
    """
    Fetches detailed movie data including credits, videos, and keywords for a given movie ID.

    Parameters:
    - session (aiohttp.ClientSession): An asynchronous HTTP client session.
    - movie_id (int): The unique identifier for the movie.

    Returns:
    - dict: A dictionary containing detailed movie data.

    Raises:
    - aiohttp.ClientError: If an error occurs during the HTTP request.
    """
    params = {"language": LANGUAGE, "append_to_response": "credits,videos,keywords"}
    url = f"{URL_DETAILS_MOVIE}{movie_id}"

    try:
        async with session.get(url, headers=HEADERS, params=params) as response:
            if response.status == 429 :
                retry_after = int(response.headers.get("Retry-After", 10))
                await asyncio.sleep(retry_after)
                return await get_movie_data(session, movie_id)  # Recursive retry
            response.raise_for_status()  # Will raise an error for 4XX/5XX status codes
            return await response.json()

    except aiohttp.ClientError as e:
        print(f"HTTP client error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return {}


async def main():
    """
    Main function to fetch movie IDs and their detailed data, then insert the data into a dataset.
    """
    async with aiohttp.ClientSession() as session:
        movie_ids = await get_movie_ids(session)
        tasks = [get_movie_data(session, movie_id) for movie_id in movie_ids]
        datas = await asyncio.gather(*tasks)
        insert_data(datas)
    print(len(datas))



asyncio.run(main())