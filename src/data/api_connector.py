import os
import json
import requests
import aiohttp
import asyncio
from dotenv import load_dotenv
from datetime import datetime, timedelta

# * important
# ? question
# ! attention
# TODO: Implement api here

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get environment variables
API_TOKEN = os.getenv('BEARER_TOKEN')
URL = "https://api.themoviedb.org/3/discover/movie"
HEADERS = {
    "accept": "application/json",
    "Authorization": API_TOKEN
}

def jprint(obj):
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


async def fetch_page(session, page_count: int)-> dict:
    """
    Set start and end date for the movie data fetch.

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
    start_date = end_date - timedelta(days=20*365)
    
    params = {
        "language": "fr-FR",
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
        "without_genres": "Documentary"
    }
    # **important** start async session
    try:
        async with session.get(URL, params=params, headers=HEADERS) as response:
            if response.status != 200:
                print(f"Error: {response.status}")
                return {}
            return await response.json()
            
    except Exception as e:
        print(f"Error: {e}")
        return {}


async def get_movie_ids() -> set:
    
    async with aiohttp.ClientSession() as session:
        answer = await fetch_page(session, 1)
        total_pages = answer.get("total_pages", 1)
        tasks = [fetch_page(session, page_count) for page_count in range(2, total_pages + 1)]
        # **important** An asterisk * denotes iterable unpacking. Its operand must be an iterable. The iterable is expanded into a sequence of items, which are included in the new tuple, list, or set, at the site of the unpacking.
        results = await asyncio.gather(*tasks)
        liste_movies_id = set()
        for movies in [answer] + results:
            for movie in movies.get("results", []):
                liste_movies_id.add(movie["id"])
            
    print(liste_movies_id)
    return list(liste_movies_id)



async def main():
    movie_ids = await get_movie_ids()
    print(movie_ids)

asyncio.run(main())