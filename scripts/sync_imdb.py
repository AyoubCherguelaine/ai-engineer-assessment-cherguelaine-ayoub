"""Download and build a compact, searchable IMDb SQLite index.

Example: python -m scripts.sync_imdb --max-movies 250000
Use --all-datasets to also cache principals for future extensions.
"""

import argparse
import csv
import gzip
import heapq
import shutil
import sqlite3
from pathlib import Path
from urllib.request import urlopen

from agents.imdb import BASE_URL, DATASETS, normalize


def download_dataset(key: str, raw_directory: Path) -> Path:
    raw_directory.mkdir(parents=True, exist_ok=True)
    destination = raw_directory / DATASETS[key]
    if destination.exists():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {DATASETS[key]}...")
    with urlopen(f"{BASE_URL}/{DATASETS[key]}", timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)
    return destination


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        yield from csv.DictReader(source, delimiter="\t")


def most_voted_titles(ratings_path: Path, count: int) -> dict[str, tuple[float, int]]:
    """Keep only popular titles so a compact index contains useful modern films."""
    # Use extra candidates because the ratings file also contains episodes and TV series.
    candidate_count = count * 2 if count else 0
    if not candidate_count:
        return {
            record["tconst"]: (float(record["averageRating"]), int(record["numVotes"]))
            for record in rows(ratings_path)
        }
    heap: list[tuple[int, str, float]] = []
    for record in rows(ratings_path):
        votes = int(record["numVotes"])
        item = (votes, record["tconst"], float(record["averageRating"]))
        if len(heap) < candidate_count:
            heapq.heappush(heap, item)
        elif votes > heap[0][0]:
            heapq.heapreplace(heap, item)
    return {tconst: (rating, votes) for votes, tconst, rating in heap}


def build_index(data_directory: Path, max_movies: int, all_datasets: bool) -> Path:
    raw_directory = data_directory / "imdb_raw"
    basics_path = download_dataset("basics", raw_directory)
    crew_path = download_dataset("crew", raw_directory)
    names_path = download_dataset("names", raw_directory)
    ratings_path = download_dataset("ratings", raw_directory)
    if all_datasets:
        download_dataset("principals", raw_directory)
        download_dataset("names", raw_directory)

    index_path = data_directory / "imdb.db"
    data_directory.mkdir(parents=True, exist_ok=True)
    ratings = most_voted_titles(ratings_path, max_movies)
    with sqlite3.connect(index_path) as connection:
        connection.executescript("""
            DROP TABLE IF EXISTS imdb_titles;
            CREATE TABLE imdb_titles (
                tconst TEXT PRIMARY KEY,
                primary_title TEXT NOT NULL,
                original_title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                normalized_search TEXT NOT NULL,
                start_year INTEGER,
                genres TEXT NOT NULL,
                directors TEXT NOT NULL DEFAULT '',
                average_rating REAL,
                num_votes INTEGER NOT NULL DEFAULT 0
            );
        """)
        movie_ids: set[str] = set()
        batch: list[tuple[str, str, str, str, str, int | None, str, float, int]] = []
        for record in rows(basics_path):
            if record["titleType"] != "movie" or record["isAdult"] != "0":
                continue
            rating = ratings.get(record["tconst"])
            if rating is None:
                continue
            year = None if record["startYear"] == "\\N" else int(record["startYear"])
            title = record["primaryTitle"]
            original_title = record["originalTitle"]
            normalized_title = normalize(title)
            batch.append((
                record["tconst"], title, original_title, normalized_title,
                normalize(f"{title} {original_title} {record['genres']}"),
                year, record["genres"], rating[0], rating[1],
            ))
            movie_ids.add(record["tconst"])
            if len(batch) == 5_000:
                connection.executemany("""
                    INSERT INTO imdb_titles (
                        tconst, primary_title, original_title, normalized_title, normalized_search,
                        start_year, genres, average_rating, num_votes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                batch.clear()
        if batch:
            connection.executemany("""
                INSERT INTO imdb_titles (
                    tconst, primary_title, original_title, normalized_title, normalized_search,
                    start_year, genres, average_rating, num_votes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
        if max_movies:
            connection.execute("""
                DELETE FROM imdb_titles WHERE tconst NOT IN (
                    SELECT tconst FROM imdb_titles ORDER BY num_votes DESC LIMIT ?
                )
            """, (max_movies,))
            movie_ids = {row[0] for row in connection.execute("SELECT tconst FROM imdb_titles")}

        director_ids: set[str] = set()
        movie_directors: dict[str, list[str]] = {}
        for record in rows(crew_path):
            if record["tconst"] in movie_ids:
                ids = [] if record["directors"] == "\\N" else record["directors"].split(",")
                movie_directors[record["tconst"]] = ids
                director_ids.update(ids)
        director_names: dict[str, str] = {}
        for record in rows(names_path):
            if record["nconst"] in director_ids:
                director_names[record["nconst"]] = record["primaryName"]
        connection.executemany(
            "UPDATE imdb_titles SET directors = ?, normalized_search = normalized_search || ? WHERE tconst = ?",
            [(
                ", ".join(director_names.get(identifier, identifier) for identifier in ids),
                normalize(" " + " ".join(director_names.get(identifier, identifier) for identifier in ids)),
                tconst,
            )
             for tconst, ids in movie_directors.items()],
        )
        connection.execute("CREATE INDEX imdb_titles_title_idx ON imdb_titles(normalized_title)")
    return index_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the optional IMDb SQLite index.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-movies", type=int, default=250_000, help="0 indexes every movie; default keeps the index manageable.")
    parser.add_argument("--all-datasets", action="store_true", help="Also download principals for future use.")
    arguments = parser.parse_args()
    print(f"IMDb index ready: {build_index(arguments.data_dir, arguments.max_movies, arguments.all_datasets)}")
