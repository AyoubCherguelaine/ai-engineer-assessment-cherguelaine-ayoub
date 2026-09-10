"""Read-only search over a locally indexed copy of IMDb's public TSV datasets."""

import re
import sqlite3
from pathlib import Path

BASE_URL = "https://datasets.imdbws.com"
DATASETS = {
    "basics": "title.basics.tsv.gz",
    "crew": "title.crew.tsv.gz",
    "principals": "title.principals.tsv.gz",
    "names": "name.basics.tsv.gz",
    "ratings": "title.ratings.tsv.gz",
}

STOP_WORDS = frozenset({
    "a", "an", "about", "and", "are", "by", "did", "directed", "film", "films",
    "for", "from", "is", "list", "movie", "movies", "of", "show", "tell", "the",
    "what", "which", "who", "with",
})


def normalize(value: str) -> str:
    """Make title variants equivalent: Batman, Bat-Man, and Bat Man -> batman."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def search_terms(question: str) -> list[str]:
    words = [word for word in re.findall(r"[a-z0-9]+", question.lower()) if word not in STOP_WORDS]
    if not words:
        return []
    # Prefer the full meaningful phrase, then individual words. This ranks
    # "The Dark Knight" and spelling/punctuation variants ahead of loose matches.
    phrases = [normalize(" ".join(words))]
    # A joined phrase such as "bat man" becomes "batman". Do not also search
    # its short fragments ("bat", "man"), which are too broad to be useful.
    phrases.extend(normalize(word) for word in words if len(normalize(word)) >= 4)
    return list(dict.fromkeys(term for term in phrases if len(term) >= 3))


class IMDbAgent:
    """Searches `data/imdb.db` after it has been prepared by scripts/sync_imdb.py."""

    def __init__(self, database_path: Path | str | None = None):
        default_path = Path(__file__).resolve().parents[1] / "data" / "imdb.db"
        self.database_path = Path(database_path or default_path)

    @property
    def available(self) -> bool:
        return self.database_path.is_file()

    def search(self, question: str, limit: int = 3) -> list[dict[str, str | int | float | None]]:
        if not self.available:
            return []
        terms = search_terms(question)
        if not terms:
            return []
        where_clause = " OR ".join("normalized_search LIKE ?" for _ in terms)
        query = f"""
            SELECT tconst, primary_title, start_year, genres, directors, average_rating, num_votes
            FROM imdb_titles
            WHERE {where_clause}
            ORDER BY CASE WHEN normalized_title = ? THEN 1 ELSE 0 END DESC, num_votes DESC
            LIMIT ?
        """
        parameters = [f"%{term}%" for term in terms] + [terms[0], limit]
        try:
            with sqlite3.connect(self.database_path) as connection:
                connection.row_factory = sqlite3.Row
                return [dict(row) for row in connection.execute(query, parameters).fetchall()]
        except sqlite3.DatabaseError:
            # A missing or incomplete local index must not take the API down.
            return []
