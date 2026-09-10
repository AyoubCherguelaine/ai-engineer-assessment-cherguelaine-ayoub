import json

from ..base import AppError, Source, SourceKind
from ..imdb import IMDbAgent


class IMDbSearchTool:
    """Tool contract: search_imdb({"query": "film title, person, or genre"})."""

    name = "search_imdb"
    description = "Search the local IMDb film dataset by title, person, genre, or film-related words."

    def __init__(self, agent: IMDbAgent):
        self.agent = agent

    async def run(self, *, query: str) -> tuple[list[str], list[Source]]:
        if not self.agent.available:
            raise AppError("IMDb data is not indexed. Run: python -m scripts.sync_imdb --max-movies 250000", 503)
        movies = self.agent.search(query)
        if not movies:
            return (
                ["IMDB BULK DATASET: No matching film was found."],
                [Source(kind=SourceKind.IMDB, label="IMDb Non-Commercial Datasets", detail="No matching film")],
            )
        return (
            ["IMDB BULK DATASET — " + json.dumps(movie) for movie in movies],
            [
                Source(kind=SourceKind.IMDB, label="IMDb Non-Commercial Datasets", detail=f"{movie['primary_title']} ({movie['start_year']})")
                for movie in movies
            ],
        )
