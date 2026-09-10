import json

from ..base import Source, SourceKind
from ..superhero import SuperheroAgent


class SuperheroSearchTool:
    """Tool contract: search_superhero({"name": "superhero name"})."""

    name = "search_superhero"
    description = "Look up factual superhero details by name using Superhero API."

    def __init__(self, agent: SuperheroAgent | None):
        self.agent = agent

    async def run(self, *, name: str) -> tuple[list[str], list[Source]]:
        if self.agent is None:
            from ..base import AppError
            raise AppError("SUPERHERO_API_TOKEN is not configured.")
        hero = await self.agent.search(name)
        return (
            ["SUPERHERO API RESULT — " + json.dumps(hero, ensure_ascii=False)],
            [Source(kind=SourceKind.SUPERHERO_API, label="Superhero API", detail=f"Search result: {hero.get('name', 'unknown')}")],
        )
