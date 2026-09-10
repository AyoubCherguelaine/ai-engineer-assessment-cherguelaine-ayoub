import logging
import re
from typing import Any

import httpx

from .base import AppError

logger = logging.getLogger(__name__)


class SuperheroAgent:
    """Fetches one named superhero record from Superhero API."""

    def __init__(self, token: str, timeout: float):
        self.token, self.timeout = token, timeout

    async def search(self, name: str) -> dict[str, Any]:
        if not self.token:
            raise AppError("SUPERHERO_API_TOKEN is not configured.")
        safe_name = re.sub(r"[^a-zA-Z0-9 .'-]", "", name).strip()
        if not safe_name:
            raise AppError("I could not identify a superhero name in that question.", 422)
        # The service redirects /api/ to /api.php/; follow it without exposing the token in logs.
        url = f"https://www.superheroapi.com/api/{self.token}/search/{safe_name}"
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
                break
            except httpx.TimeoutException as exc:
                if attempt == 0:
                    logger.info("Superhero API timed out; retrying once")
                    continue
                logger.warning("Superhero API timed out after retry")
                raise AppError("The Superhero API timed out. Please try again.", 504) from exc
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("Superhero API request failed: %s", type(exc).__name__)
                raise AppError("The Superhero API is temporarily unavailable.", 502) from exc
        if data.get("response") != "success" or not data.get("results"):
            raise AppError(f"No superhero data was found for '{safe_name}'.", 404)
        return data["results"][0]
