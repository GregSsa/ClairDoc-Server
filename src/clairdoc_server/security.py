import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Request, status


async def require_api_key(
    request: Request,
    x_clairdoc_key: Annotated[str | None, Header()] = None,
) -> None:
    expected_key = request.app.state.settings.api_key
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="L'authentification du serveur n'est pas configurée.",
        )
    if x_clairdoc_key is None or not secrets.compare_digest(x_clairdoc_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé ClairDoc invalide.",
        )
