from collections import defaultdict, deque
from time import monotonic

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class LocalRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, requests_per_minute: int) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        client = request.client.host if request.client else "local"
        now = monotonic()
        history = self.requests[client]
        while history and history[0] <= now - 60:
            history.popleft()
        if len(history) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Trop de requêtes. Réessayez dans quelques instants."},
                headers={"Retry-After": "60"},
            )
        history.append(now)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response
