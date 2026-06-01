"""FastAPI application factory for the AutoDFT dashboard."""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from autodft.api.auth import is_authenticated
from autodft.api.routes import public_router, router, set_active_settings
from autodft.config import Settings


# Path prefixes that bypass the auth middleware. ``/login`` and
# ``/logout`` belong to the unauthenticated flow; ``/static`` is
# reserved for future asset serving.
_PUBLIC_PREFIXES = ("/login", "/logout", "/static/")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    Args:
        settings: The Settings used by the running controller. When
                  provided, route handlers reach it through
                  ``autodft.api.routes.get_active_settings()`` so that
                  ``[storage].data_path`` and friends are honoured even
                  when env vars / a CLI ``--config`` aren't in scope.
    """
    if settings is not None:
        set_active_settings(settings)

    app = FastAPI(
        title="AutoDFT Dashboard",
        version="0.1.0",
        description="Monitoring dashboard and REST API for the AutoDFT pipeline.",
    )

    # ------------------------------------------------------------------
    # Auth middleware. Gates both the HTML dashboard and the /api/*
    # endpoints. Public routes (/login, /logout) are exempt. Unauthenticated
    # API requests get HTTP 401 JSON; unauthenticated browser requests
    # get a 303 redirect to /login with the original path preserved.
    # ------------------------------------------------------------------

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        # Resolve settings lazily so test setups that call create_app()
        # without a settings argument still work.
        from autodft.api.routes import get_active_settings
        if is_authenticated(request, get_active_settings()):
            return await call_next(request)

        if path.startswith("/api/"):
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required. "
                                   "Send the password via the X-AutoDFT-Password "
                                   "header or sign in at /login first."},
            )
        return RedirectResponse(url=f"/login?next={quote(path)}", status_code=303)

    app.include_router(public_router)
    app.include_router(router)
    return app
