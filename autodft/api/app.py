"""FastAPI application factory for the AutoDFT dashboard."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from autodft.api.routes import router, set_active_settings
from autodft.config import Settings


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
    app.include_router(router)
    return app
