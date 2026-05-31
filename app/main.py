from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_config import configure_logging
from app.routes import analysis, health, instagram, retrieval, youtube


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Keep routers listed in the same order a user would hit the app.
    app.include_router(health.router)
    app.include_router(analysis.router)
    app.include_router(youtube.router)
    app.include_router(instagram.router)
    app.include_router(retrieval.router)
    return app


app = create_app()
