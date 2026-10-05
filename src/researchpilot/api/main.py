from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from researchpilot.api.deps import Components, build_components
from researchpilot.api.routes import chat, health


def create_app(components: Components | None = None) -> FastAPI:
    """
    Build the app. Pass `components` (fakes) in tests to skip
    loading the embedding model, reranker, Chroma and Gemini.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.components = components or build_components()
        yield

    app = FastAPI(title="ResearchPilot", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api/v1")

    return app


app = create_app()
