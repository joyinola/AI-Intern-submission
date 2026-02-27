
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import assessment, info
from .dependencies import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eagerly load the model at startup so any missing-artefact errors
    are raised immediately — before the health check passes and traffic arrives.
    """
    print("Check Me: loading model artefacts...")
    get_model()
    print("Check Me: model ready.")
    yield


app = FastAPI(
    title="Check Me — Breast Cancer Risk API",
    description=(
        "Clinical Decision Support for Breast Cancer Risk Stratification.\n\n"
        "⚕ **For clinical use only under qualified supervision.** "
        "This tool does not diagnose breast cancer. All outputs require clinician validation.\n\n"
        "Based on: ACS 2023, NICE NG101, BI-RADS 5th Edition."
    ),
    version="1.0.0",
    # root_path tells FastAPI it is mounted at /api behind the Nginx proxy.
    # This makes Swagger UI use the correct server URL and fixes openapi.json.
    root_path="/api",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(assessment.router)
app.include_router(info.router)


@app.get("/health", tags=["System"])
def health():
    """
    Liveness probe used by Docker health check.
    Only returns 200 after the model has been loaded successfully.
    """
    return {"status": "ok", "system": "Check Me", "version": "1.0.0"}
