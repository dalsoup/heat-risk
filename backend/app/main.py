# backend/app/main.py
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------
# Settings (fallback-safe)
# --------------------------------
try:
    from .core.config import settings  # type: ignore
except Exception:
    class _FallbackSettings:
        API_PREFIX = ""
        DEBUG = True

        def cors_origins_list(self) -> List[str]:
            return ["http://localhost:3000", "http://127.0.0.1:3000"]
    settings = _FallbackSettings()  # type: ignore

logger = logging.getLogger("heatrisk")
logging.basicConfig(level=logging.INFO)

API_PREFIX = getattr(settings, "API_PREFIX", "") or ""

# --------------------------------
# Lifespan (선택)
# --------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from .db import init_db  # type: ignore
        init_db()
        logger.info("DB initialized.")
    except Exception as e:
        logger.warning(f"DB init skipped/failed: {e}")
    yield

# --------------------------------
# App
# --------------------------------
app = FastAPI(
    title="HeatRisk API",
    version="1.0.0",
    lifespan=lifespan,
)

# --------------------------------
# CORS
# --------------------------------
try:
    origins = settings.cors_origins_list()
except Exception:
    origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

# 개발 편의: 구체 오리진 + credentials 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# Routers
# --------------------------------
from .routers import health as _health  # type: ignore
app.include_router(_health.router, prefix=API_PREFIX)

try:
    from .routers import auth as _auth  # type: ignore
    app.include_router(_auth.router, prefix=API_PREFIX)
except Exception as e:
    logger.warning(f"auth router skipped: {e}")

try:
    from .routers import predict as _predict  # type: ignore
    app.include_router(_predict.router, prefix=API_PREFIX)
except Exception as e:
    logger.warning(f"predict router skipped: {e}")

try:
    from .routers import topn as _topn  # type: ignore
    app.include_router(_topn.router, prefix=API_PREFIX)
except Exception as e:
    logger.warning(f"topn router skipped: {e}")

try:
    from .routers import dong as _dong  # type: ignore
    app.include_router(_dong.router, prefix=API_PREFIX)
except Exception as e:
    logger.warning(f"dong router skipped: {e}")

# --------------------------------
# Root & Health
# --------------------------------
@app.get(f"{API_PREFIX}/")
def root():
    return {
        "name": "HeatRisk API",
        "version": "1.0.0",
        "routers": ["health", "auth?", "predict?", "topn?", "dong?"],
        "prefix": API_PREFIX or "/",
        "cors": {"allow_origins": origins, "allow_credentials": True},
    }

@app.get(f"{API_PREFIX}/health")
def health():
    return {"ok": True}
