# backend/app/serve_api.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 라우터 import
from app.routers import auth, dong, health, predict, topn

# ---------------------------------------------------
# FastAPI app 생성
# ---------------------------------------------------
app = FastAPI(
    title="Heat Risk API",
    description="온열질환 위험도 예측 및 관련 기능 제공 API",
    version="1.0.0",
)

# ---------------------------------------------------
# CORS 허용 
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# 라우터 등록
# ---------------------------------------------------
app.include_router(auth.router)
app.include_router(dong.router)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(topn.router)

# ---------------------------------------------------
# 기본 엔드포인트 (헬스체크)
# ---------------------------------------------------
@app.get("/")
def root():
    return {"msg": "Heat Risk API is running"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
