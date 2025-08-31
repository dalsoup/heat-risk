# backend/app/core/config.py
import os
from typing import List

class Settings:
    # Security / CORS
    SECRET_KEY: str = os.getenv("SECRET_KEY", "devsecret_change_me")
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./heatrisk.db")
    # 예: postgresql+psycopg2://postgres:postgres@db:5432/heatrisk

    # Artifacts
    MODEL_LOGREG_PATH: str = os.getenv("MODEL_LOGREG_PATH", "models/logreg_personal_latest.joblib")
    MODEL_XGB_PATH: str    = os.getenv("MODEL_XGB_PATH",    "models/xgb_personal_latest.joblib")
    INFER_PATH: str        = os.getenv("INFER_PATH",        "data/train/personal_features_infer_latest.csv")

    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24)))

    CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"

    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

settings = Settings()
