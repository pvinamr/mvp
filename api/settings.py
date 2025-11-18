# api/settings.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API defaults
    default_season: int = 2025
    default_week: int = 8
    cache_ttl_seconds: int = 900
    cors_origins: str = "http://localhost:3000"
    odds_api_key: str = ""

    class Config:
        env_file = ".env"   # will look for api/.env


settings = Settings()
