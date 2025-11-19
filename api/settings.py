# api/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API defaults
    default_season: int = 2025
    default_week: int = 12
    cache_ttl_seconds: int = 900

    # CORS
    cors_origins: str = "http://localhost:3000"

    # Odds API key (from .env: ODDS_API_KEY=...)
    odds_api_key: str = ""

    # Optional DB URL (from .env: DATABASE_URL=...)
    # We still read DATABASE_URL directly in db.py, but this
    # field exists so pydantic-settings doesn't complain.
    database_url: str | None = None

    # Tell pydantic-settings where to read env vars from,
    # and to ignore any extra ones.
    model_config = SettingsConfigDict(
        env_file=".env",    # if your .env is in project root
        extra="ignore",     # don't error on extra env vars
    )


settings = Settings()
