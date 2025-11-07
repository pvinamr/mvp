from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API
    cache_ttl_seconds: int = 900   # 15 min default
    default_season: int = 2025
    default_week: int = 10
    cors_origins: str = "*"        # comma-separated list in prod

    model_config = SettingsConfigDict(env_file=".env", env_prefix="API_", case_sensitive=False)

settings = Settings()