from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

# Get the backend directory (where .env should be)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API settings
    api_title: str = "LM Feature Studio API"
    api_version: str = "0.1.0"
    debug: bool = True

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Model settings (Gemma 3 with Gemma Scope 2 SAEs)
    # Available layers for 4B: 9, 17, 22, 29
    # Available widths: 16k, 65k, 262k, 1m
    # Available L0: small, medium, big
    model_name: str = "google/gemma-3-4b-it"
    sae_repo: str = "google/gemma-scope-2-4b-it"
    sae_layer: int = 17
    sae_width: str = "65k"
    sae_l0: str = "medium"
    sae_type: str = "resid_post"  # resid_post, attn_out, mlp_out, transcoder

    # Inference settings
    device: str = "cuda"
    dtype: str = "float32"
    top_k_features: int = 50

    # Generation settings
    max_new_tokens: int = 256
    default_temperature: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
