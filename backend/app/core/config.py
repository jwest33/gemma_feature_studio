from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

# Get the backend directory (where .env should be)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API settings
    api_title: str = "Gemma Feature Studio API"
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

    # Neuronpedia API settings
    neuronpedia_api_key: str | None = None
    neuronpedia_username: str | None = None
    neuronpedia_base_url: str = "https://www.neuronpedia.org/api"
    # Model ID for Neuronpedia (maps to their naming convention)
    neuronpedia_model_id: str = "gemma-3-4b-it"


# Runtime configuration state (allows dynamic updates without reloading settings)
class RuntimeConfig:
    """Mutable runtime configuration that can be updated via API."""

    _instance: Optional["RuntimeConfig"] = None

    def __init__(self):
        # Initialize from settings
        settings = Settings()
        self.model_name: str = settings.model_name
        self.sae_repo: str = settings.sae_repo
        self.sae_width: str = settings.sae_width
        self.sae_l0: str = settings.sae_l0
        self.sae_type: str = settings.sae_type
        self._previous_model_name: str = settings.model_name

    @classmethod
    def get_instance(cls) -> "RuntimeConfig":
        if cls._instance is None:
            cls._instance = RuntimeConfig()
        return cls._instance

    def update(
        self,
        model_name: str | None = None,
        sae_repo: str | None = None,
        sae_width: str | None = None,
        sae_l0: str | None = None,
        sae_type: str | None = None,
    ) -> bool:
        """
        Update runtime configuration.

        Returns True if model reload is needed (model_name changed).
        """
        self._previous_model_name = self.model_name

        if model_name is not None:
            self.model_name = model_name
        if sae_repo is not None:
            self.sae_repo = sae_repo
        if sae_width is not None:
            self.sae_width = sae_width
        if sae_l0 is not None:
            self.sae_l0 = sae_l0
        if sae_type is not None:
            self.sae_type = sae_type

        return self._previous_model_name != self.model_name

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "sae_repo": self.sae_repo,
            "sae_width": self.sae_width,
            "sae_l0": self.sae_l0,
            "sae_type": self.sae_type,
        }


def get_settings() -> Settings:
    """Get base settings (from env/defaults)."""
    return Settings()


def get_runtime_config() -> RuntimeConfig:
    """Get runtime configuration (can be modified at runtime)."""
    return RuntimeConfig.get_instance()
