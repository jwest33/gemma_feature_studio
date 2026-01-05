from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
import re

# Get the backend directory (where .env should be)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# Model Size Configuration
# =============================================================================

# Maps model size identifiers to their available SAE layers, SAE repos, and hidden dimensions
# Layers are at approximately 25%, 50%, 65%, and 85% depth
MODEL_LAYER_CONFIGS = {
    "270m": {
        "layers": [5, 9, 12, 15],
        "hidden_size": 1152,
        "sae_repo_it": "google/gemma-scope-2-270m-it",
        "sae_repo_pt": "google/gemma-scope-2-270m-pt",
        "model_it": "google/gemma-3-270m-it",
        "model_pt": "google/gemma-3-270m-pt",
        "neuronpedia_model_id": "gemma-3-270m-it",
    },
    "1b": {
        "layers": [7, 13, 17, 22],
        "hidden_size": 2560,
        "sae_repo_it": "google/gemma-scope-2-1b-it",
        "sae_repo_pt": "google/gemma-scope-2-1b-pt",
        "model_it": "google/gemma-3-1b-it",
        "model_pt": "google/gemma-3-1b-pt",
        "neuronpedia_model_id": "gemma-3-1b-it",
    },
    "4b": {
        "layers": [9, 17, 22, 29],
        "hidden_size": 4096,
        "sae_repo_it": "google/gemma-scope-2-4b-it",
        "sae_repo_pt": "google/gemma-scope-2-4b-pt",
        "model_it": "google/gemma-3-4b-it",
        "model_pt": "google/gemma-3-4b-pt",
        "neuronpedia_model_id": "gemma-3-4b-it",
    },
    "12b": {
        "layers": [12, 24, 31, 41],
        "hidden_size": 3840,
        "sae_repo_it": "google/gemma-scope-2-12b-it",
        "sae_repo_pt": "google/gemma-scope-2-12b-pt",
        "model_it": "google/gemma-3-12b-it",
        "model_pt": "google/gemma-3-12b-pt",
        "neuronpedia_model_id": "gemma-3-12b-it",
    },
    "27b": {
        "layers": [16, 31, 40, 53],
        "hidden_size": 5376,
        "sae_repo_it": "google/gemma-scope-2-27b-it",
        "sae_repo_pt": "google/gemma-scope-2-27b-pt",
        "model_it": "google/gemma-3-27b-it",
        "model_pt": "google/gemma-3-27b-pt",
        "neuronpedia_model_id": "gemma-3-27b-it",
    },
}

# Default model size if detection fails
DEFAULT_MODEL_SIZE = "4b"


def detect_model_size(model_name: str) -> str:
    """
    Detect the model size from the model name/path.

    Supports formats like:
    - google/gemma-3-4b-it
    - google/gemma-3-270m-pt
    - /path/to/gemma-3-1b-it
    - gemma-3-12b
    """
    model_lower = model_name.lower()

    # Try to match size patterns
    # Match patterns like "270m", "1b", "4b", "12b", "27b"
    size_patterns = [
        (r"270m", "270m"),
        (r"27b", "27b"),  # Check 27b before 2b to avoid false match
        (r"12b", "12b"),
        (r"4b", "4b"),
        (r"1b", "1b"),
    ]

    for pattern, size in size_patterns:
        if re.search(pattern, model_lower):
            return size

    return DEFAULT_MODEL_SIZE


def get_model_config(model_name: str) -> dict:
    """
    Get the configuration for a model based on its name.

    Returns dict with 'layers', 'sae_repo', 'neuronpedia_model_id', 'hidden_size'.
    """
    size = detect_model_size(model_name)
    config = MODEL_LAYER_CONFIGS.get(size, MODEL_LAYER_CONFIGS[DEFAULT_MODEL_SIZE])

    # Determine if it's an IT or PT model
    is_pt = "-pt" in model_name.lower()

    return {
        "size": size,
        "layers": config["layers"],
        "hidden_size": config["hidden_size"],
        "sae_repo": config["sae_repo_pt"] if is_pt else config["sae_repo_it"],
        "neuronpedia_model_id": config["neuronpedia_model_id"],
    }


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
    dtype: str = "bfloat16"  # Use bfloat16 for large models to fit in VRAM
    top_k_features: int = 50

    # Generation settings
    max_new_tokens: int = 64  # Keep low to prevent generation timeouts
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

        # Memory saver mode: unload SAEs immediately after feature encoding
        # Reduces VRAM usage but requires reload for steering
        self.memory_saver_mode: bool = False

        # Initialize model-specific config
        self._update_model_config()

    def _update_model_config(self, explicit_size: str | None = None):
        """Update model-specific configuration based on explicit size or model name."""
        if explicit_size and explicit_size in MODEL_LAYER_CONFIGS:
            # Use explicit model size
            self.model_size = explicit_size
            config = MODEL_LAYER_CONFIGS[explicit_size]
            is_pt = "-pt" in self.model_name.lower()
            self.available_layers = config["layers"]
            self.hidden_size = config["hidden_size"]
            self.neuronpedia_model_id = config["neuronpedia_model_id"]
            expected_sae_repo = config["sae_repo_pt"] if is_pt else config["sae_repo_it"]
        else:
            # Infer from model name (fallback)
            model_config = get_model_config(self.model_name)
            self.model_size = model_config["size"]
            self.available_layers = model_config["layers"]
            self.hidden_size = model_config["hidden_size"]
            self.neuronpedia_model_id = model_config["neuronpedia_model_id"]
            expected_sae_repo = model_config["sae_repo"]

        # Auto-update SAE repo if it doesn't match the model size
        if self.sae_repo != expected_sae_repo:
            print(f"Auto-updating SAE repo from {self.sae_repo} to {expected_sae_repo}")
            self.sae_repo = expected_sae_repo

    @classmethod
    def get_instance(cls) -> "RuntimeConfig":
        if cls._instance is None:
            cls._instance = RuntimeConfig()
        return cls._instance

    def update(
        self,
        model_name: str | None = None,
        model_size: str | None = None,
        sae_repo: str | None = None,
        sae_width: str | None = None,
        sae_l0: str | None = None,
        sae_type: str | None = None,
        memory_saver_mode: bool | None = None,
    ) -> bool:
        """
        Update runtime configuration.

        Returns True if model reload is needed (model_name changed).
        """
        self._previous_model_name = self.model_name

        if model_name is not None:
            self.model_name = model_name
        # Update model-specific config with explicit size (or infer from name)
        if model_name is not None or model_size is not None:
            self._update_model_config(explicit_size=model_size)
        if sae_repo is not None:
            self.sae_repo = sae_repo
        if sae_width is not None:
            self.sae_width = sae_width
        if sae_l0 is not None:
            self.sae_l0 = sae_l0
        if sae_type is not None:
            self.sae_type = sae_type
        if memory_saver_mode is not None:
            self.memory_saver_mode = memory_saver_mode

        return self._previous_model_name != self.model_name

    def get_available_layers(self) -> list[int]:
        """Get available SAE layers for the current model."""
        return self.available_layers

    def get_default_layer(self) -> int:
        """Get the default layer (first available layer) for the current model."""
        return self.available_layers[0] if self.available_layers else 9

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "hidden_size": self.hidden_size,
            "sae_repo": self.sae_repo,
            "sae_width": self.sae_width,
            "sae_l0": self.sae_l0,
            "sae_type": self.sae_type,
            "available_layers": self.available_layers,
            "neuronpedia_model_id": self.neuronpedia_model_id,
            "memory_saver_mode": self.memory_saver_mode,
        }


def get_settings() -> Settings:
    """Get base settings (from env/defaults)."""
    return Settings()


def get_runtime_config() -> RuntimeConfig:
    """Get runtime configuration (can be modified at runtime)."""
    return RuntimeConfig.get_instance()
