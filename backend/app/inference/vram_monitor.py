"""VRAM monitoring and memory management for GPU resources."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import torch


class MemoryPressure(Enum):
    """Memory pressure levels for VRAM usage."""
    LOW = "low"           # < 50% used
    MODERATE = "moderate" # 50-75% used
    HIGH = "high"         # 75-90% used
    CRITICAL = "critical" # > 90% used


@dataclass
class VRAMSnapshot:
    """Snapshot of current VRAM usage."""
    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int
    pressure: MemoryPressure

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def allocated_gb(self) -> float:
        return self.allocated_bytes / (1024**3)

    @property
    def reserved_gb(self) -> float:
        return self.reserved_bytes / (1024**3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024**3)

    def to_dict(self) -> dict:
        return {
            "total_gb": round(self.total_gb, 2),
            "allocated_gb": round(self.allocated_gb, 2),
            "reserved_gb": round(self.reserved_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "pressure": self.pressure.value,
        }


class VRAMMonitor:
    """Monitors GPU VRAM usage and provides allocation guidance."""

    def __init__(self, device: str = "cuda", safety_margin_gb: float = 2.0):
        """
        Initialize VRAM monitor.

        Args:
            device: CUDA device string
            safety_margin_gb: Buffer to keep free to avoid OOM
        """
        self.device = device
        self.safety_margin_bytes = int(safety_margin_gb * 1024**3)
        self._device_index = self._parse_device_index(device)

    def _parse_device_index(self, device: str) -> int:
        """Parse device index from device string."""
        if device == "cuda":
            return 0
        elif device.startswith("cuda:"):
            return int(device.split(":")[1])
        return 0

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    def get_snapshot(self) -> VRAMSnapshot:
        """Get current VRAM usage snapshot."""
        if not torch.cuda.is_available():
            # Return empty snapshot for CPU-only systems
            return VRAMSnapshot(
                total_bytes=0,
                allocated_bytes=0,
                reserved_bytes=0,
                free_bytes=0,
                pressure=MemoryPressure.LOW,
            )

        # Sync and clear cache before measuring
        torch.cuda.synchronize(self._device_index)

        total = torch.cuda.get_device_properties(self._device_index).total_memory
        allocated = torch.cuda.memory_allocated(self._device_index)
        reserved = torch.cuda.memory_reserved(self._device_index)

        # Free memory is what's actually available for new allocations
        # Use allocated (not reserved) for more accurate free calculation
        # Reserved includes cached memory that can be reused
        free = total - allocated

        # Ensure free is never negative (can happen with memory mapping)
        free = max(0, free)

        # Calculate pressure level based on allocated (actual usage)
        usage_ratio = allocated / total if total > 0 else 0
        if usage_ratio < 0.50:
            pressure = MemoryPressure.LOW
        elif usage_ratio < 0.75:
            pressure = MemoryPressure.MODERATE
        elif usage_ratio < 0.90:
            pressure = MemoryPressure.HIGH
        else:
            pressure = MemoryPressure.CRITICAL

        return VRAMSnapshot(
            total_bytes=total,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            free_bytes=free,
            pressure=pressure,
        )

    def can_allocate(self, bytes_needed: int) -> bool:
        """Check if we can safely allocate the requested bytes."""
        snapshot = self.get_snapshot()
        return snapshot.free_bytes > (bytes_needed + self.safety_margin_bytes)

    def estimate_sae_size(self, width: str, dtype: torch.dtype = torch.float32, d_model: int = 4096) -> int:
        """
        Estimate SAE memory footprint based on width and model hidden dimension.

        The SAE has these parameters:
        - w_enc: (d_model, d_sae) - encoder weights
        - w_dec: (d_sae, d_model) - decoder weights
        - threshold: (d_sae,) - threshold values
        - b_enc: (d_sae,) - encoder bias
        - b_dec: (d_model,) - decoder bias

        Args:
            width: SAE width string (16k, 65k, 262k, 1m)
            dtype: Data type for parameters
            d_model: Model hidden dimension (varies by model size)

        Returns:
            Estimated size in bytes
        """
        width_map = {
            "16k": 16384,
            "65k": 65536,
            "262k": 262144,
            "1m": 1048576,
        }
        d_sae = width_map.get(width, 65536)

        bytes_per_param = 4 if dtype == torch.float32 else 2

        # Total parameters
        num_params = (
            d_model * d_sae +  # w_enc
            d_sae * d_model +  # w_dec
            d_sae +            # threshold
            d_sae +            # b_enc
            d_model            # b_dec
        )

        return num_params * bytes_per_param

    def get_available_for_saes(self, model_loaded: bool = True) -> int:
        """
        Get bytes available for loading SAEs.

        Args:
            model_loaded: Whether the main model is already loaded

        Returns:
            Available bytes for SAEs
        """
        snapshot = self.get_snapshot()
        return max(0, snapshot.free_bytes - self.safety_margin_bytes)

    def suggest_max_layers(self, width: str = "65k", dtype: torch.dtype = torch.float32, d_model: int = 4096) -> int:
        """
        Suggest maximum number of SAE layers that can be loaded.

        Args:
            width: SAE width string
            dtype: Data type for SAE parameters
            d_model: Model hidden dimension

        Returns:
            Maximum number of layers (0-4)
        """
        available = self.get_available_for_saes()
        sae_size = self.estimate_sae_size(width, dtype, d_model)

        if sae_size == 0:
            return 0

        return min(4, available // sae_size)

    def format_bytes(self, bytes_val: int) -> str:
        """Format bytes as human-readable string."""
        if bytes_val >= 1024**3:
            return f"{bytes_val / (1024**3):.2f} GB"
        elif bytes_val >= 1024**2:
            return f"{bytes_val / (1024**2):.2f} MB"
        elif bytes_val >= 1024:
            return f"{bytes_val / 1024:.2f} KB"
        return f"{bytes_val} B"
