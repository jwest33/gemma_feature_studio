"""SAE Registry for managing multiple loaded SAEs with LRU eviction."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import time
import torch

if TYPE_CHECKING:
    from app.inference.vram_monitor import VRAMMonitor


def _free_sae_vram(sae: "torch.nn.Module") -> None:
    """
    Properly free SAE VRAM by moving tensors to CPU before deletion.

    Simply calling `del sae` doesn't immediately free CUDA memory.
    Moving to CPU first ensures the GPU memory is released.
    """
    try:
        sae.cpu()  # Move all parameters to CPU, freeing GPU memory
    except Exception:
        pass  # If already on CPU or other issue, continue
    del sae


@dataclass
class SAEEntry:
    """Represents a loaded SAE with metadata."""
    sae: "torch.nn.Module"  # JumpReLUSAE instance
    layer: int
    width: str
    l0: str
    sae_type: str
    size_bytes: int
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def sae_id(self) -> str:
        """Get unique SAE identifier string."""
        return f"{self.sae_type}/layer_{self.layer}_width_{self.width}_l0_{self.l0}"

    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used = time.time()

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "layer": self.layer,
            "width": self.width,
            "l0": self.l0,
            "sae_type": self.sae_type,
            "sae_id": self.sae_id,
            "size_mb": round(self.size_bytes / (1024**2), 2),
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
        }


class SAERegistry:
    """
    Registry for managing multiple loaded SAEs with LRU eviction.

    Maintains SAEs in an OrderedDict for LRU ordering.
    Most recently used SAEs are moved to the end.
    """

    def __init__(
        self,
        vram_monitor: Optional["VRAMMonitor"] = None,
        max_sae_budget_gb: float = 20.0,
    ):
        """
        Initialize SAE registry.

        Args:
            vram_monitor: Optional VRAMMonitor instance for memory checks
            max_sae_budget_gb: Maximum GB to use for all SAEs combined
        """
        self.vram_monitor = vram_monitor
        self.max_sae_budget_bytes = int(max_sae_budget_gb * 1024**3)
        self._saes: OrderedDict[str, SAEEntry] = OrderedDict()
        self._total_loaded_bytes = 0

    def _make_key(self, layer: int, width: str, l0: str, sae_type: str) -> str:
        """Create unique key for SAE configuration."""
        return f"{sae_type}/layer_{layer}_width_{width}_l0_{l0}"

    def get(
        self,
        layer: int,
        width: str = "65k",
        l0: str = "medium",
        sae_type: str = "resid_post",
    ) -> Optional[SAEEntry]:
        """
        Get a loaded SAE by configuration.

        Updates last_used timestamp and moves to end of OrderedDict (most recent).

        Args:
            layer: Layer index
            width: SAE width (16k, 65k, 262k, 1m)
            l0: L0 regularization level (small, medium, big)
            sae_type: Hook type (resid_post, attn_out, mlp_out, transcoder)

        Returns:
            SAEEntry if found, None otherwise
        """
        key = self._make_key(layer, width, l0, sae_type)
        if key in self._saes:
            entry = self._saes[key]
            entry.touch()
            # Move to end (most recently used)
            self._saes.move_to_end(key)
            return entry
        return None

    def get_by_layer(self, layer: int) -> Optional[SAEEntry]:
        """
        Get any SAE loaded for a specific layer.

        Args:
            layer: Layer index

        Returns:
            First SAEEntry found for layer, None otherwise
        """
        for entry in self._saes.values():
            if entry.layer == layer:
                entry.touch()
                return entry
        return None

    def register(
        self,
        sae: "torch.nn.Module",
        layer: int,
        width: str,
        l0: str,
        sae_type: str,
        size_bytes: int,
    ) -> SAEEntry:
        """
        Register a newly loaded SAE.

        Args:
            sae: The JumpReLUSAE instance
            layer: Layer index
            width: SAE width string
            l0: L0 regularization level
            sae_type: Hook type
            size_bytes: Actual size in bytes

        Returns:
            The created SAEEntry
        """
        key = self._make_key(layer, width, l0, sae_type)

        # If already exists, unload first
        if key in self._saes:
            self.unload(layer, width, l0, sae_type)

        entry = SAEEntry(
            sae=sae,
            layer=layer,
            width=width,
            l0=l0,
            sae_type=sae_type,
            size_bytes=size_bytes,
        )

        self._saes[key] = entry
        self._total_loaded_bytes += size_bytes
        return entry

    def unload(
        self,
        layer: int,
        width: str = "65k",
        l0: str = "medium",
        sae_type: str = "resid_post",
    ) -> bool:
        """
        Unload a specific SAE.

        Args:
            layer: Layer index
            width: SAE width
            l0: L0 regularization level
            sae_type: Hook type

        Returns:
            True if SAE was found and unloaded, False otherwise
        """
        key = self._make_key(layer, width, l0, sae_type)
        if key in self._saes:
            entry = self._saes.pop(key)
            self._total_loaded_bytes -= entry.size_bytes
            _free_sae_vram(entry.sae)
            torch.cuda.empty_cache()
            return True
        return False

    def unload_layer(self, layer: int) -> bool:
        """
        Unload any SAE for a specific layer.

        Args:
            layer: Layer index

        Returns:
            True if any SAE was unloaded
        """
        keys_to_remove = [
            key for key, entry in self._saes.items()
            if entry.layer == layer
        ]

        for key in keys_to_remove:
            entry = self._saes.pop(key)
            self._total_loaded_bytes -= entry.size_bytes
            _free_sae_vram(entry.sae)

        if keys_to_remove:
            torch.cuda.empty_cache()
            return True
        return False

    def evict_lru(self, bytes_to_free: int) -> int:
        """
        Evict least recently used SAEs until bytes_to_free is available.

        Args:
            bytes_to_free: Minimum bytes to free

        Returns:
            Actual bytes freed
        """
        freed = 0
        to_evict = []

        # Iterate from oldest to newest (OrderedDict maintains insertion order)
        for key, entry in self._saes.items():
            if freed >= bytes_to_free:
                break
            to_evict.append(key)
            freed += entry.size_bytes

        for key in to_evict:
            entry = self._saes.pop(key)
            self._total_loaded_bytes -= entry.size_bytes
            print(f"Evicted SAE: {entry.sae_id} ({entry.size_bytes / (1024**2):.1f}MB)")
            _free_sae_vram(entry.sae)

        if to_evict:
            torch.cuda.empty_cache()

        return freed

    def get_loaded_layers(self) -> list[int]:
        """Get list of currently loaded layer indices (sorted)."""
        layers = list(set(entry.layer for entry in self._saes.values()))
        return sorted(layers)

    def get_loaded_entries(self) -> list[SAEEntry]:
        """Get all loaded SAE entries."""
        return list(self._saes.values())

    def get_total_loaded_bytes(self) -> int:
        """Get total bytes used by all loaded SAEs."""
        return self._total_loaded_bytes

    def get_total_loaded_gb(self) -> float:
        """Get total GB used by all loaded SAEs."""
        return self._total_loaded_bytes / (1024**3)

    def is_loaded(self, layer: int, width: str = "65k", l0: str = "medium", sae_type: str = "resid_post") -> bool:
        """Check if a specific SAE configuration is loaded."""
        key = self._make_key(layer, width, l0, sae_type)
        return key in self._saes

    def count(self) -> int:
        """Get number of loaded SAEs."""
        return len(self._saes)

    def clear_all(self) -> None:
        """Unload all SAEs."""
        for entry in self._saes.values():
            _free_sae_vram(entry.sae)
        self._saes.clear()
        self._total_loaded_bytes = 0
        torch.cuda.empty_cache()
        print("All SAEs cleared from registry")

    def get_status(self) -> dict:
        """Get registry status for API responses."""
        return {
            "loaded_count": self.count(),
            "loaded_layers": self.get_loaded_layers(),
            "total_size_mb": round(self._total_loaded_bytes / (1024**2), 2),
            "total_size_gb": round(self.get_total_loaded_gb(), 3),
            "max_budget_gb": round(self.max_sae_budget_bytes / (1024**3), 2),
            "entries": [entry.to_dict() for entry in self._saes.values()],
        }
