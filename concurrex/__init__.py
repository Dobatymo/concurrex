"""Python concurrent execution helpers"""

from .thread import _map_unordered_sem as map_unordered_mt

__version__ = "0.0.1"

__all__ = ["map_unordered_mt"]
