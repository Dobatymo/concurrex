"""Python concurrent execution helpers"""

from .thread import ThreadPool
from .thread import map_unordered as map_unordered_mt

__version__ = "0.0.2"

__all__ = ["ThreadPool", "map_unordered_mt"]
