"""Inference research foundations; approximate reuse is never the default."""
from .engine import CacheEngine, Snapshot, Edit, UnsupportedEdit

__all__ = ["CacheEngine", "Snapshot", "Edit", "UnsupportedEdit"]
