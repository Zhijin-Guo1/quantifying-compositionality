"""Data loaders for real datasets."""

from .movielens_loader import MovieLensLoader
from .dialogue_loader import DialogueLoader
from .morpholex_loader import MorphoLEXLoader

__all__ = [
    'MovieLensLoader',
    'DialogueLoader',
    'MorphoLEXLoader'
]