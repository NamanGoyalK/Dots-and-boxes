"""Tournament package for Dots and Boxes."""
from .match import play_series, SeriesResult
from .benchmark import Tournament

__all__ = ["play_series", "SeriesResult", "Tournament"]
