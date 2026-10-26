"""Baselines classiques (sans apprentissage)."""

from .watershed import WatershedConfig, WatershedSegmenter, watershed_segment

__all__ = ["WatershedConfig", "WatershedSegmenter", "watershed_segment"]
