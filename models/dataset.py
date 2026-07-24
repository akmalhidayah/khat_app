"""Backward-compatible re-export — dataset records live in gambar_dataset."""

from .dataset_image import Dataset, GambarDataset

__all__ = ["Dataset", "GambarDataset"]
