"""Adapters exposing trained PyTorch models through the NumPy ``Restorer`` interface."""

from .torch_restorer import TorchRestorer

__all__ = ["TorchRestorer"]
