"""Reusable configuration and environment utilities for the LLM pipeline."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on environment
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - depends on environment
    torch = None  # type: ignore[assignment]


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON dictionary from disk.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        The parsed JSON content as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the path is not a file or the JSON content is invalid.
        TypeError: If the JSON root is not an object.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Configuration path is not a file: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in configuration file: {config_path}") from exc

    if not isinstance(data, dict):
        raise TypeError(f"Configuration file must contain a JSON object: {config_path}")

    return data


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    """Save a dictionary as formatted JSON to disk.

    Args:
        path: Destination path for the JSON file.
        data: Dictionary data to serialize.

    Returns:
        None.

    Raises:
        ValueError: If the provided data is not a dictionary.
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be provided as a dictionary.")

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)
        handle.write("\n")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return its path.

    Args:
        path: Directory path to ensure exists.

    Returns:
        A Path object pointing to the created directory.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger with a console handler.

    Args:
        name: Logger name.

    Returns:
        A configured logger instance.

    Raises:
        ValueError: If the logger name is empty.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Logger name must be a non-empty string")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger


def set_seed(seed: int) -> None:
    """Set the random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.

    Returns:
        None.

    Raises:
        TypeError: If seed is not an integer.
    """
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
