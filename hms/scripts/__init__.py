"""HMS v3 — Hierarchical Memory Scaffold scripts package."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for HMS."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # Add a StreamHandler to stdout so users can see logs in the console
    root = logging.getLogger()
    # Only add if no StreamHandler already exists
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(handler)


# Setup logging once when the scripts package is imported
setup_logging()
