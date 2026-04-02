"""HMS v3 — Hierarchical Memory Scaffold scripts package."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for HMS."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# Setup logging once when the scripts package is imported
setup_logging()
