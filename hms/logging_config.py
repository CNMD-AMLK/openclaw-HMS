"""HMS — Basic logging configuration."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for HMS."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


setup_logging()
