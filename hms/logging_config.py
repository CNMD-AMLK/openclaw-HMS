"""HMS — Basic logging configuration."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for HMS."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# Auto-setup only if HMS is the main entry point (not when imported as a library)
if __name__ == "__main__":
    setup_logging()
