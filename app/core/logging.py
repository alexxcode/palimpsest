import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
