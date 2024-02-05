import logging
import os
import sys


logging.basicConfig(
    format=">> | %(asctime)s | %(levelname)s | %(name)s | %(message)s |",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


def getLogger(name):
    # Setting logging configurations
    logger = logging.getLogger(name)
    return logger

