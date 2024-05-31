import logging
import os
import sys

# ANSI escape sequences for colors
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)

# Define a function to add color codes to the log level names
def add_color(levelname, color_code):
    return f"\033[{color_code}m{levelname}\033[0m"

# Custom formatter class
class ColorFormatter(logging.Formatter):
    COLORS = {
        'WARNING': add_color('WARNING', YELLOW),
        'INFO': add_color('INFO', GREEN),
        'DEBUG': add_color('DEBUG', BLUE),
        'CRITICAL': add_color('CRITICAL', RED),
        'ERROR': add_color('ERROR', MAGENTA)
    }

    def format(self, record):
        record.levelname = self.COLORS.get(record.levelname, record.levelname)
        return super().format(record)

logging.basicConfig(
    format=">> | %(asctime)s | %(levelname)s | %(name)s | %(message)s |",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


def getLogger(name):
    # Setting logging configurations
    logger = logging.getLogger(name)
    color_formatter = ColorFormatter(">> | %(asctime)s | %(levelname)s | %(name)s | %(message)s |",
                                     datefmt='%Y-%m-%d %H:%M:%S',
                                     )
    logger.propagate = False
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(color_formatter)
        logger.addHandler(ch)
    return logger

