import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def timed(func):
    """This decorator logs the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function that times the execution."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        logger.debug("{} ran in {}s".format(func.__name__, elapsed_time))
        return result

    return wrapper
