import os
import warnings
from typing import Callable, Optional, TypeVar

T = TypeVar("T", int, float)


def parse_optional_positive_env(
    env_name: str,
    *,
    default: T,
    parser: Callable[[str], T],
) -> Optional[T]:
    """Parse a positive numeric env var, with non-positive values disabling it."""
    value = os.environ.get(env_name)
    if value is None:
        return default
    try:
        parsed = parser(value)
    except ValueError:
        warnings.warn(
            f"Invalid {env_name} value; using default {default}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default
    if parsed <= 0:
        return None
    return parsed
