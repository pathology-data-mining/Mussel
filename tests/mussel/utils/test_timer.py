import time

from mussel.utils.timer import timed


def test_timed_decorator():
    """Test that the timed decorator works and returns the correct result"""

    @timed
    def example_function(x, y):
        time.sleep(0.01)  # Small delay to ensure measurable time
        return x + y

    result = example_function(2, 3)
    assert result == 5


def test_timed_decorator_with_no_args():
    """Test that the timed decorator works with no arguments"""

    @timed
    def no_arg_function():
        time.sleep(0.01)
        return "done"

    result = no_arg_function()
    assert result == "done"


def test_timed_decorator_with_kwargs():
    """Test that the timed decorator works with keyword arguments"""

    @timed
    def kwargs_function(a, b=10):
        time.sleep(0.01)
        return a * b

    result = kwargs_function(5, b=3)
    assert result == 15


def test_timed_decorator_preserves_function_name():
    """Test that the timed decorator preserves the original function name"""

    @timed
    def my_named_function():
        return True

    assert my_named_function.__name__ == "my_named_function"
