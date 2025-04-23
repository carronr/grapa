import numpy as np
import pytest

from grapa.curve import Curve

from grapa.mathModule import smooth, is_number

from mathModule import _fractionstr_to_float


@pytest.fixture
def x_y():
    x = [0, 1, 2, 4, 5]
    y = [1, 2, 3, 0, 3]
    return x, y


@pytest.fixture
def curve1(x_y):
    return Curve([x_y[0], x_y[1]], {"label": "test"})


def are_equal(series1, series2, tolabs=1e-5):
    assert (np.abs(series1 - series2) < tolabs).all(), f"not equal {series1}, {series2}"


def test_smooth(x_y):
    x, y = x_y
    y_init = np.array([1, 2, 3, 0, 3])
    assert (y == y_init).all()
    assert (smooth(y, window_len=1) == y_init).all()
    # NB: edges are treated with reflected copies at both sides
    are_equal(smooth(y, window_len=3, window="flat"), [5 / 3, 2, 5 / 3, 2, 2])
    are_equal(smooth(y, window_len=3, window="hanning"), [1, 2, 3, 0, 3])
    are_equal(
        smooth(y, window_len=3, window="hamming"),
        [1.13793103, 2.0, 2.72413793, 0.4137931, 2.79310345],
    )
    are_equal(smooth(y, window_len=3, window="bartlett"), [1, 2, 3, 0, 3])
    are_equal(smooth(y, window_len=3, window="bartlett"), [1, 2, 3, 0, 3])

    # wind = ["flat", "hanning", "hamming", "bartlett", "blackman"]
    are_equal(smooth(y, window_len=5, window="flat"), [2.2, 1.6, 1.8, 2.2, 1.8])
    are_equal(smooth(y, window_len=5, window="hanning"), [1.5, 2.0, 2.0, 1.5, 2.25])
    are_equal(
        smooth(y, window_len=5, window="hamming"),
        [1.625, 1.92857143, 1.96428571, 1.625, 2.16964286],
    )
    are_equal(smooth(y, window_len=5, window="bartlett"), [1.5, 2.0, 2.0, 1.5, 2.25])
    are_equal(
        smooth(y, window_len=5, window="blackman"),
        [1.4047619, 2.0, 2.19047619, 1.21428571, 2.39285714],
    )

    try:
        smooth(y, window_len=6)
    except ValueError:
        pass  # expect issues if window len > number of points


def test_is_number():
    assert is_number(1)
    assert is_number(0)
    assert is_number(np.inf)

    assert is_number(np.nan)  # hu, is that really the expected behavior?

    assert not is_number("anc")
    assert not is_number(None)
    assert not is_number([1, 2])
    assert is_number(True)
    assert is_number(False)


def test_fractionstr_to_float():
    """Test function fractionToFloat"""
    to_test = {1: 1, 2.0: 2.0, "3.0": 3, "4": 4, "5/2": 2.5, "inf": np.inf}
    for input_, output in to_test.items():
        assert _fractionstr_to_float(input_) == output


@pytest.mark.skip("Not implemented")
def test_find_nearest():
    assert False


@pytest.mark.skip("Not implemented")
def test_trapz():
    assert False


@pytest.mark.skip("Not implemented")
def test_x_at_value():
    assert False


@pytest.mark.skip("Not implemented")
def test_derivative2nd():
    assert False


@pytest.mark.skip("Not implemented")
def test_derivative():
    assert False


@pytest.mark.skip("Not implemented")
def test_roundgraphlim():
    assert False


@pytest.mark.skip("Not implemented")
def test_round_significant_range():
    assert False
