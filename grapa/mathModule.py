# -*- coding: utf-8 -*-
"""Functions dealing with math

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import warnings
import logging

import numpy as np

logger = logging.getLogger(__name__)

SMOOTH_WINDOW = ["flat", "hanning", "hamming", "bartlett", "blackman"]


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def roundSignificant(series, n_digits):
    if isinstance(series, (int, float)):
        out = roundSignificant([series], n_digits)
        return out[0]
    try:
        out = [
            x
            if (x == 0 or np.isnan(x) or np.isinf(x))
            else np.round(x, int(n_digits - 1 - np.floor(np.log10(np.abs(x)))))
            for x in series
        ]
    except TypeError:
        msg = "roundSignificant TypeError, returned input ({}, {})."
        logger.error(msg.format(type(series), series), exc_info=True)
        out = series
    return np.array(out)


def roundSignificantRange(series, n_digits):
    n_digits_add = 0
    span = np.abs(series[1] - series[0])
    refs = np.abs([span, (series[1] + series[0]) / 2, series[0], series[1]])
    try:
        with warnings.catch_warnings():  # don't want warnings div by 0, etc.
            warnings.simplefilter("ignore")
            n_digits_add = int(np.max([0, np.log10(np.max(refs) / span)]))
    except Exception:
        pass
    return roundSignificant(series, n_digits + n_digits_add)


def roundgraphlim(lim):
    """Returns rounded graph limits. lim=[1.5,3.0] for example."""
    lim = np.array(lim)
    lim.sort()
    diff = lim[1] - lim[0]
    target = np.array([lim[0] - diff / 10, lim[1] + diff / 10])
    magn = int(max(1 + np.ceil(-np.log10(np.abs(target)))))
    #    print ('roundgraphlim', lim, target, magn)
    target[0] = np.floor(target[0] * 10**magn) / 10**magn
    target[1] = np.ceil(target[1] * 10**magn) / 10**magn
    #    print ('   ',target)
    return target


def derivative(x, y):
    """numerical derivative dy/dx for the x and y datapoint series.
    Symetrical differentialtion i+1, i+1
    x is supposed to be monotonic (sorted up/down) !
    """
    if not isinstance(x, (np.ndarray,)):
        x = np.array(x)
    if not isinstance(y, (np.ndarray,)):
        y = np.array(y)
    if len(x) != len(y):
        msg = "derivative: x and y not same length! ({}, {})"
        logger.error(msg.format(x, y))
        return False
    if len(y) == 0:
        return np.array([])
    if len(y) == 1:
        return np.array([0])
    d = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    return np.append(np.append([d[0]], 0.5 * (d[:-1] + d[1:])), d[-1])


def derivative2nd(x, y):
    """
    Second numerical derivative d2y/dx2 for the x and y datapoint series.

    x is supposed to be monotonic (sorted up/down) !

    CAUTION: this function is locally more exact, but much more noisier than
    derivative(derivative)! Consider derivating twice on experiemental datasets.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if len(x) != len(y):
        msg = "derivative2nd: x and y not same length! ({}, {})"
        logger.error(msg.format(x, y))
        return False
    d = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    dd = (d[1:] - d[:-1]) / (x[2:] - x[:-2]) * 2
    return np.append(
        np.append((d[1] - d[0]) / (x[0] - x[1]), dd), (d[-1] - d[-2]) / (x[-1] - x[-2])
    )


def smooth(x, window_len=11, window="hanning"):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    :param x: the input signal
    :param window_len: the dimension of the smoothing window; should be an odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett',
           'blackman'. flat window will produce a moving average smoothing.
    :return: the smoothed signal as np.array

    Example: ::

        t = linspace(-2, 2, 0.1)
        x = sin(t) + randn(len(t)) * 0.1
        y = smooth(x)

    See also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve,
    scipy.signal.lfilter

    TODO: check the following was done: length(output) != length(input)
    return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not is_number(window_len) or window_len < 1:
        msg = (
            "smooth: cannot interpret window_len value (got {}, request int "
            "larger than 0). Set 1."
        )
        logger.warning(msg.format(window_len))
        window_len = 1
    window_len = int(window_len)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in SMOOTH_WINDOW:
        raise ValueError("Window is not of {}".format(SMOOTH_WINDOW))

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[int((window_len - 1) / 2) : int(len(y) - (window_len - 1) / 2)]


def xAtValue(x_series, y_series, value, xmin_xmax=[-np.inf, np.inf], silent=False):
    x = np.nan
    # find x value at intercept
    if len(x_series) - len(y_series) != 0:
        msg = "xAtValues: x_series and y_series not same lenght! Return 0. ({}, {})"
        logger.error(msg.format(len(x_series), len(y_series)))
        return 0

    idxBot = -1
    idxTop = -1
    limits = 2 - (x_series > min(xmin_xmax)) - (x_series < max(xmin_xmax))
    for i in range(len(x_series)):
        if limits[i]:
            continue
        x = x_series[i]
        y = y_series[i]
        #        print (i, x, y, idxBot, idxTop)
        if y <= value:
            if idxBot < 0 or np.abs(y - value) < np.abs(y_series[idxBot] - value):
                idxBot = i

        else:
            if idxTop < 0 or np.abs(y - value) < np.abs(y_series[idxTop] - value):
                idxTop = i

    if np.abs(idxBot - idxTop) != 1:
        if not silent:
            print(
                "Error function xAtValues: possibly 2 crossing values, or",
                "data not sorted.",
                idxBot,
                idxTop,
                value,
            )
            print(y_series)

    if idxBot + idxTop == -2:
        logger.error("xAtValues: no suitable data found.")
    elif idxBot != -1 and idxTop != -1:
        # linear interpolation
        x = x_series[idxBot] + (x_series[idxTop] - x_series[idxBot]) * (
            value - y_series[idxBot]
        ) / (y_series[idxTop] - y_series[idxBot])
    elif idxBot != -1:
        x = x_series[idxBot]
    elif idxTop != -1:
        x = x_series[idxTop]
    if np.isnan(x):
        msg = "xAtValue: cannot find. Value: {}\n  x_series: {}\n  y_series: {}"
        logger.error(msg.format(value, x_series, y_series))
    return x


def trapz(y, x, *args):
    """At some point numpy changed the function name for its trapz function"""
    if hasattr(np, "trapz"):
        return np.trapz(y, x, *args)
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, *args)
    return RuntimeError("Cannot find trapz nor trapezoid in numpy")


def _fractionstr_to_float(frac_str):
    """Returns a numeric value for floats, or fractions in the form '2/5'"""
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split("/")
        except ValueError:
            msg = "_fractionstr_to_float: cannot make sense of input, return nan: {}."
            logger.warning(msg.format(frac_str))  # maybe just or logger.warning ?
            return np.nan
        return float(num) / float(denom)


class MathOperator:  # pylint: disable=too-few-public-methods
    """To help write nicer code when dealing with math operations on Curves"""

    ADD = "add"
    SUB = "subtract"
    MUL = "multiply"
    DIV = "divide"
    POW = "power"

    _operations = {
        ADD: lambda x, y: x + y,
        SUB: lambda x, y: x - y,
        MUL: lambda x, y: x * y,
        DIV: lambda x, y: x / y,
        POW: lambda x, y: x**y,
    }

    @classmethod
    def operate(cls, xseries, yseries, operator: str) -> np.ndarray:
        """
        :param xseries: list or np.array
        :param yseries: list or np.array
        :param operator: one of the class constant ADD, SUB, MUL, DIV or POW.
        :return: np.array
        """
        if not isinstance(xseries, np.ndarray):
            xseries = np.array(xseries)
        if not isinstance(yseries, np.ndarray):
            yseries = np.array(yseries)
        if operator in cls._operations:
            return cls._operations[operator](xseries, yseries)
        msg = "MathOperator.operate: unexpected operator ({}), performs '+'."
        logger.warning(msg.format(operator))
        return cls._operations[cls.ADD](xseries, yseries)
