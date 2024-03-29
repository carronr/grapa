# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:19:42 2015

@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""
import numpy as np
import ast
import warnings
import codecs


MOJIBAKE_WINDOWS = [u'\xc2', u'\xc3']
ESCAPE_WARNING = {'a': '\a', 'b': '\b', 'r': '\r', 'f': '\f'}

"""
ESCAPEWARNING = ['\a', '\b', '\f', '\r', '\v']  # likely errors
# likely intent was latex code, eg. \nu instead of \n+u
ESCAPEWARNING += ['\nu', '\nabla', '\nexists', '\neg', '\neq', '\notin']
ESCAPEWARNING += ['\tau', '\theta', '\triangle', '\times']
# '\newline': likely not an error
# \\ ooo, \\ xhh: checks not implemented
"""


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def roundSignificant(xSeries, nDigits):
    if isinstance(xSeries, (int, float)):
        out = roundSignificant([xSeries], nDigits)
        return out[0]
    try:
        out = [x if (x == 0 or np.isnan(x) or np.isinf(x)) else np.round(x, int(nDigits-1-np.floor(np.log10(np.abs(x))))) for x in xSeries]
    except TypeError as e:
        print('TypeError in roundSignificant, returned input.')
        print(e)
        print(type(xSeries), xSeries)
        out = xSeries
    return np.array(out)


def roundSignificantRange(xSeries, nDigits):
    nDigitsAdd = 0
    span = np.abs(xSeries[1] - xSeries[0])
    refs = np.abs([span, (xSeries[1] + xSeries[0])/2, xSeries[0], xSeries[1]])
    try:
        with warnings.catch_warnings():  # don't want warnings div by 0, etc.
            warnings.simplefilter("ignore")
            nDigitsAdd = int(np.max([0, np.log10(np.max(refs)/span)]))
    except Exception:
        pass
    return roundSignificant(xSeries, nDigits+nDigitsAdd)


def roundgraphlim(lim):
    """ Returns rounded graph limits. lim=[1.5,3.0] for example. """
    lim = np.array(lim)
    lim.sort()
    diff = lim[1] - lim[0]
    target = np.array([lim[0] - diff/10, lim[1] + diff/10])
    magn = int(max(1 + np.ceil(-np.log10(np.abs(target)))))
#    print ('roundgraphlim', lim, target, magn)
    target[0] = np.floor(target[0] * 10**magn) / 10**magn
    target[1] = np.ceil(target[1] * 10**magn) / 10**magn
#    print ('   ',target)
    return target


def stringToVariable(val):
    print('WARNING: stringToVariable, use strToVar instead (arg', val, ')')
    return strToVar(val)


def strToVar(val):
    # val = codecs.getdecoder("unicode_escape")(val)[0]
    flagast = False
    try:
        val = float(val)
    except Exception:
        try:
            val = ast.literal_eval(val)
            flagast = True
        except Exception:
            if isinstance(val, str) and len(val) > 1:
                if ((val[0] == '[' and val[-1] == ']')
                        or (val[0] == '(' and val[-1] == ')')):
                    try:
                        val = [float(v) for v in val[1:-1].replace(' ', '').replace('np.inf', 'inf').split(',')]
                    except Exception:
                        pass
    # print('strToVar', val, flagast)
    if not flagast:  # not needed when created through ast.literal_eval
        val = strUnescapeIter(val)
    # print('   ', val)
    return val


def strUnescapeIter(var):
    if isinstance(var, list):
        for i in range(len(var)):
            var[i] = strUnescapeIter(var[i])
    elif isinstance(var, dict):
        for key in var:
            var[key] = strUnescapeIter(var[key])
    elif isinstance(var, str):

        try:
            var = codecs.getdecoder("unicode_escape")(var)[0]
        except Exception as e:
            print('Exception in function strToVar', type(e), e)
            print('Input:', var)
        try:
            # handling of special characters transformed into mojibake
            # The 2-pass code below can clean inputs with special characters
            # encoded in:
            # ANSI (e.g. Windows-1252), UTF-8, UTF-8 escaped (eg. \xb5 for mu)
            # First pass
            for char in MOJIBAKE_WINDOWS:
                if char in var:
                    # print('Suspicion mojibake, latin-1-utf-8 conversion')
                    var = var.encode('latin-1').decode('utf-8')
                    break
            # Second pass. Appears necessary with some charset input
            for char in MOJIBAKE_WINDOWS:
                if char in var:
                    # print('Suspicion mojibake (2), latin-1-utf-8 conversion')
                    var = var.encode('latin-1').decode('utf-8')
                    break
            for key in ESCAPE_WARNING:
                if ESCAPE_WARNING[key] in var:
                    print('CAUTION, escape character detected (\\' + key
                          + '). Please add double backslash (\\\\' + key + ')'
                          + ' for Latex code.')
        except Exception as e:
            print('Exception in function strToVar (mojibake)', type(e), e)
            print('Possibly, mix of special characters and escape sequences',
                  'in same input')
            print('Input:', var)
            # keep current out value. Likely to fail
    return var


def varToStr(val):
    # return codecs.getencoder("unicode_escape")(out)[0].decode('utf-8')
    try:
        out = repr(val).strip("'")
    except Exception as e:
        print('varToStr Exception', type(e), e)
        print('Input:', val)
        out = ''
    return out


def listToString(val):
    """ Converts a list to a string with suitable formatting """
    return '[' + ', '.join([str(element) if not isinstance(element, str) else '\''+element+'\'' for element in val]) + ']'


def derivative(x, y):
    """ numerical derivative dy/dx for the x and y datapoint series.
    Symetrical differentialtion i+1, i+1
    x is supposed to be monotonic (sorted up/down) !
    """
    if not isinstance(x, (np.ndarray)):
        x = np.array(x)
    if not isinstance(y, (np.ndarray)):
        y = np.array(y)
    if len(x) != len(y):
        print('ERROR Math derivative: x and y not same length!')
        print(x, y)
        return False
    if len(y) == 0:
        return np.array([])
    if len(y) == 1:
        return np.array([0])
    d = (y[1:]-y[:-1])/(x[1:]-x[:-1])
    return np.append(np.append([d[0]], 0.5*(d[:-1]+d[1:])), d[-1])


def derivative2nd(x, y):
    """
    Second numerical derivative d2y/dx2 for the x and y datapoint series.
    x is supposed to be monotonic (sorted up/down) !
    CAUTION: this function is locally more exact, but much more noisier than
        derivative(derivative)!
    Consider derivating twice on experiemental datasets.
    """
    if not isinstance(x, (np.ndarray)):
        x = np.array(x)
    if not isinstance(y, (np.ndarray)):
        y = np.array(y)
    if len(x) != len(y):
        print('ERROR Math derivative: x and y not same length!')
        print(x, y)
        return False
    d = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    dd = (d[1:] - d[:-1]) / (x[2:] - x[:-2]) * 2
    return np.append(np.append((d[1]-d[0])/(x[0]-x[1]), dd), (d[-1]-d[-2])/(x[-1]-x[-2]))


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd
            integer
        window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    Example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    See also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve, scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead
        of a string
    NOTE: length(output) != length(input), to correct this:
        return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming',",
                         " 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int((window_len-1)/2):int(len(y)-(window_len-1)/2)]


def xAtValue(xSeries, ySeries, value, xMinMax=[-np.inf, np.inf], silent=False):
    x = np.nan
    # find x value at intercept
    if len(xSeries) - len(ySeries) != 0:
        print('Error function xAtValues: lenght of xSeries and ySeries not',
              'identical!')
        print('   len(xSeries): ', len(xSeries))
        print('   len(ySeries): ', len(ySeries))
        return 0

    idxBot = -1
    idxTop = -1
    limits = 2 - (xSeries > min(xMinMax)) - (xSeries < max(xMinMax))
    for i in range(len(xSeries)):
        if limits[i]:
            continue
        x = xSeries[i]
        y = ySeries[i]
#        print (i, x, y, idxBot, idxTop)
        if y <= value:
            if idxBot < 0 or np.abs(y - value) < np.abs(ySeries[idxBot]-value):
                idxBot = i

        else:
            if idxTop < 0 or np.abs(y - value) < np.abs(ySeries[idxTop]-value):
                idxTop = i

    if np.abs(idxBot - idxTop) != 1:
        if not silent:
            print('Error function xAtValues: possibly 2 crossing values, or',
                  'data not sorted.', idxBot, idxTop, value)
            print(ySeries)

    if idxBot + idxTop == -2:
        print('Error function xAtValues: no suitable data found.')
    elif idxBot != -1 and idxTop != -1:
        # linear interpolation
        x = xSeries[idxBot] + (xSeries[idxTop] - xSeries[idxBot]) * (value - ySeries[idxBot]) / (ySeries[idxTop] - ySeries[idxBot])
    elif idxBot != -1:
        x = xSeries[idxBot]
    elif idxTop != -1:
        x = xSeries[idxTop]
    if np.isnan(x):
        print('ERROR mathModule xAtValue: cannot find suitable value.')
        print('   xSeries', xSeries)
        print('   ySeries', ySeries)
        print('   value', value)
    return x


# def bandgapFromTauc(nm, EQE, xLim=[600, 1500], yLim=[25, 70]):
#     if max(EQE) < 1:
#         print('Function bandgapFromTauc: max(EQE) < 1. Multiplied by 100 for',
#               'datapoint selection.')
#         EQE *= 100
#
#     mask = np.ones(len(nm), dtype=bool)
#     for i in reversed(range(len(mask))):
#         if (EQE[i] < yLim[0] or EQE[i] > yLim[1] or nm[i] < xLim[0]
#                 or nm[i] > xLim[1]):
#             mask[i] = False
#     nm = nm[mask]
#     EQE = EQE[mask]
#
#     eV = 1239.5 / nm
#     tauc = (eV * EQE)**2
#     if len(tauc > 1):
#         z = np.polyfit(eV, tauc, 1, full=True)[0]
#         # p = np.poly1d(z)
#         bandgap = -z[1]/z[0]
#         return [bandgap, z[0]]
#     # print ('Function bandgapFromTauc: not enough suitable datapoints.')
#     return [np.nan, np.nan]


# def polynomFromDataSeries(xSeries, ySeries):
#     if len(xSeries) == 0 or len(ySeries) == 0:
#         print('Error polynomFromDataSeries: cannot find polynom with no data.',
#               'Length of series:', len(xSeries), len(ySeries))
#     if len(xSeries) != len(ySeries):
#         print('Error polynomFromDataSeries: cannot find polynom when length',
#               'of x and y series do not match. Length of series:',
#               len(xSeries), len(ySeries))
#     degree = len(xSeries)
#     print('function not written yet')
#     raise NotImplementedError
#     return -1
