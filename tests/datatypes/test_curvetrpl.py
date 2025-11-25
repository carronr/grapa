import numpy as np
import pytest
import warnings

from .. import HiddenPrints, Graph, GrapaWarning


from grapa.datatypes.curveTRPL import CurveTRPL, find_onset, _roi_to_mask


@pytest.fixture
def curve1():
    """a minimalist CurveTRPL"""
    y = [1, 0, 1, 0, 1, 10, 5, 3, 2, 1.5, 1, 1, 0, 1]
    x = np.arange(len(y)) * 0.1
    return CurveTRPL([x, y], {})


@pytest.fixture
def curve2():
    """a CurveTRPL with noise, to test fits"""
    target = [0, 1, 5, 0.5, 20]
    t = np.arange(100, dtype=float) - 10
    y = t * 0.0 + target[0]
    y += target[1] * np.exp(-t / target[2])
    y += target[3] * np.exp(-t / target[4])
    y += 0.1 * np.sin(t * 2)  # add simili-noise
    y[t < 0] = 0
    return CurveTRPL([t, y], {"target": target})


@pytest.fixture
def curvefit():
    popt = [0, 1, 5, 0.5, 20]
    t = np.arange(100, dtype=float)
    out = CurveTRPL([t, t], {"_popt": popt, "_fitfunc": "func_fitExp"})
    out.updateFitParam(*popt)
    return out


@pytest.fixture
def graph(curve1, curve2, curvefit):
    graph = Graph()
    graph.append(curve1)
    graph.append(curve2)
    graph.append(curvefit)
    return graph


def test_funcListGUI(graph):
    # About good enough to check the code can actually execute.
    # The sensible choice of default values may be best evaluated by users
    try:
        for c in range(len(graph)):
            out = graph[c].funcListGUI(graph=graph, graph_i=c)
            assert isinstance(out, list), "output should be a list"
    except Exception as e:
        raise AssertionError("an Exception occured {} {}".format(type(e), e))


def test_printHelp(curve1):
    # only about good enough that the code can execute
    with warnings.catch_warnings():
        # because curve1 not part of a graph and FunclistGui not called beforehand,
        # not the help may not be as complete as if embedded into a graph
        warnings.simplefilter("ignore", category=GrapaWarning)
        # with HiddenPrints():
        assert curve1.print_help()


def test_alterListGUI(graph):
    # About good enough that the code can execute
    for curve in graph:
        out = curve.alterListGUI()
        assert isinstance(out, list)


def test_add_getOffset(curve1):
    offset = 123.4
    y = list(curve1.y())
    curve1.addOffset(offset)
    diff = curve1.y() - y - offset
    assert np.isclose(diff, 0).all()
    assert np.isclose(
        curve1.getOffset(), offset
    ), "offset value not correctly recorded in curve metadata"
    assert not curve1.addOffset("a"), "input should fail with non-numeric"


def test_find_onset(curve1):
    assert 0.4 <= find_onset(curve1.x(), curve1.y()) <= 0.5
    # assert 0


def test_add_getXoffset(curve1):
    curve1.addXOffset(-curve1.findOnset())
    # not good test in case rise time covers several datapoints, to release somehow
    x = curve1.x()
    assert 0 <= x[np.argmax(curve1.y())] <= (x[1] - x[0])
    # assert False  # SHOULD BE SKIPPED, BUT IS NOT???


def test_roi_to_mask(curve1):
    mask, xmasked = _roi_to_mask(curve1.x(), [0.3, 0.6])
    assert (curve1.x()[mask] == xmasked).all()
    assert (curve1.y()[mask] == [0, 1, 10]).all()
    mask, xmasked = _roi_to_mask(curve1.x(), None)
    assert (curve1.x() == xmasked).all()


def test_normalize_revert(curve1):
    y = np.array(list(curve1.y()))
    mask = ~np.isclose(y, 0)
    args = [1e6, 120, 100]
    offsets = [0, 10]
    for offset in offsets:
        curve1.setY(y)  # make sure correct starting point
        factor = 1 / (args[0] * args[1] * args[2] * 1e-12)  # last in ps versus s
        curve1.addOffset(offset)
        curve1.normalize(*args)
        assert np.isclose(curve1.y()[mask], (y[mask] + offset) * factor).all()
        assert np.isclose(curve1.attr("_TRPLFactor"), factor)
        assert curve1.attr("_units")[1] == "cts/Hz/s/s"
        assert np.isclose(curve1.attr("_TRPLOffset"), offset * factor)
        curve1.normalizerevert()
        assert np.isclose(curve1.y()[mask], y[mask] + offset).all()
        assert curve1.attr("_unit") == ""


def test_fitexp(curve2):
    target = curve2.attr("target")
    kwargs = {
        "nbExp": int((len(target) - 1) / 2),
        "ROI": [0, np.max(curve2.x())],
        "fixed": [0, "", ""],
    }
    with HiddenPrints():
        out = curve2.CurveTRPL_fitExp(**kwargs)
    popt = out.attr("_popt")
    for i in range(len(target)):
        errorrel = np.abs(popt[i] - target[i]) / max(1e-300, target[i])
        assert errorrel < 0.1 or np.isclose(popt[i], target[i])


def test_CurveTRPL_sequence_fitexp(curve2):
    target = curve2.attr("target")
    targettaumin = np.min(target[2::2])
    targettaumax = np.max(target[2::2])
    targetsuma = np.sum(target[1::2])
    kwargs = {
        "nb_exp": 1,
        "roi": [1, np.max(curve2.x())],
        "roi0": [1, np.max(curve2.x()) / 8],
        "multiplier": 4,
        "maxpoints": 50,
    }
    out = curve2.CurveTRPL_sequence_fitexp(**kwargs)
    assert len(out) == 3  # by construction above, 1/8 * 1 * 4 * 4
    for curve in out:
        popt = curve.attr("_popt")
        assert np.isclose(popt[0], 0)
        assert 0 < popt[1] <= targetsuma
        assert targettaumin <= popt[2] <= targettaumax


# maybe make this depend on testing function spline()
def test_CurveTRPL_spline(curve2):
    try:
        with HiddenPrints():
            out = curve2.CurveTRPL_spline(roi=[1, np.max(curve2.x())])
    except Exception:
        raise AssertionError("Exception during CurveTRPL_spline")
    # at least the script runs
    assert len(out.x()) > 0
    diffy = []
    for i in range(len(out.x())):
        argclosest = np.argmin(np.abs(curve2.x() - out.x()[i]))
        diffy.append(curve2.y()[argclosest] - out.y()[i])
    # for i in range(len(out.x())):
    #    print(out.x()[i], out.y()[i], diffy[i])
    assert np.sum(np.abs(diffy)) < 0.1 * len(diffy)
    # Not quite sure how to automate tests further.
    # Honestly would recommend to evaluate visually the quality of the spline fit


def test_differential_lifetime(curvefit):
    try:
        out = curvefit.Curve_differential_lifetime()
    except Exception:
        raise AssertionError
    assert (out.x() == curvefit.x()).all()
    y = out.y()
    assert np.min(y) >= np.min(curvefit.attr("_popt")[2::2])
    assert np.max(y) <= np.max(curvefit.attr("_popt")[2::2])
    assert ((y[1:] - y[:-1]) > 0).all()  # differential lifetime expected to increase


def test_differential_lifetime_vs_signal(curvefit):
    try:
        out = curvefit.Curve_differential_lifetime_vs_signal()
    except Exception:
        raise AssertionError
    assert (out.x() == curvefit.y()).all()
    y = out.y()
    assert np.min(y) >= np.min(curvefit.attr("_popt")[2::2])
    assert np.max(y) <= np.max(curvefit.attr("_popt")[2::2])
    assert ((y[1:] - y[:-1]) > 0).all()  # differential lifetime expected to increase


def test_fit_resampleX(curvefit):
    def test_with(curve, x, y, spacing):
        # by construction of test, provide initial x, y, please make sure to not
        # request generating outside the initial range
        curve.fit_resampleX(spacing)
        try:
            _ = (e for e in spacing)
        except TypeError:  # not iterable
            expected = np.arange(
                np.min(curve.x()), np.max(curve.x()) + spacing, spacing
            )
            assert (
                len(expected) == len(curve.x()) and (expected == curve.x()).all()
            ), "not the one expected"
            assert len(curve.x()) > 1, "len not 1"
        else:  # is an iterable
            if len(spacing) == 3:
                assert (np.arange(*spacing) == curve.x()).all(), "list3elementstoarange"
            else:
                assert (spacing == curve.x()).all(), "list-like comparison"
        assert np.min(y) <= np.min(curve.y()), "min too low"
        assert np.max(y) >= np.max(curve.y()), "max too high"

    x = list(curvefit.x())
    y = list(curvefit.y())
    xmin, xmax = np.min(x), np.max(x)
    test_with(curvefit, x, y, list(np.linspace(xmin, xmax, 10)))
    test_with(curvefit, x, y, np.linspace(xmin, xmax, 20))
    test_with(curvefit, x, y, np.arange(xmin, xmax, (xmax - xmin) / 5))
    test_with(curvefit, x, y, [0, 40, 0.1])
    test_with(curvefit, x, y, [0, 2, 10, 30])
    test_with(curvefit, x, y, 5.0)


def test_CurveTRPL_smoothBin(curve2):
    ymin, ymax = np.min(curve2.y()), np.max(curve2.y())
    out = curve2.CurveTRPL_smoothBin(window_len=1, binning=1)
    assert (curve2.x() == out.x()).all()
    assert (curve2.y() == out.y()).all()
    out = curve2.CurveTRPL_smoothBin(window_len=5, binning=1)
    assert (curve2.x() == out.x()).all()
    assert np.min(out.y()) >= ymin
    assert np.max(out.y()) <= ymax
    out = curve2.CurveTRPL_smoothBin(window_len=1, binning=4)
    assert np.abs(len(out.x()) - int(len(curve2.x()) / 4)) <= 1
    assert np.min(out.y()) >= ymin
    assert np.max(out.y()) <= ymax
    out = curve2.CurveTRPL_smoothBin(window_len=10, binning=8)
    assert np.abs(len(out.x()) - int(len(curve2.x()) / 8)) <= 1
    assert np.min(out.y()) >= ymin
    assert np.max(out.y()) <= ymax
    dx = 5
    out = curve2.CurveTRPL_smoothBin(window_len=9, window="flat", binning=1)
    assert np.isclose(out.y()[4 + dx], np.average(curve2.y()[dx : 9 + dx]))


def test_integrate(curve2):
    x = curve2.x()
    y = curve2.y()

    def integr(series):  # corresponds to trapz with uniformly spaced data
        return np.sum(series[1:-1]) + 0.5 * (series[0] + series[-1])

    out = curve2.integrate()
    res = integr(y)
    assert np.isclose(out, res), "sum not match {}, {}".format(out, res)

    roi = [10, 20]
    out = curve2.integrate(ROI=roi)
    ys = [y[i] for i in range(len(x)) if roi[0] <= x[i] <= roi[1]]
    res = integr(ys)
    assert np.isclose(out, res), "sum not match {}, {}".format(out, res)

    curve = 0 - curve2
    out = curve.integrate()
    res = -integr(y)
    # print("sum curve2.integrate", out, "-integry", res)
    assert np.isclose(out, res), "sum not match {}, {}".format(out, res)

    with warnings.catch_warnings():  # y contains 0, A RuntimeWarning is expected
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = curve2.integrate(alter=["", "log10abs"])
        res = integr(np.log10(np.abs(y)))
    assert np.isclose(out, res), "sum not match {}, {}".format(out, res)


def test_fitparams_weightedaverage(curvefit):
    p = curvefit.attr("_popt")
    res = [
        (p[1] * p[2] + p[3] * p[4]) / (p[1] + p[3]),
        (p[1] * p[2] ** 2 + p[3] * p[4] ** 2) / (p[1] * p[2] + p[3] * p[4]),
    ]
    with HiddenPrints():
        out = curvefit.fitparams_weightedaverage()
    assert list(out) == res


def test_printhelp(graph):
    graph[2].funcListGUI(graph=graph, graph_i=0)
    with HiddenPrints():
        graph[2].print_help()

# NOT TESTED
# class CurveTRPL(Curve):
#     def fitparams_to_clipboard(self):
