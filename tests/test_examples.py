# import pytest

from . import grapa_folder, HiddenPrints, open_files_in_subfolder

from grapa.curve import Curve
from grapa.graph import Graph
from grapa.curve_image import Curve_Image

from grapa.utils.string_manipulations import strToVar
from grapa.utils.funcgui import FuncGUI

from grapa.datatypes.curveCf import CurveCf
from grapa.datatypes.curveCV import CurveCV
from grapa.datatypes.curveEQE import CurveEQE
from grapa.datatypes.curveJV import CurveJV
from grapa.datatypes.curveJscVoc import CurveJscVoc
from grapa.datatypes.curveSIMS import CurveSIMS
from grapa.datatypes.curveSpectrum import CurveSpectrum
from grapa.datatypes.curveTRPL import CurveTRPL
from grapa.datatypes.curveMCA import CurveMCA
from grapa.datatypes.curveMath import CurveMath


def check_graphlist(graphs, curvesubclass, lengraph=1, testtypeonlyfirst=False):
    """check that a list of Graphs, each contains 1 (or lengraph) curves, and that each
    Curve is of a given subclass"""
    funclistgui_counter = {}
    assert len(graphs) > 0
    for graph in graphs:
        assert len(graph) == lengraph

        for curve in graph:
            assert len(curve.x()) > 0

        if testtypeonlyfirst:
            assert isinstance(graph[0], curvesubclass)
        else:
            for curve in graph:
                assert isinstance(curve, curvesubclass)

        curvesubclass = type(graph[0])
        if curvesubclass not in funclistgui_counter:
            funclistgui_counter[curvesubclass] = 0
        if funclistgui_counter[curvesubclass] < 4:  # only check the first few graphs
            check_funclistgui(graph[0], graph, 0)
        funclistgui_counter[curvesubclass] += 1


def check_funclistgui(curve, graph, graphi):
    """Does not test much, mostly that the default GUI options can be executed with
    their default parameters, they do not raise exceptions, and that return False are
    understood.
    There is no check on their return value, or on the changes to the Curves.
    These test verify that all Curve actions proposed by the GUI can be executed."""
    # list of method that deliver False under basic (not appropriate) test conditions.
    # - computeAbsorptance, computeAlpha: need both R% and T% curves
    # - appendReplaceCurveRatioGUISmt: fails if there is no ^71Ga+ Curve
    tolerate_False = [
        "computeAbsorptance",
        "computeAlpha",
        "appendReplaceCurveRatioGUISmt",
    ]

    # must do twice, otherwise following tests would fail
    do_twice = ["dataModifySwapNmEv"]

    for item in curve.funcListGUI(graph=graph, graph_i=graphi):
        totest = (
            item if isinstance(item, FuncGUI) else FuncGUI(None, None).init_legacy(item)
        )
        func = totest.func
        kwargs = totest.hiddenvars
        args = []
        for field in totest.fields:
            if field["widgetclass"] not in ["Frame", None]:
                if field["keyword"] is None:
                    args.append(field["value"])
                else:
                    kwargs[field["keyword"]] = field["value"]
        for i in range(len(args)):
            args[i] = strToVar(args[i]) if isinstance(args[i], str) else args[i]

        if not hasattr(func, "__self__"):  # eg GraphSIMS.something, CurveTRPL.integrate
            args = [graph] + args

        try:
            with HiddenPrints():
                res = func(*args, **kwargs)
                if len([t for t in do_twice if t in func.__name__]) > 0:
                    res = func(*args, **kwargs)
        except Exception as e:
            print(
                "EXCEPTION OCCURED",
                curve.attr("filename"),
                "\n",
                totest.textsave,
                func,
                args,
                kwargs,
            )
            print(totest.fields)
            raise e
            # no assert here, cannot catch much, at least make sure no exception
        if res is not None and not res:
            if len([to for to in tolerate_False if to in func.__name__]) == 0:
                print(
                    "NOT RESULT",
                    curve.attr("filename").replace("\\", "/").split("/")[-1],
                    func,
                    res,
                )
                print("   ", func, args, kwargs)
                _ = func(*args, **kwargs)  # for printing purpose
                assert False


def test_open_files_subplotsinsets(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/_subplots_insets", "*.*")
    assert len(graphs) == 6


def test_open_files_math(grapa_folder):
    graphs = [Graph(grapa_folder + "/examples/_subplots_insets/scatter.txt")]
    check_graphlist(graphs, CurveMath, lengraph=2, testtypeonlyfirst=True)


def test_open_files_boxplot(grapa_folder):
    out = open_files_in_subfolder(grapa_folder, "examples/boxplot", "*.*")
    assert len(out) == 4


def test_open_files_cf(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/Cf", "*.*")
    assert len(graphs) == 19
    check_graphlist(graphs, CurveCf)


def test_open_files_cv(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/CV", "*.*")
    assert len(graphs) == 19
    check_graphlist(graphs, CurveCV)


def test_open_files_eqe(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/EQE", "*.*")
    assert len(graphs) == 4
    check_graphlist(graphs, CurveEQE)


def test_open_files_hls(grapa_folder):
    with HiddenPrints():
        graphs = open_files_in_subfolder(grapa_folder, "examples/HLsoaking", "*.*")
    assert len(graphs) == 1
    check_graphlist(graphs, Curve, lengraph=9)
    graphs = open_files_in_subfolder(
        grapa_folder, "examples/HLsoaking/52_Oct1143", "*.*"
    )
    assert len(graphs) == 99
    check_graphlist(graphs, CurveJV)


def test_open_files_jscvoc(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/JscVoc", "*.*")
    assert len(graphs) == 1
    check_graphlist(graphs, CurveJscVoc, lengraph=2, testtypeonlyfirst=True)


def test_open_files_jv(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/JV/SAMPLE_A", "*.*")
    assert len(graphs) == 7
    graphs = [gr for gr in graphs if "area.txt" not in gr[0].attr("filename")]
    assert len(graphs) == 6
    check_graphlist(graphs, CurveJV)

    graphs = open_files_in_subfolder(
        grapa_folder, "examples/JV/SAMPLE_B_3layerMo", "*.*"
    )
    assert len(graphs) == 23
    graphs = [g for g in graphs if "SAMPLE_B_3LayerMo.txt" not in g[0].attr("filename")]
    graphs = [gr for gr in graphs if "_Param.txt" not in gr[0].attr("filename")]
    assert len(graphs) == 21
    check_graphlist(graphs, CurveJV)

    graphs = open_files_in_subfolder(
        grapa_folder, "examples/JV/SAMPLE_B_5layerMo", "*.*"
    )
    assert len(graphs) == 21
    graphs = [gr for gr in graphs if "area.txt" not in gr[0].attr("filename")]
    graphs = [gr for gr in graphs if "_Param.txt" not in gr[0].attr("filename")]
    assert len(graphs) == 19
    check_graphlist(graphs, CurveJV)

    graphs = open_files_in_subfolder(
        grapa_folder, "examples/JV/SAMPLE_C", "*.*"
    )
    assert len(graphs) == 19


def test_open_files_sims(grapa_folder):
    with HiddenPrints():
        graphs = open_files_in_subfolder(grapa_folder, "examples/SIMS", "*.*")
    assert len(graphs) == 2
    check_graphlist([graphs[0]], CurveSIMS, lengraph=35)
    check_graphlist([graphs[1]], CurveSIMS, lengraph=24)


def test_open_files_spectra(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/Spectra", "*.*")
    assert len(graphs) == 4
    check_graphlist(graphs, CurveSpectrum)


def test_open_files_tiv(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/TIV", "*.*")
    assert len(graphs) == 2
    check_graphlist(graphs, CurveJV)

    graphs = open_files_in_subfolder(grapa_folder, "examples/TIV/dark", "*.*")
    assert len(graphs) == 10
    check_graphlist(graphs, CurveJV)

    graphs = open_files_in_subfolder(grapa_folder, "examples/TIV/illum", "*.*")
    assert len(graphs) == 10
    check_graphlist(graphs, CurveJV)


def test_open_files_trpl(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/TRPL", "*.*")
    assert len(graphs) == 2
    check_graphlist(graphs, CurveTRPL)


def test_open_files_xps(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples/XPS", "*.*")
    assert len(graphs) == 2
    check_graphlist(graphs, Curve)


def test_open_files_xrf(grapa_folder):
    with HiddenPrints():
        graphs = open_files_in_subfolder(grapa_folder, "examples/XRF", "*.*")
    assert len(graphs) == 2
    graphstest = [gr for gr in graphs if gr[0].attr("filename").endswith(".mca")]
    check_graphlist(graphstest, CurveMCA)
    graphstest = [gr for gr in graphs if gr[0].attr("filename").endswith(".html")]
    check_graphlist(graphstest, Curve)


def test_open_files_examples(grapa_folder):
    graphs = open_files_in_subfolder(grapa_folder, "examples", "*.txt")
    assert len(graphs) == 7

    graphstest = [
        gr for gr in graphs if gr[0].attr("filename").endswith("_imshow_picture.txt")
    ]
    assert len(graphstest) == 1
    check_graphlist(graphstest, Curve_Image, testtypeonlyfirst=True, lengraph=3)

    graphstest = [
        gr for gr in graphs if gr[0].attr("filename").endswith("imshow_datafile.txt")
    ]
    assert len(graphstest) == 1
    check_graphlist(graphstest, Curve_Image)

    graphstest = [
        gr for gr in graphs if gr[0].attr("filename").endswith("example_datafile.txt")
    ]
    assert len(graphstest) == 1
    check_graphlist(graphstest, Curve_Image, testtypeonlyfirst=True, lengraph=40)
