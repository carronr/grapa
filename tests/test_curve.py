"""
Tests for the content of module curve.py
"""
import os
import pytest

from . import grapa_folder, HiddenPrints
from grapa.curve import Curve
from grapa.graph import Graph



@pytest.fixture
def curve1():
    return Curve([[0, 2], [1, 2]], {"label": "yes"})


@pytest.fixture
def curve2():
    return Curve([[1, 3], [3, 4]], {"linespec": "--"})


def test_math_operation_curves(curve1, curve2):
    def validateusual(curve):
        assert list(curve.x()) == [1, 2]
        assert curve.get_attributes() == {"label": "yes", "linespec": "--"}

    out = curve1 + curve2
    assert list(out.y()) == [4.5, 5.5]
    validateusual(out)
    out = curve2 + curve1
    assert list(out.y()) == [4.5, 5.5]
    validateusual(out)
    out = curve1 - curve2
    assert list(out.y()) == [-1.5, -1.5]
    validateusual(out)
    out = curve1 * curve2
    assert list(out.y()) == [1.5 * 3, 2 * 3.5]
    validateusual(out)
    out = curve1 / curve2
    assert list(out.y()) == [0.5, 2 / 3.5]
    validateusual(out)
    out = curve1**curve2
    assert list(out.y()) == [1.5**3, 2**3.5]
    validateusual(out)


def test_math_operation_scalar(curve1):
    def validatesame(curve):
        assert list(curve.x()) == [0, 2]
        assert curve.get_attributes() == {"label": "yes"}

    out = curve1 + 1.5
    assert list(out.y()) == [2.5, 3.5]
    validatesame(out)
    out = 0.5 + curve1
    assert list(out.y()) == [1.5, 2.5]
    validatesame(out)
    out = curve1 - 1.5
    assert list(out.y()) == [-0.5, 0.5]
    validatesame(out)
    out = 0.5 - curve1
    assert list(out.y()) == [-0.5, -1.5]
    validatesame(out)
    out = curve1 * 1.5
    assert list(out.y()) == [1.5, 3]
    validatesame(out)
    out = 0.5 * curve1
    assert list(out.y()) == [0.5, 1]
    validatesame(out)
    out = curve1 / 2.0
    assert list(out.y()) == [0.5, 1]
    validatesame(out)
    out = 0.5 / curve1
    assert list(out.y()) == [0.5, 0.25]
    validatesame(out)
    out = curve1**3
    assert list(out.y()) == [1, 8]
    validatesame(out)


def test_label_auto(grapa_folder):
    """Test function auto_label"""
    graph = Graph(os.path.join(grapa_folder, "examples", "EQE", "SAMPLE_A_d1_1.sr"))
    assert len(graph) == 1
    assert graph[0].attr("label") == "SAMPLE A d1"
    graph[0].label_auto("${sample}")
    assert graph[0].attr("label") == "SAMPLE A"
    graph[0].label_auto("${sample} ${_simselement}")
    assert graph[0].attr("label") == "SAMPLE A"
    graph[0].label_auto("A ${temperature [k]:.0f} K")  # not found -> empty
    assert graph[0].attr("label") == "A K"
    graph[0].label_auto("${lockinsensitivity:.2f}ABC")
    assert graph[0].attr("label") == "16.00ABC"
    graph[0].label_auto("${lockinsensitivity:.1f} ${sample}")
    assert graph[0].attr("label") == "16.0 SAMPLE A"
    graph[0].label_auto("ABC")
    assert graph[0].attr("label") == "ABC"
    graph[0].label_auto("")
    assert graph[0].attr("label") == ""


def test_label_auto_fail(grapa_folder):
    """Test function auto_label"""
    graph = Graph(os.path.join(grapa_folder, "examples", "EQE", "SAMPLE_A_d1_1.sr"))
    with HiddenPrints():
        graph[0].label_auto("ABC ${cell:.1f} ${sample}")  # fails to format, should print
    assert graph[0].attr("label") == "ABC d1 SAMPLE A"


@pytest.mark.skip("Not implemented math_operation_interpolate_offsets")
def test_math_operation_interpolate_offsets():
    raise NotImplementedError
    # interpolate 0, offset False
    # interpolate 0, offset True
    # interpolate 1, offset False
    # interpolate 1, offset True
    # interpolate 2, offset False
    # interpolate 2, offset True


@pytest.mark.skip("Not implemented math_operation_interpolate_offsets")
def test_math_operation_interpolate_offsets():
    raise NotImplementedError


@pytest.mark.skip("Not implemented update_curve_values_dictkeys")
def test_update_curve_values_dictkeys():
    """test function update_curve_values_dictkeys"""
    raise NotImplementedError


@pytest.mark.skip("Not implemented update_graph_values_dictkeys_conditional")
def test_update_graph_values_dictkeys_conditional():
    """test function update_update_graph_values_dictkeys_conditional"""
    raise NotImplementedError


@pytest.mark.skip("Not implemented ContainerMetadata")
def test_containermetadata():
    """Test class ContainerMetadata"""
    raise NotImplementedError


@pytest.mark.skip("Not implemented Curve")
def test_curve():
    """test class Curve"""
    raise NotImplementedError


@pytest.mark.skip("Not implemented plot_curve")
def test_plot_curve():
    """Tst function plot_curve"""
    raise NotImplementedError


def test_print_help(grapa_folder):
    graph = Graph(os.path.join(grapa_folder, "examples\EQE\SAMPLE_A_d1_1.sr"))
    curve = graph[0]
    curve.funcListGUI(graph=graph, graph_i=0)
    # curve.print_help()

    with HiddenPrints():
        graph = Graph(os.path.join(grapa_folder, r"examples\SIMS\SAMPLE_D_c_1.TXT"))
    curve = graph[0]
    curve.funcListGUI(graph=graph, graph_i=0)
    # curve.print_help()


    with HiddenPrints():
        graph = Graph(os.path.join(grapa_folder, r"examples\TRPL\003.dat"))
    curve = graph[0]
    curve.funcListGUI(graph=graph, graph_i=0)
    #urve.print_help()

    with HiddenPrints():
        graph = Graph(os.path.join(grapa_folder, r"examples\Cf\C-f_SAMPLE_f1_133K.txt"))
    curve = graph[0]
    curve.funcListGUI(graph=graph, graph_i=0)
    # curve.print_help()

    with HiddenPrints():
        graph = Graph(os.path.join(grapa_folder, r"examples\CV\C-V_SAMPLE_f1_T=143_K.txt"))
    curve = graph[0]
    curve.funcListGUI(graph=graph, graph_i=0)
    # curve.print_help()

    graph = Graph([[1,2,3], [4, 5, 6]])
    graph.castCurve("CurveXRD", 0, silentSuccess=True)
    curve = graph[0]
    curve.funcListGUI(graph=graph, graph_i=0)
    #curve.print_help()


