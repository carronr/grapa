import os
import pytest

import numpy as np

from . import grapa_folder, Graph, HiddenPrints
from grapa.graph import ConditionalPropertyApplier
from grapa.utils.string_manipulations import TextHandler
from grapa.curve import Curve


def are_graphs_same(graph1, graph2):
    attr1, attr2 = graph1.get_attributes(), graph2.get_attributes()
    assert len(attr1.keys()) == len(attr2.keys())
    for key in attr1:
        if key not in ["collabels"]:
            assert key in attr2
            assert attr1[key] == attr2[key]

    assert len(graph1) == len(graph2)
    for c, curve in enumerate(graph1):
        attr1 = graph1[c].get_attributes()
        attr2 = graph2[c].get_attributes()
        assert len(attr1.keys()) == len(attr2.keys())
        for key in attr1:
            if key in ["filename"]:
                assert os.path.abspath(attr1[key]) == os.path.abspath(attr2[key])
            else:
                assert key in attr2
                assert attr1[key] == attr2[key]

        assert np.array_equal(graph1[c].x(), graph2[c].x())
        assert np.array_equal(graph1[c].y(), graph2[c].y())
    return True


@pytest.fixture
def graph(grapa_folder):
    files = [
        "examples/CV/C-V_SAMPLE_f1_T=123_K.txt",
        "examples/CV/C-V_SAMPLE_f1_T=133_K.txt",
        "examples/CV/C-V_SAMPLE_f1_T=143_K.txt",
    ]
    out = Graph()
    for file in files:
        out.merge(Graph(os.path.join(grapa_folder, file)))
    yield out
    # after yield: code for teardown


def test_ConditionalPropertyApplier(graph):
    assert len(graph) == 3
    cpa = ConditionalPropertyApplier
    cpa.apply(graph, "label", "==", "SAMPLE f1 123 K", "color", "r")
    assert graph[0].attr("color") == "r"
    cpa.apply(graph, "label", "==", "SAMPLE f1 123 K ", "color", "b")
    assert graph[0].attr("color") == "r"
    cpa.apply(graph, "label", "!=", "SAMPLE f1 123 K", "color", "g")
    assert graph[0].attr("color") == "r"
    assert graph[1].attr("color") == "g"
    assert graph[2].attr("color") == "g"
    graph[0].update({"linewidth": 1})
    cpa.apply(graph, "label", ">", "SAMPLE f1 123 K", "color", "k")
    assert graph[0].attr("color") == "r"
    assert graph[1].attr("color") == "k"
    with HiddenPrints():
        cpa.apply(graph, "linewidth", "<", 2, "color", "c")
    assert graph[0].attr("color") == "c"
    assert graph[1].attr("color") == "k"
    cpa.apply(graph, "label", "startswith", "SAMPLE f1 123", "color", "y")
    assert graph[0].attr("color") == "y"
    assert graph[1].attr("color") == "k"
    cpa.apply(graph, "label", "endswith", "33 K", "color", "r")
    assert graph[0].attr("color") == "y"
    assert graph[1].attr("color") == "r"
    cpa.apply(graph, "label", "contains", "123", "color", "b")
    assert graph[0].attr("color") == "b"
    assert graph[1].attr("color") == "r"
    cpa.apply(graph, "label", "contains", " ", "color", "c")
    assert graph[0].attr("color") == "c"
    assert graph[1].attr("color") == "c"
    cpa.apply(graph, "label", "contains", "4", "color", "k")
    assert graph[0].attr("color") == "c"
    assert graph[1].attr("color") == "c"


def test_TextHandler_add_remove(graph):
    graph = Graph()
    text = "New Text"
    textxy = (0.6, 0.6)
    textargs = {"color": "red"}
    TextHandler.add(graph, text, textxy, textargs)
    assert graph.attr("text") == [text]
    assert graph.attr("textxy") == [textxy]
    assert graph.attr("textargs") == [textargs]

    TextHandler.add(graph, text, textxy, textargs)
    assert graph.attr("text") == [text, text]
    assert graph.attr("textxy") == [textxy, textxy]
    assert graph.attr("textargs") == [textargs, textargs]

    TextHandler.add(graph, text, textxy, textargs)
    graph.update({"text": [text, "abc", "def"]})
    TextHandler.remove(graph, 1)
    assert graph.attr("text") == [text, "def"]
    assert graph.attr("textxy") == [textxy, textxy]
    assert graph.attr("textargs") == [textargs, textargs]

    TextHandler.add(
        graph,
        ["Text1", "Text2"],
        [(0.1, 0.1), (0.2, 0.2)],
        [{"color": "blue"}, {"color": "green"}],
    )
    assert graph.attr("text") == [text, "def"] + ["Text1", "Text2"]

    with pytest.raises(IndexError):
        TextHandler.remove(graph, 10)
    TextHandler.remove(graph, 0)
    TextHandler.remove(graph, 0)
    TextHandler.remove(graph, 0)

    # tolerates faulty input: number, type
    TextHandler.add(graph, text, [textxy, 1], "z")
    assert graph.attr("text") == ["Text2", text]
    assert graph.attr("textxy") == [(0.2, 0.2), textxy]  # not 3rd argument
    assert graph.attr("textargs") == [{"color": "green"}, {}]  # not z


def test_TextHandler_repair1(graph):
    # screws up, try repair
    text = "abc"
    textxy = (0.6, 0.8)
    textargs = {"color": "red"}
    TextHandler.add(graph, text, textxy, textargs)
    TextHandler.add(graph, "def", textxy, textargs)

    graph.update({"textargs": ""})
    with HiddenPrints():
        TextHandler.check_valid(graph)
    assert graph.attr("text") == ["abc", "def"]
    assert graph.attr("textxy") == [textxy, textxy]
    assert graph.attr("textargs") == [{}, {}]

    graph.update({"textxy": ""})
    TextHandler.check_valid(graph)
    assert graph.attr("text") == ["abc", "def"]
    assert graph.attr("textxy") == ["", ""]
    assert graph.attr("textargs") == [{}, {}]

    graph.update({"text": ["ghi"]})
    TextHandler.check_valid(graph)
    assert graph.attr("text") == ["ghi"]
    assert graph.attr("textxy") == [""]
    assert graph.attr("textargs") == [{}]

    graph.update({"text": ""})
    TextHandler.check_valid(graph)
    assert graph.attr("text") == ""
    assert graph.attr("textxy") == ""
    assert graph.attr("textargs") == ""

    graph.update({"text": ""})
    TextHandler.check_valid(graph)
    graph.update({"text": ["a", "b", "c"]})
    with HiddenPrints():
        TextHandler.check_valid(graph)
    assert graph.attr("text") == ["a", "b", "c"]
    assert graph.attr("textxy") == ["", "", ""]
    assert graph.attr("textargs") == [{}, {}, {}]


def test_TextHandler_repair2(graph):
    graph.update({"text": ["a"], "textxy": "", "textargs": ""})
    with HiddenPrints():
        TextHandler.check_valid(graph)
    assert graph.attr("text") == ["a"]
    assert graph.attr("textxy") == [""]
    assert graph.attr("textargs") == [{}]

    graph.update({"text": ["a"], "textxy": "abc", "textargs": [{}, {}]})
    with HiddenPrints():
        TextHandler.check_valid(graph)
    assert graph.attr("text") == ["a"]
    assert graph.attr("textxy") == [""]
    assert graph.attr("textargs") == [{}]

    graph.update({"text": ["a"], "textxy": [0.1, 0.2], "textargs": [{}, {}]})
    with HiddenPrints():
        TextHandler.check_valid(graph)
    assert graph.attr("text") == ["a"]
    assert graph.attr("textxy") == [[0.1, 0.2]]
    assert graph.attr("textargs") == [{}]

    graph.update({"text": ["a"], "textxy": [[0.1, 0.2]], "textargs": [{}, {}]})
    TextHandler.check_valid(graph)
    assert graph.attr("text") == ["a"]
    assert graph.attr("textxy") == [[0.1, 0.2]]
    assert graph.attr("textargs") == [{}]

    graph.update({"text": ["a"], "textxy": [0.1, 0.2, 0.3], "textargs": ["abc", {}]})
    TextHandler.check_valid(graph)
    assert graph.attr("text") == ["a"]
    assert graph.attr("textxy") == [""]
    assert graph.attr("textargs") == [{}]
    # do not attempt to test all possible irregular formats, just a few
    # the goal is that grapa manages to send something to matplotlib. User can also rely
    # on matplotlib to catch the rest


def test_graph_creation_variable():
    graph = Graph([[1, 2, 3], [4, 5.0, 6.5]])
    assert len(graph) == 1
    assert (graph[0].x() == [1, 2, 3]).all()
    assert (graph[0].y() == [4, 5, 6.5]).all()


@pytest.mark.skip("Not yet implemented")
def test_Graph():
    raise NotImplementedError


@pytest.fixture
def filetest(grapa_folder):
    filename = os.path.join(os.path.split(__file__)[0], "TESTFILE.txt")
    yield filename
    # after yield: code for teardown
    if os.path.exists(filename):
        # print("REMOVE FILE", filename)
        os.remove(filename)


def test_graph_export0(filetest):
    graph = Graph()
    graph.export(filetest.replace(".txt", ""))
    with open(filetest, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
    target = ["label"]  # "meastype\tGraph": only from GUI
    assert target == content


def test_graph_export1(filetest):
    graph = Graph()
    graph.append(Curve([[1], [2]], {}))
    graph.export(filetest.replace(".txt", ""))
    with open(filetest, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
    target = ["label", "1\t2"]  # "meastype\tGraph": only from GUI
    assert target == content

    graph[0].update({"label": "abc"})
    graph.export(filetest.replace(".txt", ""))
    with open(filetest, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
    target = ["label\tabc", "1\t2"]  # "meastype\tGraph": only from GUI
    assert target == content


def test_graph_export2(filetest):
    fname = filetest.replace(".txt", "")
    graph = Graph()
    graph.append(Curve([[1], [2]], {"muloffset": [2, 3]}))
    graph.append(Curve([[1], [2]], {"label": "abc", "useless": "yes"}))

    graph.export(fname)
    with open(filetest, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
    target = ["label\t\tabc", "muloffset\t[2, 3]", "useless\t\tyes", "1\t2\t2"]
    assert content == target

    graph.export(fname, save_altered=True)
    with open(filetest, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
    target = ["label\t\t\t\tabc", "useless\t\t\t\tyes", "2\t6\t\t1\t2"]
    assert content == target

    graph.export(fname, if_template=True)
    with open(filetest, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
    target = ["label\t\t\t\tabc", "muloffset\t[2, 3]", "1\t0\t\t1\t0\t\t"]
    # hu why 2 additional \t at end?
    assert content == target

    graph.export(fname, if_compact=False)
    with open(filetest, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
    target = [
        "label\t\t\t\tabc",
        "muloffset\t[2, 3]",
        "useless\t\t\t\tyes",
        "1\t2\t\t1\t2",
    ]
    assert content == target

    graph.export(fname, if_only_labels=True)
    with open(filetest, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
    target = ["label\t\tabc", "1\t2\t2"]
    assert content == target


def test_graph_export_real(filetest, grapa_folder):
    fname = filetest.replace(".txt", "")
    folder = os.path.join(grapa_folder, "examples", "CV")
    graph = Graph()
    graph.merge(Graph(os.path.join(folder, "C-V_SAMPLE_f1_T=123_K.txt")))
    graph.merge(Graph(os.path.join(folder, "C-V_SAMPLE_f1_T=133_K.txt")))

    graph.export(fname)
    graphtest = Graph(filetest)
    are_graphs_same(graph, graphtest)

    graph.export(fname, if_compact=False)
    graphtest = Graph(filetest)
    are_graphs_same(graph, graphtest)

    graph.export(fname, save_altered=True)
    graphtest = Graph(filetest)
    are_graphs_same(graph, graphtest)

    graph.update({"alter": ["CurveCV.x_CVdepth_nm", "CurveCV.y_CV_Napparent"]})
    graph.export(fname, save_altered=True)
    graphtest = Graph(filetest)
    graphcompare = Graph()
    graphcompare.update(graph.get_attributes())
    alter = graph.get_alter()
    graphcompare.update({"alter": ""})
    for curve in graph:
        x = curve.x_offsets(alter=alter[0])
        y = curve.y_offsets(alter=alter[1])
        graphcompare.append(Curve([x, y], curve.get_attributes()))
        graphcompare[-1].update({"muloffset": ""})
    are_graphs_same(graphcompare, graphtest)


def test_graph_export_real2(filetest, grapa_folder):
    fname = filetest.replace(".txt", "")

    graph = Graph(os.path.join(grapa_folder, "examples", "subplots_examples.txt"))
    graph.export(fname)
    graphtest = Graph(filetest)
    are_graphs_same(graphtest, graph)

    graph = Graph(os.path.join(grapa_folder, "examples", "fancyAnnotations.txt"))
    graph.export(fname)
    graphtest = Graph(filetest)
    are_graphs_same(graphtest, graph)

    graph = Graph(os.path.join(grapa_folder, "examples", "example_imshow_picture.txt"))
    graph.export(fname)
    graphtest = Graph(filetest)
    are_graphs_same(graphtest, graph)

    graph = Graph(
        os.path.join(grapa_folder, "examples", "example_imshow_datatable.txt")
    )
    graph.export(fname)
    graphtest = Graph(filetest)
    are_graphs_same(graphtest, graph)

    graph = Graph(os.path.join(grapa_folder, "examples", "example_imshow_datafile.txt"))
    graph.export(fname)
    graphtest = Graph(filetest)
    are_graphs_same(graphtest, graph)
