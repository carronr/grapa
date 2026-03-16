"""At this stage, only function folder_abs is tested, not subplot itself"""
from grapa.curve_subplot import folder_initialdir, Curve_Subplot

from . import grapa_folder, Graph  # , HiddenPrints, GrapaWarning


def test_folder_initialdir_graphempty():
    """with graph created on scratch -> graph.filename is what it is"""
    graph = Graph()
    graph.append(Curve_Subplot([0, 0], {}))

    graph[0].update({"subplotfile": ""})
    subfile = graph[0].attr("subplotfile")
    assert folder_initialdir(subfile, graph) == ""

    graph[0].update({"subplotfile": "abc.txt"})
    subfile = graph[0].attr("subplotfile")
    assert folder_initialdir(subfile, graph) == ""

    graph[0].update({"subplotfile": r"C:\whatever.txt"})
    subfile = graph[0].attr("subplotfile")
    assert folder_initialdir(subfile, graph) == "C:\\"


def test_folder_initialdir_graphloaded(grapa_folder):
    """with graph created on scratch -> corresponding graph.filename"""
    graph = Graph(grapa_folder + "/examples/subplots_examples.txt")

    subfile = graph[0].attr("subplotfile")
    assert subfile == "fancyAnnotations.txt"
    assert folder_initialdir(subfile, graph).endswith("examples")

    graph[0].update({"subplotfile": "Cf/C-f_SAMPLE_f1_133K.txt"})
    subfile = graph[0].attr("subplotfile")
    value = folder_initialdir(subfile, graph)
    assert value.endswith("Cf") and "examples" in value

    graph[0].update({"subplotfile": r"C:\whatever.txt"})
    subfile = graph[0].attr("subplotfile")
    assert folder_initialdir(subfile, graph) == "C:\\"
