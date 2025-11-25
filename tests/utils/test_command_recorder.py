"""Tests for command_recorder module"""

# pylint: disable=redefined-outer-name

import pytest

import numpy as np

from grapa import Graph, Curve
from grapa.utils.command_recorder import (
    Action,
    Command,
    SuspendCommandRecorder,
)


@pytest.fixture
def graph1_nolog() -> Graph:
    """Graph with one curve, no recorder"""
    graph = Graph()
    graph.append(Curve([[1, 2], [3, 4]], {"label": "graph 0"}))
    return graph


@pytest.fixture
def graph1() -> Graph:
    """Graph with one curve, with recorder"""
    graph = Graph()
    graph.append(Curve([[1, 2], [3, 4]], {"label": "graph 0"}))
    graph.recorder.is_log_active(True)
    return graph


@pytest.fixture
def graph2() -> Graph:
    """Graph with two curves, with recorder"""
    graph = Graph()
    graph.append(Curve([[1, 2, 3], [4, 5, 6]], {"label": "graph 0"}))
    graph.append(Curve([[5], [6]], {"label": "graph 1"}))
    graph.recorder.is_log_active(True)
    return graph


@pytest.fixture
def action_test() -> Action:
    """An action for testing"""
    return Action("test_func", [1, 2], {"a": 3})


@pytest.fixture
def curve_test() -> Curve:
    """A curve with some attributes and a recorder"""
    graph = Graph()
    graph.append(Curve([[1, 2, 3], [4, 5, 6]], {"label": "abc", "color": "r"}))
    graph.recorder.is_log_active(False)
    return graph[0]


def test_register_graph(graph1_nolog):
    """Test Curve.register_graph and unregister_graph, and recorder activation
    propagation."""
    assert graph1_nolog.recorder.is_log_active() is False
    assert sum(curve.recorder.is_log_active() for curve in graph1_nolog) == 0
    graph1_nolog.recorder.is_log_active(True)
    assert graph1_nolog.recorder.is_log_active() is True
    assert sum(curve.recorder.is_log_active() for curve in graph1_nolog) == 1
    graph1_nolog.recorder.is_log_active(False)
    assert graph1_nolog.recorder.is_log_active() is False
    assert sum(curve.recorder.is_log_active() for curve in graph1_nolog) == 0
    # different instantiation
    graph = Graph()
    graph.append(Curve([[1, 2], [3, 4]], {}))
    assert not graph.recorder.is_log_active()
    assert not graph[0].recorder.is_log_active()
    graph.recorder.is_log_active(True)
    assert graph.recorder.is_log_active()
    assert graph[0].recorder.is_log_active()
    # different instatiation
    graph = Graph(log_active=True)
    graph.append(Curve([[1, 2], [3, 4]], {}))
    assert graph[0].recorder.is_log_active()
    graph.recorder.is_log_active(False)
    assert not graph.recorder.is_log_active()
    assert not graph[0].recorder.is_log_active()


def test_add_remove_curves_to_graph(graph1):
    """Test that when Curves are added or removed from a Graph with a recorder,
    the recorder is registered or unregistered in the Curve, as appropriate."""
    curve0 = graph1[0]
    assert curve0.recorder.is_log_active()
    del graph1[0]
    assert not curve0.recorder.is_log_active()  # test unregister_curve
    graph1.append(curve0)
    assert curve0.recorder.is_log_active()  # test register_curve


def test_contextmanager_suspend_command_recorder(graph2):
    """Test the context manager SuspendCommandRecorder"""
    assert len(graph2.recorder.past) == 0
    graph2.update({"figsize": [2, 3]})
    assert len(graph2.recorder.past) == 1
    assert graph2.recorder.is_log_active()

    with SuspendCommandRecorder(graph2.recorder):
        graph2.update({"xlabel": "abc"})
        assert len(graph2.recorder.past) == 1
        assert not graph2.recorder.is_log_active()

    graph2.update({"ylabel": "def"})
    assert len(graph2.recorder.past) == 2
    assert graph2.recorder.is_log_active()

    graph2.recorder.is_log_active(False)
    graph2.update({"ylabel": "ghi"})
    assert len(graph2.recorder.past) == 2
    assert not graph2.recorder.is_log_active()

    graph2.recorder.is_log_active(True)
    graph2.update({"ylabel": "jkl"})
    assert len(graph2.recorder.past) == 3
    assert graph2.recorder.is_log_active()


def test_command_recorder_caller_description_and_reference(graph2):
    """Test CommandRecorder.caller_description and caller_reference."""
    graph2.append(Curve([[4, 5], [7, 8]], {"label": "curve c 2"}))
    cr = graph2.recorder

    assert cr.caller_description(graph2) == []
    assert cr.caller_description(graph2.headers) == [Action("headers")]
    assert cr.caller_description(graph2.graphinfo) == [Action("graphinfo")]
    assert cr.caller_description(graph2._curves) == [Action("_curves")]
    desc = cr.caller_description(graph2[0])
    assert desc == [Action("curve", [0])]
    desc = cr.caller_description(graph2[1]._attr)
    assert desc == [Action("curve", [1]), Action("_attr")]
    desc = cr.caller_description(graph2[1].data)
    assert desc == [Action("curve", [1]), Action("data")]

    # caller_reference
    assert cr.caller_reference([]) is graph2
    assert cr.caller_reference([Action("headers")]) is graph2.headers
    assert cr.caller_reference([Action("graphinfo")]) is graph2.graphinfo
    assert cr.caller_reference([Action("_curves")]) is graph2._curves
    assert cr.caller_reference([Action("curve", [0])]) is graph2[0]
    # _attr and data
    assert (
        cr.caller_reference([Action("curve", [1]), Action("_attr")]) is graph2[1]._attr
    )
    assert cr.caller_reference([Action("curve", [0]), Action("data")]) is graph2[0].data


def test_action_str(action_test):
    """Test Action.__str__"""
    s = str(action_test)
    assert "test_func" in s
    assert "1" in s and "2" in s and "a" in s


def test_action_execute_graph_ok(graph1_nolog):
    """Test Action.execute on a Graph"""
    action = Action("update", [{"fontsize": 18}], {})
    action.execute(graph1_nolog)
    assert graph1_nolog.attr("fontsize") == 18


def test_action_execute_graph_fail(graph1_nolog):
    """Test Action.execute on a Graph, with a non existing method"""
    action = Action("doesnotexist", ["whatever"], {})
    try:
        action.execute(graph1_nolog)
        raise AssertionError
    except AttributeError:
        pass


def test_action_execute_curve_ok():
    """Test Action.execute on a Curve"""
    curve = Curve([[1, 2], [3, 4]], {})
    action = Action("update", [{"label": "abcdef"}], {})
    action.execute(curve)
    assert curve.attr("label") == "abcdef"


def test_command_do_and_undo(graph2):
    """Test Command.redo and Command.undo"""
    former = graph2.attr("xlabel")
    after = "after"
    do = Action("update", [{"xlabel": after}], {})
    undo = Action("update", [{"xlabel": former}], {})
    cmd = Command([], do, undo)

    assert graph2.attr("xlabel") == former
    cmd.redo(graph2.recorder)
    assert graph2.attr("xlabel") == after
    cmd.undo(graph2.recorder)
    assert graph2.attr("xlabel") == former


def test_command_recorder_log_and_undo_redo(graph2):
    """Test CommandRecorder.log, undo_last_transaction and redo_next_transaction"""
    past, future = graph2.recorder.transactions_length()
    assert len(past) == 0
    assert len(future) == 0

    graph2[0].update({"label": "after label", "color": "r"})
    graph2.update({"ylabel": "after ylabel"})
    graph2.recorder.tag_as_end_transaction()
    graph2.update({"xlabel": "after xlabel"})
    graph2.recorder.tag_as_end_transaction()
    past, future = graph2.recorder.transactions_length()
    assert len(past) == 2
    assert past[-1] == 1
    assert len(future) == 0
    assert graph2.attr("xlabel") == "after xlabel"
    assert graph2[0].attr("label") == "after label"

    graph2.recorder.undo_last_transaction()
    past, future = graph2.recorder.transactions_length()
    assert len(past) == 1
    assert past[-1] == 3
    assert len(future) == 1
    assert graph2.attr("xlabel") == ""
    assert graph2.attr("ylabel") == "after ylabel"
    assert graph2[0].attr("label") == "after label"

    graph2.recorder.undo_last_transaction()
    past, future = graph2.recorder.transactions_length()
    assert len(past) == 0
    assert len(future) == 2
    assert future[0] == 3
    assert graph2.attr("xlabel") == ""
    assert graph2.attr("xlabel") == ""
    assert graph2[0].attr("label") == "graph 0"

    graph2.recorder.redo_next_transaction()
    past, future = graph2.recorder.transactions_length()
    assert len(past) == 1
    assert len(future) == 1
    assert graph2.attr("xlabel") == ""
    assert graph2.attr("ylabel") == "after ylabel"
    assert graph2[0].attr("label") == "after label"

    graph2.recorder.redo_next_transaction()
    past, future = graph2.recorder.transactions_length()
    assert len(past) == 2
    assert len(future) == 0
    assert graph2.attr("xlabel") == "after xlabel"
    assert graph2[0].attr("label") == "after label"


def test_command_recorder_in_container_curves_delitem_0(graph2):
    """test container curves: __delitem__, undo and redo"""
    curve0 = graph2[0]
    curve1 = graph2[1]
    assert len(graph2) == 2
    assert graph2[0].recorder.is_log_active()
    assert graph2[1].recorder.is_log_active()
    # test delitem, and do and undo
    del graph2[0]
    graph2.recorder.tag_as_end_transaction()
    assert len(graph2) == 1
    assert graph2[0] is curve1
    assert not curve0.recorder.is_log_active()
    graph2.recorder.undo_last_transaction()
    assert len(graph2) == 2
    assert graph2[0] is curve0
    assert graph2[1] is curve1
    assert graph2[0].recorder.is_log_active()
    graph2.recorder.redo_next_transaction()
    assert len(graph2) == 1
    assert graph2[0] is curve1
    assert not curve0.recorder.is_log_active()

    del graph2[5]


def test_command_recorder_in_container_curves_setitem_0(graph1):
    """test container curves: __setitem__, undo and redo"""
    curve0 = graph1[0]
    curve1 = Curve([[1], [2]], {})
    assert len(graph1) == 1
    assert not curve1.recorder.is_log_active()
    graph1[0] = curve1
    graph1.recorder.tag_as_end_transaction()
    assert len(graph1) == 1
    assert graph1[0] is curve1
    assert not curve0.recorder.is_log_active()
    assert curve1.recorder.is_log_active()
    graph1.recorder.undo_last_transaction()
    assert len(graph1) == 1
    assert graph1[0] is curve0
    assert curve0.recorder.is_log_active()
    assert not curve1.recorder.is_log_active()
    graph1.recorder.redo_next_transaction()
    assert len(graph1) == 1
    assert graph1[0] is curve1
    assert not curve0.recorder.is_log_active()
    assert curve1.recorder.is_log_active()


def test_command_recorder_in_container_curves_clear(graph2):
    """test container curves: clear, undo and redo"""
    curve0 = graph2[0]
    curve1 = graph2[1]
    graph2._curves.clear()
    graph2.recorder.tag_as_end_transaction()
    assert len(graph2) == 0
    assert not curve0.recorder.is_log_active()
    assert not curve1.recorder.is_log_active()
    graph2.recorder.undo_last_transaction()
    assert len(graph2) == 2
    assert curve0.recorder.is_log_active()
    assert curve1.recorder.is_log_active()
    assert graph2[0] is curve0
    assert graph2[1] is curve1
    graph2.recorder.redo_next_transaction()
    assert len(graph2) == 0
    assert not curve0.recorder.is_log_active()
    assert not curve1.recorder.is_log_active()


def test_command_recorder_in_container_curves_append(graph1):
    """test container curves: append, undo and redo"""
    curve1 = Curve([[1], [2]], {})
    assert not curve1.recorder.is_log_active()
    assert len(graph1) == 1
    graph1.append(curve1)
    graph1.recorder.tag_as_end_transaction()
    assert len(graph1) == 2
    assert curve1.recorder.is_log_active()
    graph1.recorder.undo_last_transaction()
    assert len(graph1) == 1
    assert not curve1.recorder.is_log_active()
    graph1.recorder.redo_next_transaction()
    assert len(graph1) == 2
    assert curve1.recorder.is_log_active()


def test_command_recorder_in_container_curves_pop_0(graph2):
    """test container curves: pop, undo and redo"""
    curve0 = graph2[0]
    curve1 = graph2[1]
    del graph2[0]
    graph2.recorder.tag_as_end_transaction()
    assert len(graph2) == 1
    assert graph2[0] is curve1
    assert not curve0.recorder.is_log_active()
    graph2.recorder.undo_last_transaction()
    assert len(graph2) == 2
    assert graph2[0] is curve0
    assert graph2[1] is curve1
    assert curve0.recorder.is_log_active()
    graph2.recorder.redo_next_transaction()
    assert len(graph2) == 1
    assert graph2[0] is curve1
    assert not curve0.recorder.is_log_active()
    graph2.recorder.undo_last_transaction()


def test_command_recorder_in_container_curves_pop_1(graph2):
    """test container curves: pop, undo and redo"""
    curve0 = graph2[0]
    curve1 = graph2[1]
    assert len(curve1.graphs_membersof._list_graphs) == 1
    del graph2[1]
    graph2.recorder.tag_as_end_transaction()
    assert len(graph2) == 1
    assert graph2[0] is curve0
    assert len(curve0.graphs_membersof._list_graphs) == 1
    assert len(curve1.graphs_membersof._list_graphs) == 0
    assert not curve1.recorder.is_log_active()
    graph2.recorder.undo_last_transaction()
    assert len(graph2) == 2
    assert graph2[0] is curve0
    assert graph2[1] is curve1
    assert curve1.recorder.is_log_active()
    graph2.recorder.redo_next_transaction()
    assert len(graph2) == 1
    assert graph2[0] is curve0
    assert not curve1.recorder.is_log_active()
    graph2.recorder.undo_last_transaction()


def test_command_recorder_in_container_curves_reverse(graph2):
    """test container curves: reverse, undo and redo"""
    curve0 = graph2[0]
    curve1 = graph2[1]
    graph2.curves_reverse()
    graph2.recorder.tag_as_end_transaction()
    assert graph2[0] is curve1
    assert graph2[1] is curve0
    graph2.recorder.undo_last_transaction()
    assert graph2[0] is curve0
    assert graph2[1] is curve1
    graph2.recorder.redo_next_transaction()
    assert graph2[0] is curve1
    assert graph2[1] is curve0


def test_command_recorder_in_container_curves_reversegraph(graph2):
    """test container curves: curves_reverse, undo and redo"""
    curve0 = graph2[0]
    curve1 = graph2[1]
    graph2.curves_reverse()
    graph2.recorder.tag_as_end_transaction()
    assert graph2[0] is curve1
    assert graph2[1] is curve0
    graph2.recorder.undo_last_transaction()
    assert graph2[0] is curve0
    assert graph2[1] is curve1
    graph2.recorder.redo_next_transaction()
    assert graph2[0] is curve1
    assert graph2[1] is curve0


def test_command_recorder_in_container_metadata_clear(graph1):
    """test container metadata: clear, undo and redo"""
    curve = graph1[0]
    assert len(curve.get_attributes()) == 1
    curve._attr.clear()
    graph1.recorder.tag_as_end_transaction()
    assert len(curve.get_attributes()) == 0
    graph1.recorder.undo_last_transaction()
    assert len(curve.get_attributes()) == 1
    assert curve.attr("label") == "graph 0"
    graph1.recorder.redo_next_transaction()
    assert len(curve.get_attributes()) == 0


def test_command_recorder_in_container_metadata_update(graph1):
    """test container metadata: update, undo and redo"""
    curve = graph1[0]
    assert len(curve.get_attributes()) == 1

    curve.update({"linespec": "--"})  # new
    graph1.recorder.tag_as_end_transaction()
    assert len(curve.get_attributes()) == 2
    graph1.recorder.undo_last_transaction()
    assert len(curve.get_attributes()) == 1
    graph1.recorder.redo_next_transaction()
    assert len(curve.get_attributes()) == 2
    graph1.recorder.undo_last_transaction()

    curve.update({"label": "b"})  # update existing
    graph1.recorder.tag_as_end_transaction()
    assert len(curve.get_attributes()) == 1
    assert curve.attr("label") == "b"
    graph1.recorder.undo_last_transaction()
    assert len(curve.get_attributes()) == 1
    assert curve.attr("label") == "graph 0"
    graph1.recorder.redo_next_transaction()
    assert len(curve.get_attributes()) == 1
    assert curve.attr("label") == "b"
    graph1.recorder.undo_last_transaction()

    curve.update({"label": ""})  # delete
    graph1.recorder.tag_as_end_transaction()
    assert len(curve.get_attributes()) == 0
    graph1.recorder.undo_last_transaction()
    assert len(curve.get_attributes()) == 1
    assert curve.attr("label") == "graph 0"
    graph1.recorder.redo_next_transaction()
    assert len(curve.get_attributes()) == 0


def test_command_recorder_in_container_metadata_pop_exist(graph1):
    """test container metadata: pop existing, undo and redo"""
    curve = graph1[0]
    out = curve.attr_pop("label")
    graph1.recorder.tag_as_end_transaction()
    assert out == "graph 0"
    assert len(curve.get_attributes()) == 0
    graph1.recorder.undo_last_transaction()
    assert len(curve.get_attributes()) == 1
    graph1.recorder.redo_next_transaction()
    assert len(curve.get_attributes()) == 0


def test_command_recorder_in_container_metadata_pop_notexist(graph1):
    """test container metadata: pop non existing, undo and redo"""
    curve = graph1[0]
    out = curve.attr_pop("doesnotexist")
    graph1.recorder.tag_as_end_transaction()
    assert out == ""
    assert len(curve.get_attributes()) == 1
    graph1.recorder.undo_last_transaction()
    assert len(curve.get_attributes()) == 1
    graph1.recorder.redo_next_transaction()
    assert len(curve.get_attributes()) == 1


def test_command_recorder_in_curve_setx_indexnone(graph2):
    """test command recorder in Curve.setX with index=None, undo and redo"""
    curve = graph2[0]
    curve.setX([7, 8, 9])
    graph2.recorder.tag_as_end_transaction()
    assert (curve.x() == np.array([7, 8, 9])).all()
    graph2.recorder.undo_last_transaction()
    assert (curve.x() == np.array([1, 2, 3])).all()
    graph2.recorder.redo_next_transaction()
    assert (curve.x() == np.array([7, 8, 9])).all()


def test_command_recorder_in_curve_setx_indexwith0(graph2):
    """test command recorder in Curve.setX with index provided, undo and redo"""
    curve = graph2[0]
    curve.setX([8], index=[1])
    graph2.recorder.tag_as_end_transaction()
    assert (curve.x() == np.array([1, 8, 3])).all()
    graph2.recorder.undo_last_transaction()
    assert (curve.x() == np.array([1, 2, 3])).all()
    graph2.recorder.redo_next_transaction()
    assert (curve.x() == np.array([1, 8, 3])).all()


def test_command_recorder_in_curve_setx_indexwith1(graph2):
    """test command recorder in Curve.setX with index provided, undo and redo"""
    curve = graph2[0]
    curve.setX([8, 9], index=range(1, 3))
    graph2.recorder.tag_as_end_transaction()
    assert (curve.x() == np.array([1, 8, 9])).all()
    graph2.recorder.undo_last_transaction()
    assert (curve.x() == np.array([1, 2, 3])).all()
    graph2.recorder.redo_next_transaction()
    assert (curve.x() == np.array([1, 8, 9])).all()


def test_command_recorder_in_curve_sety_indexnone(graph1):
    """test command recorder in Curve.setY with index=None, undo and redo"""
    curve = graph1[0]
    curve.setY([7, 8])
    graph1.recorder.tag_as_end_transaction()
    assert (curve.y() == np.array([7, 8])).all()
    graph1.recorder.undo_last_transaction()
    assert (curve.y() == np.array([3, 4])).all()
    graph1.recorder.redo_next_transaction()
    assert (curve.y() == np.array([7, 8])).all()


def test_command_recorder_in_curve_sety_indexwith(graph2):
    """test command recorder in Curve.setY with index provided, undo and redo"""
    curve = graph2[0]
    curve.setY([7, 9], index=(0, 2))
    graph2.recorder.tag_as_end_transaction()
    assert (curve.y() == np.array([7, 5, 9])).all()
    graph2.recorder.undo_last_transaction()
    assert (curve.y() == np.array([4, 5, 6])).all()
    graph2.recorder.redo_next_transaction()
    assert (curve.y() == np.array([7, 5, 9])).all()


def test_command_recorder_in_curve_appendpoints(graph1):
    """test command recorder in Curve.appendPoints, undo and redo"""
    curve = graph1[0]
    curve.appendPoints([4, 5], [7, 8])
    graph1.recorder.tag_as_end_transaction()
    assert (curve.x() == np.array([1, 2, 4, 5])).all()
    assert (curve.y() == np.array([3, 4, 7, 8])).all()
    graph1.recorder.undo_last_transaction()
    assert (curve.x() == np.array([1, 2])).all()
    assert (curve.y() == np.array([3, 4])).all()
    graph1.recorder.redo_next_transaction()
    assert (curve.x() == np.array([1, 2, 4, 5])).all()
    assert (curve.y() == np.array([3, 4, 7, 8])).all()


def test_command_recorder_in_curve_set_data(graph1):
    """test command recorder in Curve.set_data, undo and redo"""
    curve = graph1[0]
    data = np.array([[0], [1]])
    curve.set_data(data)
    graph1.recorder.tag_as_end_transaction()
    assert (curve.x() == np.array([0])).all()
    assert (curve.y() == np.array([1])).all()
    graph1.recorder.undo_last_transaction()
    assert (curve.x() == np.array([1, 2])).all()
    assert (curve.y() == np.array([3, 4])).all()
    graph1.recorder.redo_next_transaction()
    assert (curve.x() == np.array([0])).all()
    assert (curve.y() == np.array([1])).all()


def test_command_recorder_in_curve_set_data2(graph1):
    """test command recorder in Curve.set_data followed by appendPoints,"""
    curve = graph1[0]
    data = np.array([[0], [1]])
    curve.set_data(data)
    graph1.recorder.tag_as_end_transaction()
    curve.appendPoints([1], [2])
    graph1.recorder.tag_as_end_transaction()
    assert (curve.x() == [0, 1]).all()
    assert (curve.y() == [1, 2]).all()
    graph1.recorder.undo_last_transaction()
    assert (curve.x() == [0]).all()
    assert (curve.y() == [1]).all()
    graph1.recorder.undo_last_transaction()
    assert (curve.x() == np.array([1, 2])).all()
    assert (curve.y() == np.array([3, 4])).all()
    graph1.recorder.redo_next_transaction()
    assert (curve.x() == [0]).all()
    assert (curve.y() == [1]).all()
    graph1.recorder.redo_next_transaction()
    assert (curve.x() == [0, 1]).all()
    assert (curve.y() == [1, 2]).all()
