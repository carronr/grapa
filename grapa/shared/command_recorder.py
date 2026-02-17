"""
@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING
from abc import abstractmethod, ABC

import numpy as np

from grapa.shared.error_management import issue_warning

if TYPE_CHECKING:
    from grapa.graph import Graph
    from grapa.curve import Curve

logger = logging.getLogger(__name__)


class Action:
    """Action describes a function call: func, args, kwargs.
    func is a str. args a list, kwargs a dict"""

    def __init__(
        self, func: str, args: Optional[list] = None, kwargs: Optional[dict] = None
    ):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, o: "Action"):
        # used for sure in unit tests. Possibly in actual code.
        if self.func == o.func and self.args == o.args and self.kwargs == o.kwargs:
            return True
        return False

    def __str__(self):
        return "{}(*{}, **{})".format(self.func, self.args, self.kwargs)

    def execute(self, obj):
        """Exectue the Action on an object"""
        getattr(obj, self.func)(*self.args, **self.kwargs)


class Command:
    """A command contains a subject, and do and undo Actions."""

    ACTION_NULL = Action("")

    def __init__(
        self, who: List[Action], redo: Action, undo: Action, tag_special: str = ""
    ):
        self._who = who
        self._redo = redo
        self._undo = undo
        self._tag_end_transaction = False
        self.tag_special = tag_special

    def __str__(self):
        msg = "Command tagged {}: {}. {}, {}."
        return msg.format(
            self._tag_end_transaction,
            ["{}(*{})".format(w.func, w.args) for w in self._who],
            self.tag_special,
            self._redo,
            self._undo,
        )

    def is_null(self):
        """returns True if Command has no effect"""
        return self._redo is self.ACTION_NULL and self._undo is self.ACTION_NULL

    def is_end_transaction(self, new_tag: Optional[bool] = None):
        """returns True if Command was tagged as end of a transaction"""
        if isinstance(new_tag, bool):
            self._tag_end_transaction = new_tag
        return self._tag_end_transaction

    def redo(self, cr: "CommandRecorderBase"):
        """Perform the actions recorded in the command."""
        if self._redo.func != "":  # possible reason: undo defined, do should do nothing
            # print("Command do", self._who, self._redo)
            callee = cr.caller_reference(self._who)
            with SuspendCommandRecorder(cr):
                self._redo.execute(callee)
        return True

    def undo(self, cr: "CommandRecorderBase"):
        """Undo the actions recorded in the command."""
        if self._undo.func != "":
            # print("Command undo", self._who, self._undo)
            callee = cr.caller_reference(self._who)
            with SuspendCommandRecorder(cr):
                self._undo.execute(callee)
        return True


class CommandRecorderBase(ABC):
    """Records commands (actions) performed on a graph, to allow undo/redo.
    Each command is associated to a caller (graph, graphinfo, headers, curve, curve.data).
    The caller is identified automatically by inspecting the call stack."""

    MAX_N_TRANSACTIONS = 15

    COMMAND_NULL = Command([], Command.ACTION_NULL, Command.ACTION_NULL)

    def __init__(self, log_active=True):
        self.past: List[Command] = []
        self.future: List[Command] = []
        self._log_active = log_active

    @abstractmethod
    def caller_description(self, reference) -> List[Action]:
        """Returns a list of Action, that can be used later to identify 'caller'"""

    @abstractmethod
    def caller_reference(self, description: List[Action]):
        """Returns e.g. reference of en object, identified from a list of Action."""

    def is_log_active(self, new: Optional[bool] = None):
        """is CommandRecorder active? can be paused"""
        if isinstance(new, bool):
            self._log_active = new
        return self._log_active

    def log(
        self,
        caller,
        do: Tuple[str, list, dict],
        undo: Tuple[str, list, dict],
        tag_special: str = "",
        blend_into_transaction=False,
    ):
        """Register a command, with do and undo actions."""
        if self._log_active:
            who = self.caller_description(caller)
            command = Command(who, Action(*do), Action(*undo), tag_special=tag_special)
            # merge into existing transaction, if requested. presumably, for tag_special
            if blend_into_transaction:
                if len(self.past) > 0 and self.past[-1].is_end_transaction():
                    self.past[-1].is_end_transaction(False)
                    command.is_end_transaction(True)
            self.past.append(command)
            # print("CommandRecorder active", command)
            # deletes the content of future - we took a new timeline
            if not command.is_null():
                self.future.clear()
            # limit number of transactions
            past, _future = self.transactions_length()
            n_transactions = len(past)
            while n_transactions > self.MAX_N_TRANSACTIONS:
                self.delete_oldest_transaction()
                n_transactions -= 1
        # else:  # e.g. currently doing do/undo actions that should not be logged
        # print(
        #    "CommandRecorder inactive",
        #    Command(self.caller_description(caller), Action(""), Action("")),
        # )

    def log_special(self, tag: str, blend_into_transaction=True):
        """Log a special command with no effect, just a tag.
        Application example: mark save points."""
        self.log(
            None,
            ("", [], {}),
            ("", [], {}),
            tag_special=tag,
            blend_into_transaction=blend_into_transaction,
        )

    def last_command(self):
        """Return the last command in the past stack, or COMMAND_NULL if empty."""
        if len(self.past) > 0:
            return self.past[-1]
        return self.COMMAND_NULL

    def delete_oldest_transaction(self):
        """deletes data corresponding to the oldest transaction in .past"""
        while len(self.past) > 0:
            item = self.past.pop(0)
            if item.is_end_transaction():
                return

    def transactions_length(self) -> Tuple[List[int], List[int]]:
        """Return the number of commands in past and future, grouped by marks."""

        def count(series: List[Command]):
            # if series[0] == self.COMMAND_NULL:
            #     series = series[1:]
            lengths = [0]
            for command in series:
                lengths[-1] += 1
                if command.is_end_transaction():
                    lengths.append(0)
            if lengths[-1] == 0:
                del lengths[-1]
            return lengths

        # for command in self.past:
        #     print(command)
        lengths_past = count(self.past)
        lengths_future = count(self.future)
        return lengths_past, lengths_future

    def redo_once(self):
        """Redo one command from the future stack."""
        if len(self.future) > 0:
            self.future[0].redo(self)
            self.past.append(self.future.pop(0))
        else:
            issue_warning(logger, "CommandRecorder do_once: cannot do, future is empty")

    def undo_once(self):
        """Undo one command from the past stack."""
        if len(self.past) > 0:
            self.past[-1].undo(self)
            self.future.insert(0, self.past.pop(-1))
        else:
            issue_warning(logger, "CommandRecorder undo_once: cannot do, past is empty")

    def tag_as_end_transaction(self):
        """Set a mark on the last command in the stack past."""
        if len(self.past) > 0:
            self.past[-1].is_end_transaction(True)

    def redo_next_transaction(self, revert_if_fail=True):
        """Redo commands until a marked command is reached (including that one)."""
        try:
            while len(self.future) > 0:
                self.redo_once()
                if self.past[-1].is_end_transaction():
                    # the one that was just executed
                    return
        except Exception as e:
            msg = "%s during do_next_transaction: %s."
            logger.error(msg, type(e), e, exc_info=True)
            if revert_if_fail:
                self.undo_last_transaction(revert_if_fail=False)

    def undo_last_transaction(self, revert_if_fail=True):
        """Undo at least 1 command until a marked command is reached (excluding)"""
        try:
            while len(self.past) > 0:
                self.undo_once()
                if len(self.past) == 0 or self.past[-1].is_end_transaction():
                    return
        except Exception as e:
            msg = "%s during undo_next_transaction: %s."
            logger.error(msg, type(e), e, exc_info=True)
            if revert_if_fail:
                self.undo_last_transaction(revert_if_fail=False)


class CommandRecorderGraph(CommandRecorderBase):
    """Decouple: CommandRecorderBase takes care of the logic, and
    implementation details specific to grapa in CommandRecorderGraph e.g. how to
    retrieve caller from description, and how to dscribe caller

    The Curves must be part of the Graph on which the CommandRecorder was defined.
    """

    def __init__(self, graph: "Graph", log_active=True):
        super().__init__(log_active=log_active)
        self.graph = graph

    def caller_description(self, reference) -> List[Action]:
        """Opposite of caller_reference"""
        # design choice: CommandRecorder has knowledge of data organisation within grapa
        # see also functions register_... and unregister_...
        who = []
        if reference is None:
            return who
        if reference is self.graph:
            return who
        if reference is self.graph.headers:
            who.append(Action("headers"))
            return who
        if reference is self.graph.graphinfo:
            who.append(Action("graphinfo"))
            return who
        if reference is self.graph._curves:
            who.append(Action("_curves"))
            return who
        for c, curve in enumerate(self.graph):
            if reference is curve:  # e.g. curve.update(...)
                who.append(Action("curve", [c]))
                return who
            if reference is curve._attr:
                # curve attribute - if possible, rather use curve
                who.append(Action("curve", [c]))
                who.append(Action("_attr"))
                return who
            if reference is curve.data:  # curve data e.g. modified values
                who.append(Action("curve", [c]))
                who.append(Action("data"))
                return who
        msg = "CommandRecorder caller_description cannot identify '{}' type {}. {}. who {}.\n{}"
        raise RuntimeError(
            msg.format(reference, type(reference), reference.host, who, self.graph)
        )

    def caller_reference(self, description: List[Action]):
        """Return the object on which the actions will be performed.xx
        E.g. graph, graph.headers, graph.graphinfo, graph.curve(c), graph.curve(c).data
        Opposite of caller_reference
        Search by whitelist, not aggressive search.
        """
        from grapa.graph import Graph
        from grapa.curve import Curve

        caller = self.graph
        for action in description:
            func, args = action.func, action.args
            if func == "headers" and isinstance(caller, Graph):
                caller = caller.headers
            elif func == "graphinfo" and isinstance(caller, Graph):
                caller = caller.graphinfo
            elif func == "_curves" and isinstance(caller, Graph):
                caller = caller._curves
            elif func == "curve" and isinstance(caller, Graph):
                caller = caller[args[0]]
            elif func == "_attr" and isinstance(caller, Curve):
                caller = caller._attr
            elif func == "data" and isinstance(caller, Curve):
                caller = caller.data
            else:
                msg = "CommandRecorder caller_reference not found {}: {}, {}"
                raise NotImplementedError(msg.format(type(caller).__name__, func, args))
        return caller


class SuspendCommandRecorder:
    """A context manager to temporarily suspend logging in CommandRecorder"""

    def __init__(self, command_recorder: CommandRecorderBase):
        self.command_recorder = command_recorder
        self.former = True  # will be set in __enter__

    def __enter__(self):
        if self.command_recorder is not None:
            self.former = self.command_recorder.is_log_active()
            self.command_recorder.is_log_active(False)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.command_recorder is not None:
            self.command_recorder.is_log_active(self.former)


class CommandRecorderCurve:
    """Records operations performed on a Curve, to allow undo/redo functionality.
    Used by Curve.recorder.
    Similar interfaces as CommandRecorderGraph.
    NB: log an operation makes sense irrespective of which Graphs the Curve belongs to
    but undo_last_transaction() may screw up -> open door for undesired changes on
    unrelated Graphs
    -> undo_last_transaction() and redo_last_transaction() are NOT implemented in
    CommandRecorderCurve
    """

    def __init__(self, curve: "Curve"):
        self.curve = curve

    def is_log_active(self, new: Optional[bool] = None):
        """Get/set if logging is active for any of the Graphs containing the Curve.
        See CommandRecorderGraph.is_log_active().
        :param new: if not None, set new logging state"""
        al = [gr.recorder.is_log_active(new=new) for gr in self.curve.graphs_membersof]
        return np.array(al).any()

    def log(
        self,
        caller,
        do: Tuple[str, list, dict],
        undo: Tuple[str, list, dict],
        tag_special: str = "",
        blend_into_transaction=False,
    ):
        """Log an operation performed on the Curve."""
        for graph in self.curve.graphs_membersof:
            graph.recorder.log(
                caller,
                do,
                undo,
                tag_special=tag_special,
                blend_into_transaction=blend_into_transaction,
            )

    def log_special(self, tag: str, blend_into_transaction=True):
        """Log a special operation performed on the Curve."""
        for graph in self.curve.graphs_membersof:
            graph.recorder.log_special(
                tag, blend_into_transaction=blend_into_transaction
            )


class CurveGraphsReferrer:
    """Keeps track of Graphs containing a Curve.
    Used by Curve to know which Graphs it belongs to. Taken advantage by CommandRecorder
    Workflow: action on Graph.ContainerCurves, which triggers action onto
    CurveGraphsReferrer"""

    def __init__(self):
        self._list_graphs: List["Graph"] = []

    def __len__(self):
        return len(self._list_graphs)

    def __iter__(self):
        return self._list_graphs.__iter__()

    def register_graph(self, graph):
        """Register a graph as containing the Curve."""
        if graph in self._list_graphs:
            msg = "CurveGraphsReferrer.register: graph already in _list. Proceed."
            msg += str([item for item in self._list_graphs])
            for c, curve in enumerate(graph):
                args = [
                    c,
                    curve.attr("label"),
                    curve.graphs_membersof is self,
                    self in [curve.graphs_membersof],
                ]
                msg += "\n  {} {} {} {}".format(*args)
            issue_warning(logger, msg, exc_info=True)
            return
        self._list_graphs.append(graph)
        self.cleanup()

    def unregister_graph(self, graph):
        """Unregister a graph as containing the Curve."""
        if graph not in self._list_graphs:
            msg = "CurveGraphsReferrer.unregister: graph not in _list. Proceed."
            issue_warning(logger, msg, exc_info=True)
            return
        if self in [curve.graphs_membersof for curve in graph]:
            msg = "CurveGraphsReferrer.unregister: curve still in graph, keep."
            issue_warning(logger, msg, exc_info=True)
            return
        self._list_graphs.remove(graph)
        self.cleanup()

    def cleanup(self):
        """Cleans up _list_graphs from invalid/outdated entries.
        purose: bug tracking. Should never find anything.
        REMOVE FUNCTION at some point"""
        to_remove = []
        for i, gr in enumerate(self._list_graphs):
            if self not in [curve.graphs_membersof for curve in gr]:
                to_remove.append(i)
        for i in to_remove[::-1]:
            del self._list_graphs[i]
            msg = "CurveGraphsReferrer.cleanup: graph %s del from list. Proceed."
            logger.error(msg, i)

        for graph in self._list_graphs:
            for c, curve in enumerate(graph):
                if graph not in curve.graphs_membersof:
                    msg = (
                        "CurveGraphsReferrer.cleanup, issue with graph %s, curve %s"
                        " %s, graph not registered. Not attempted correction."
                    )
                    logger.error(msg, graph.filename, c, curve.attr("label"))
