# -*- coding: utf-8 -*-
"""
Defines Curves, a fondamental object to grapa. It stores both data and metadata.

@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import warnings
import importlib
import logging
from copy import deepcopy
from typing import List, Union, Optional, Any, Tuple, Callable, TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

from grapa.shared.constants import CST
from grapa.shared.command_recorder import CommandRecorderCurve, CurveGraphsReferrer
from grapa.shared.error_management import IncorrectInputError, GrapaError, issue_warning
from grapa.shared.maths import fractionstr_to_float, MathOperator
from grapa.shared.funcgui import FuncGUI, AlterListItem, funclistgui_graph_axislabels
from grapa.shared.string_manipulations import (
    format_string_curveattr,
    restructuredtext_to_text,
)
from grapa.internal.container_metadata import MetadataContainer
from grapa.internal.funclistguihelper import FuncListGUIHelper
from grapa.internal.curve_utils import (
    update_values_keys,
    update_graph_values_keys_condition,
)

if TYPE_CHECKING:
    from grapa.graph import Graph

logger = logging.getLogger(__name__)

Index = Union[int, slice, range]  # shortcut notation for what can be provided to .x()


class Curve:
    """A Curve object, fundamental to grapa. Contains data (presumably 2x 1-D vectors),
    and metadata (can be thought as a dict of key, values). keys are lower-case."""

    CURVE = "Curve"  # to replace with subclass identifier, used when loading files.

    # for subclasses to define suggestions of axis labels depending on data transform
    AXISLABELS_X = {}  # e.g. {KEY_ALTER: ["Axis label", "symbol", "unit"]}
    AXISLABELS_Y = {}  # also, funcListGUI must call funclistgui_axislabels
    # Keys for attributes, to suggest the user graph axis labels
    KEY_AXISLABEL_X = "_axislabel_x"
    KEY_AXISLABEL_Y = "_axislabel_y"

    # Values for alter[0] - further are defined in Curve subclasses
    ALTER_NM_EV = "nmeV"
    ALTER_NM_CM = "nmcm-1"
    ALTER_MCA_KEV = "MCAkeV"  # deprecated since 0.6.4.1
    ALTER_SIMS_DEPTH = "SIMSdepth"
    ALTER_Y = "y"

    # Values for alter[1] - further are defined in Curve subclasses
    ALTER_LOG10ABS = "log10abs"
    ALTER_ABS = "abs"
    ALTER_ABS0 = "abs0"
    ALTER_TAUC = "tauc"
    ALTER_TAUCLN1MINUSEQE = "taucln1-eqe"
    ALTER_NORMALIZED = "normalized"
    ALTER_X = "x"

    # keyword values for offset and muloffset
    OFFSET_MINMAX = "minmax"
    OFFSET_0MAX = "0max"

    # Accepted keyword values to hide a Curve (not visible), key linestyle
    LINESTYLEHIDE = ["none", "None", None]

    def __init__(self, data, attributes: dict, silent: bool = True):
        # silent: for debugging purpose. Subclasses may print additional information
        self.silent = silent
        self.graphs_membersof = CurveGraphsReferrer()
        self.data: np.ndarray = np.array([])
        self._attr = MetadataContainer(self)
        self.recorder = CommandRecorderCurve(self)
        self._init_data(data)
        self._attr.update(attributes)
        self._funclistgui_last = []
        if len(self.x()) == 0:
            self.appendPoints([0], [0])
            if self.attr("readas") != "database":
                print("Curve init: no data, pad with (0, 0).", self.get_attributes())

    def _init_data(self, newdata) -> None:
        """Format input data"""
        if isinstance(newdata, np.ndarray):
            self.data = newdata
        else:
            self.data = np.array(newdata)

        if self.data.shape[0] < 2:
            self.data = self.data.reshape((2, len(self.data)))
            msg = "Curve: data need to contain at least 2 columns. Shape of data: {}"
            issue_warning(logger, msg.format(self.data.shape))
        if self.data.ndim == 1:
            self.data = self.data.reshape(2, -1)
        # clean nan pairs at end of data
        imax = self.data.shape[1]
        for i in range(self.data.shape[1] - 1, 0, -1):
            if np.isnan(self.data[:, i]).all():
                imax -= 1
            else:
                break
        if imax < self.data.shape[1]:
            self.data = self.data[:, :imax]

    # *** Methods related to GUI
    @classmethod
    def classNameGUI(cls) -> str:
        """Provide base data, for the GUI to show what subclass of curve this is.
        Can be overridden, see CurveArrhenius."""
        return cls.CURVE

    def funcListGUI(self, **kwargs) -> list:
        """
        Fills in the Curve actions specific to the Curve type. Retuns a list, which
        elements are instances of `FuncGUI`, or (old style): ::

            [func,
             'Button text',
             ['label 1', 'label 2', ...],
             ['value 1', 'value 2', ...],
             {'hiddenvar1': 'value1', ...}, (optional)
             [dictFieldAttributes, {}, ...]] (optional)

        By default, returns quick modifs for offset and muloffset (if already set), and
        a help for some plot types (errorbar, scatter).

        :param kwargs: this function should be called specifying kwargs['graph'] the
               graph self is embedded in, and kwargs['graph_i'] as position of self in
               graph.
        """
        out = []
        out += FuncListGUIHelper.typeplot(self, **kwargs)  # e.g. scatter, boxplot etc.
        out += FuncListGUIHelper.offset_muloffset(self, **kwargs)
        if type(self) is Curve:  # pylint: disable=unidiomatic-typecheck
            # only if type Curve specifically - not subclass
            out += funclistgui_graph_axislabels(self, **kwargs)
        return out

    def _funclistgui_memorize(self, funclistgui):
        """Possibly used for print_help.
        Generally, want to generate funclistgui on-demand."""
        self._funclistgui_last = funclistgui

    def alterListGUI(self) -> list:
        """
        Determines the possible curve visualisations. One element has the form:
        AlterListItem('Label GUI', ['alter_x', 'alter_y'], 'semilogx', "print help doc")
        By default only neutral (i.e. raw data) is provided
        """
        # out = [AlterListItem.item_neutral()]
        return []

    # *** Methods about _attr
    def attr(self, key: str, default: Any = MetadataContainer.VALUE_DEFAULT) -> Any:
        """Get attribute value"""
        return self._attr(key, default=default)

    def has_attr(self, key: str) -> bool:
        """True if attribute key is defined, False if not defined or default"""
        return self._attr.has_attr(key)

    def is_attr_value_default(self, value) -> bool:
        """True is value is default (or does not exist), False if defined"""
        return self._attr.is_attr_value_default(value)

    def get_attributes(self, keys_list: Optional[list] = None) -> dict:
        """Returns all attributes, or a subset with keys given in keyList"""
        return self._attr.values(keys_list=keys_list)

    def update(self, attributes: dict) -> None:
        """Updates attributes. a dict must be provided"""
        return self._attr.update(attributes)

    def attr_pop(self, key: str):
        """Deletes an attribute, return the value"""
        return self._attr.pop(key)

    def label_auto(self, formatter):
        """
        Update the curve label according to formatting using python string Template,
        with curve attributes as variables. Examples:
        "${sample} ${_simselement}", "${sample} ${cell}", "${temperature [k]:.0f} K"
        """
        # if modify implementation: beware GraphSIMS, label_auto(...)
        string = format_string_curveattr(self, formatter)
        self.update({"label": string.replace("  ", " ").strip()})
        return True

    def visible(self, state: Optional[bool] = None):
        """Set/get state of visibility.
        Replaces .isHidden(), with added setter functionality.

        :param state: True to show the Curve, False to hide. None: unchanged state.
        :return: True if shown (visible), False if hidden
        """
        if state is not None:
            self.update({"linestyle": "" if state else "none"})
        return False if self.attr("linestyle") in Curve.LINESTYLEHIDE else True

    def data_units(self, unit_x=None, unit_y=None):
        """Set/get units for x and y data series.

        :param unit_x: new value for unit of data x
        :param unit_y: new value for unit of data y
        :returns: list[str] as ["nm", "%"], or ["V", "nF cm-2"]
        """
        units = self.attr("_units")
        if not isinstance(units, list):
            units = list(units)
        if len(units) != 2:  # ensure proper formatting
            while len(units) < 2:
                units.append("")
            units = units[:2]
            self.update({"_units": units})
        if unit_x is not None or unit_y is not None:  # update value if needed
            # make anew, otherwise issues with Curve([], other.getAttributes())
            units = list(units)
            if unit_x is not None:
                units[0] = str(unit_x)
            if unit_y is not None:
                units[1] = str(unit_y)
            self.update({"_units": units})
        return units

    def get_muloffset(self) -> list:
        """equivalent to attr("muloffset"), with ensure formatting as 2-element list"""
        value = self.attr("muloffset")
        if self.is_attr_value_default(value):
            return [1, 1]
        if isinstance(value, (int, float, str)):
            return [1, value]
        if isinstance(value, (list, tuple)):
            if isinstance(value, tuple):
                value = list(value)
            while len(value) < 2:
                value.append(1)
            return value[:2]
        msg = "Curve.get_muloffset, weird value not properly managed %s."
        issue_warning(logger, msg, value)
        return [1, 1]

    def update_values_keys(self, *values, keys: Optional[List[str]] = None):
        """Performs update({key1: value1, key2: value2, ...}). Handy to call from GUI

        :param values: value1, value2, ...
        :param list_keys: keys=['key1', 'key2', ...]. Same length as args. Not optional.
        :returns: True if success, False otherwise
        """
        # Shortcut: handy to be used from GUI
        return update_values_keys(self, *values, keys=keys)

    def update_values_keys_graph_condition(
        self,
        *args,
        keys: Optional[list] = None,
        graph: "Graph" = None,
        also_attr: Optional[list] = None,
        also_vals: Optional[list] = None,
    ):
        """Similar as update_values_keys, for all curves inside a provided graph.
        Implemented as curve method for easier integration into GUI.

        :param args: value1, value2, ...
        :param keys: ['key1', 'key2', ...]
        :param graph: Graph object, will apply to all Curves within
        :param also_attr: list of attribute keys.
        :param also_vals: list, same len as also_attr. Test each curve in graph, only
               perfom modifications if
               for all i in range, curve.attr(also_attr[i]) == also_vals[i].
        """
        return update_graph_values_keys_condition(
            graph, *args, keys=keys, also_attr=also_attr, also_vals=also_vals
        )

    def update_plotter_boxplot_addon(self, fun, kw: dict) -> bool:
        """set boxplot_addon keyword."""
        # cannot be placed in Plotter, in case user want to apply function on several
        # curves at once as selected from the GUI
        if fun == 0:
            self.update({"boxplot_addon": ""})
        else:
            if not isinstance(kw, dict):
                try:
                    kw = dict(kw)
                except ValueError:
                    msg = "update_plotter_boxplot_addon invalid input, failed ({})."
                    issue_warning(logger, msg.format(kw))
                    return False
            self.update({"boxplot_addon": [fun, kw]})
        return True

    def update_scatter_next_curves(self, *values, graph=None, graph_i=None) -> bool:
        """To receive update instructions for Curves following self in graph, when self
        "type" is "scatter".

        :param values: values for keyword "type", to apply on the Curves following self
               in graph.
        :param graph: the Graph object self is part of
        :param graph_i: index of self within graph (NB: graph may contain self >1x)
        :return: True if success, False otherwise
        """
        if graph is None or graph_i is None or graph[graph_i] != self:
            msg = (
                'Error in Curve.update_scatter_next_curves: keyword arguments "graph" '
                'and "graph_i" are mandatory, and graph[graph_i] must be equal to self.'
            )
            issue_warning(logger, msg)
            return False
        for i, val in enumerate(values):
            if graph_i + i + 1 < len(graph):
                graph[graph_i + 1 + i].update({"type": val})
        return True

    # *** Methods about data
    def get_data(self):
        """Returns the data as a np.array"""
        return self.data

    def set_data(self, data: np.ndarray):
        """Set curve data.
        Raise IncorrectInputError if failed."""
        if isinstance(data, np.ndarray):
            if self.recorder.is_log_active():
                old = deepcopy(self.data)
            self.data = data
            if self.recorder.is_log_active():
                self.recorder.log(
                    self, ("set_data", [data], {}), ("set_data", [old], {})
                )
            return True
        msg = "Curve.set_data: data must be ndarray. type %s."
        logger.error(msg, type(data))
        raise IncorrectInputError(msg % type(data))

    def shape(self, idx=":"):
        """Returns the shape of the data array, or one of its dimension if idx is 0 or 1"""
        if idx == ":":
            return self.data.shape
        return self.data.shape[idx]

    def x(
        self,
        index: Optional[Index] = None,
        alter="",
        xyValue=None,
        error_if_mix_xy=False,  # unused, there for similarity with curve.y()
    ) -> np.ndarray:
        """Returns the x data over the range index.

        :param index: range to the data.
        :param alter: requested alteration to the data (i.e. '', 'nmeV')
        :param xyValue: provide [x, y] values pair to be alter-ed. x, y can be np.array
               In that case, the curve x values are ignored.
        :param error_if_mix_xy: throw ValueError exception if alter value calculation
               requires both x and y components. Useful for transforming xlim and
               ylim, where x do not know y in advance.
        :return: a nd.array of x values
        """

        def reverse_xyvalue(xyval):
            if xyval is None:
                return xyval
            xyval = np.array(xyval)
            if len(xyval.shape) > 1:
                xyval = [xyval[0, ::-1], xyval[1, ::-1]]
            return xyval

        if alter == self.ALTER_X:
            alter = ""
        if alter != "":
            if alter == self.ALTER_NM_EV:  # "nmeV"
                xyValue = reverse_xyvalue(xyValue)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return CST.nm_eV / self.x(index, xyValue=xyValue)
            if alter == self.ALTER_NM_CM:  # "nmcm-1"
                xyValue = reverse_xyvalue(xyValue)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return 1e7 / self.x(index, xyValue=xyValue)
            if alter == self.ALTER_MCA_KEV:  # "MCAkeV":  # deprecated since 0.6.4.1
                offset = self.attr("_MCA_CtokeV_offset", default=0)
                mult = self.attr("_MCA_CtokeV_mult", default=1)
                return mult * (self.x(index, xyValue=xyValue) + offset)
            if alter == self.ALTER_SIMS_DEPTH:  # "SIMSdepth":
                offset = self.attr("_SIMSdepth_offset", default=0)
                mult = self.attr("_SIMSdepth_mult", default=1)
                return mult * (self.x(index, xyValue=xyValue) + offset)
            if alter == self.ALTER_Y:  # "y":
                if xyValue is not None:  # swap x and y
                    xyValue = xyValue[::-1]
                return self.y(index, xyValue=xyValue)
            if alter != "":
                success, result = self._xy_alterspecial(alter, index, xyValue)
                if success:
                    return result
        # alter might be used by subclasses
        val = self.data if xyValue is None else np.array(xyValue)
        if len(val.shape) > 1:
            if index is None or np.isnan(index).any():
                return val[0, :]
            return val[0, index]
        return val[0]

    def y(
        self,
        index: Optional[Index] = None,
        alter="",
        xyValue=None,
        error_if_mix_xy=False,
    ) -> np.ndarray:
        """Returns the y data over the range index.

        :param index: range to the data.
        :param alter: requested alteration to the data (i.e. '', 'nmeV')
        :param xyValue: provide [x, y] values pair to be alter-ed. x, y can be np.array
               In that case, the curve x values are ignored.
        :param error_if_mix_xy: raise ValueError if alter value calculation requires
               both x and y components. Used for transforming xlim and ylim, where y do
               not know x in advance.
        :return: a nd.array of y values
        """
        # if some alteration required: make operation on data requested by
        # itself, without operation
        if alter == self.ALTER_Y:
            alter = ""
        if alter != "":
            if alter == self.ALTER_LOG10ABS:  # "log10abs":
                jsc = self.attr("Jsc", 0)
                return np.log10(np.abs(self.y(index, xyValue=xyValue) + jsc))
            if alter == self.ALTER_ABS:  # "abs":
                jsc = self.attr("Jsc", 0)
                return np.abs(self.y(index, xyValue=xyValue) + jsc)
            if alter == self.ALTER_ABS0:  # "abs0":
                return np.abs(self.y(index, xyValue=xyValue))
            if alter == self.ALTER_TAUC:  # "tauc":  (eV * eqe)**2
                if error_if_mix_xy:
                    raise ValueError
                return np.power(
                    self.y(index, xyValue=xyValue)
                    * self.x(index, alter=self.ALTER_NM_EV, xyValue=xyValue),
                    2,
                )
            if alter == self.ALTER_TAUCLN1MINUSEQE:  # "taucln1-eqe":
                if error_if_mix_xy:
                    raise ValueError
                return np.power(
                    np.log(1 - self.y(index, xyValue=xyValue))
                    * self.x(index, alter=self.ALTER_NM_EV, xyValue=xyValue),
                    2,
                )
            if alter == self.ALTER_NORMALIZED:  # "normalized":
                out = self.y(index, xyValue=xyValue)
                return out / np.max(out)
            if alter == self.ALTER_X:  # "x":
                if xyValue is not None:  # swap x and y
                    xyValue = xyValue[::-1]
                return self.x(index, xyValue=xyValue)

            if alter != "":
                success, result = self._xy_alterspecial(alter, index, xyValue)
                if success:
                    return result
        # return (subset of) data
        val = self.data if xyValue is None else np.array(xyValue)
        if len(val.shape) > 1:
            if index is None or np.isnan(index).any():
                return val[1, :]
            return val[1, index]
        return val[1]

    def _xy_alterspecial(self, alter: str, index, xyvalue) -> Tuple[bool, np.ndarray]:
        """Execute a grapa function according to the semantics in 'alter'
        Return True if success, and result. Else, False and empty"""
        split = alter.split(".")
        if len(split) == 2:
            base = "grapa.datatypes.{}{}"
            module_name = base.format(split[0][0].lower(), split[0][1:])
            try:
                module = importlib.import_module(module_name)
                func = getattr(getattr(module, split[0]), split[1])
                try:
                    return (True, func(self, index=index, xyValue=xyvalue))
                except TypeError:
                    index_ = np.nan if index is None else index
                    return (True, func(self, index=index_, xyValue=xyvalue))
            except ImportError:
                msg = "Curve.x or .y: ImportError during module import %s:"
                logger.error(msg, module_name, exc_info=True)
                raise GrapaError(msg % module_name) from None
        else:
            if not alter.startswith("idle"):
                msg = "Curve.x or .y: cannot process keyword alter (%s)."
                logger.error(msg, alter)
                raise IncorrectInputError(msg % alter) from None
        return (False, np.array([]))

    def x_offsets(self, **kwargs):
        """Calls y(), and apply the effects of offset and muloffset on output."""
        return _xy_offsets(self, "x", **kwargs)

    def y_offsets(self, **kwargs):
        """Calls y(), and apply the effects of offset and muloffset on output."""
        return _xy_offsets(self, "y", **kwargs)

    def set_x(self, x, index=None):
        """Set new value for the x data.
        Index can be provided (self.data[0,index] = x)."""
        if index is None:
            if self.recorder.is_log_active():
                old = deepcopy(self.data[0, :])
            self.data[0, :] = x
            if self.recorder.is_log_active():
                self.recorder.log(
                    self, ("set_x", [deepcopy(x)], {}), ("set_x", [old], {})
                )
            return

        if self.recorder.is_log_active():
            old = deepcopy(self.data[0, index])
        self.data[0, index] = x
        if self.recorder.is_log_active():
            self.recorder.log(
                self,
                ("set_x", [deepcopy(x)], {"index": index}),
                ("set_x", [old], {"index": index}),
            )

    def set_y(self, y, index=None):
        """Set new value for the y data.
        Index can be provided (self.data[1,index] = y)."""
        if len(self.shape()) > 1:
            if index is None:
                if self.recorder.is_log_active():
                    old = deepcopy(self.data[1, :])
                self.data[1, :] = y
                if self.recorder.is_log_active():
                    ynew = deepcopy(y)
                    self.recorder.log(self, ("set_y", [ynew], {}), ("set_y", [old], {}))
            else:
                if self.recorder.is_log_active():
                    old = deepcopy(self.data[1, index])
                self.data[1, index] = y
                if self.recorder.is_log_active():
                    ynew = deepcopy(y)
                    self.recorder.log(
                        self,
                        ("set_y", [ynew], {"index": index}),
                        ("set_y", [old], {"index": index}),
                    )
        else:
            if self.recorder.is_log_active():
                old = deepcopy(self.data[1])
            self.data[1] = y
            if self.recorder.is_log_active():
                ynew = deepcopy(y)
                self.recorder.log(self, ("set_y", [ynew], {}), ("set_y", [old], {}))

    def appendPoints(self, x_series, y_series) -> bool:
        """Append datapoints.

        :param x_series: a list or np.array
        :param y_series: a list or np.array. same length as x_series
        :return: True if success, False otherwise.
        """
        x_series, y_series = np.array(x_series), np.array(y_series)
        if len(x_series) == 0:
            issue_warning(logger, "Curve.appendPoints: empty x_series provided.")
            return False

        if len(y_series) != len(x_series):
            msg = "Curve.appendPoints: cannot handle series with different lengths."
            issue_warning(logger, msg)
            return False

        if self.recorder.is_log_active():
            old = deepcopy(self.data)
        self.data = np.append(self.data, np.array([x_series, y_series]), axis=1)
        if self.recorder.is_log_active():
            self.recorder.log(
                self,
                ("appendPoints", [x_series, y_series], {}),
                ("set_data", [old], {}),
            )
        return True

    # Function related to Fits
    def updateFitParam(self, *param, func: Optional[Callable] = None):
        """Update the curve y values of a fit curve.
        Raise IncorrectInputError if problem with input.

        :param param: the new parameters to consider for the fit formula.
               Will be placed into _popt.
        :param func: if provided, will call func(...) instead of
               getattr(self, self.attr("_fitfunc"))(...)
               Used for example in curveMCA for fit
        """
        f = self.attr("_fitfunc")
        if not self.has_attr("_popt"):
            msg = "updateFitParam: Empty parameter (_fitfunc: {}, _popt: {})."
            raise IncorrectInputError(msg.format(f, self.attr("_popt")))
        if func is None:
            new_y = getattr(self, f)(self.x(), *param)
        else:
            new_y = func(self.x(), *param)
        self.set_y(new_y)
        self.update({"_popt": list(self.updateFitParamFormatPopt(f, param))})
        return True  # return True so GUI does not display anything

    def updateFitParamFormatPopt(self, _f, param):
        """Format the input parameters to be stored in _popt, depending on the fit
        function f. This is useful when the fit function has some fixed parameters, or
        when the input needs to be reshaped."""
        # possibility to override in subclasses, esp. when handling of test
        # input is required.
        # by default, assumes all numeric -> best stored in a np.array
        return np.array(param)

    # *** Methods relevant to Curve casting
    def cast_listpossibletypes(
        self, only_different: bool = True
    ) -> List[Tuple[str, str, type]]:
        """Returns a list of Curve subclasses into which a curve can be cast

        :param only_different: if false, also returns the object own subclass"""
        subclasses = [] if only_different else [Curve]
        subclasses += Curve.__subclasses__()
        out = []
        for subclass in subclasses:
            if not only_different or not isinstance(self, subclass):
                name = subclass.classNameGUI()
                out.append((name, subclass.__name__, subclass))
        # remove duplicates based on name classNameGUI
        key = [x[0] for x in out]
        for i in range(len(key) - 2, -1, -1):
            if key[i] in key[:i]:
                del key[i]
                del out[i]
        # sort list by name
        out = [x for (y, x) in sorted(zip(key, out), key=lambda pair: pair[0])]
        return out

    def cast_as(self, new_type: str) -> "Curve":
        """Returns a new Curve of the specified subtype, with same data and attributes
        as self."""
        data = self.data
        attributes = self.get_attributes()
        if new_type == "Curve":
            return Curve(data, attributes=attributes, silent=True)

        curve_types = self.cast_listpossibletypes()
        for type_ in curve_types:
            # typ_[0] subclass.CURVE, typ_[1] subclass.__name__
            if type_[0] == new_type or type_[1] in [
                new_type,
                new_type.replace(" ", ""),
            ]:
                if isinstance(self, type_[2]):  # rather a print. warning for debugging
                    msg = "Curve.cast_as: already that type, return a copy (%s)."
                    issue_warning(logger, msg % (new_type))
                return type_[2](data, attributes=attributes, silent=True)

        msg = "Curve.cast_as: could not cast (requested %s, possible types: %s)"
        msa = (new_type, list(curve_types))
        logger.error(msg, *msa)
        raise IncorrectInputError(msg % msa)
        # return False

    # *** Other methods...
    def select_data(self, xlim=None, ylim=None, alter="", offsets=False, data=None):
        """Returns slices of the data self.x(), self.y() which satisfy
        xlim[0] <= x <= xlim[1], and ylim[0] <= y <= ylim[1].

        :param xlim: [xmin, xmax], or None
        :param ylim: [ymin, ymax], or None
        :param alter: '', or ['', 'abs']. Affect the test AND the output.
        :param offsets: if True, calls to x and y are affected by offset and muloffset
        :param data:  [xSeries, ySeries] can be provided. If provided, alter and offsets
               are ignored.
        :return (x, y): as np.array
        """
        if isinstance(alter, str):
            alter = ["", alter]
        # retrieves entry data
        if data is None:
            if offsets:
                x, y = self.x_offsets(alter=alter[0]), self.y_offsets(alter=alter[1])
            else:
                x, y = self.x(alter=alter[0]), self.y(alter=alter[1])
        else:
            x, y = np.array(data[0]), np.array(data[1])
        # identify boundaries
        if xlim is None:
            xlim = [min(x), max(x)]
        if ylim is None:
            ylim = [min(y), max(y)]
        # construct a data mask
        mask = np.ones(len(x), dtype=bool)
        for i, _ in enumerate(mask):
            if x[i] < xlim[0] or x[i] > xlim[1]:
                mask[i] = False
            if y[i] < ylim[0] or y[i] > ylim[1]:
                mask[i] = False
        return (x[mask], y[mask])

    def getDataCustomPickerXY(
        self, idx, alter: Union[str, List[str]] = "", strDescription=False
    ):
        """
        Given an index in the Curve data, returns a modified data which has
        some sense to the user. Can be overridden in child classes, see e.g. CurveCf
        If strDescription is True, then returns a string which describes what
        this method is doing.
        For example, CurveCf: return temperature, omega instead of f, C
        """
        if strDescription:
            return "(x,y) data with offset and transform"
        if isinstance(alter, str):
            alter = ["", alter]
        attr = {}
        return (
            self.x_offsets(index=idx, alter=alter[0]),
            self.y_offsets(index=idx, alter=alter[1]),
            attr,
        )

    @classmethod
    def print_help_func(cls, func, funclabel: str = "", nparammax: int = -1):
        """prints the docstring of a function"""
        lines = restructuredtext_to_text(func.__doc__, nparammax=nparammax)
        if funclabel != "":
            lines[0] = funclabel + ": " + lines[0]
        lines = ["  " + line for line in lines]
        if len(lines) > 0 and len(lines[0]) > 0:
            lines[0] = "-" + lines[0][1:]
        print("\n".join(lines))

    def print_help(self):
        """Prints help for the Curve subclass, generated from docstrings."""
        print("*** *** ***")
        print(str(type(self).__doc__).strip())
        # List of data transform
        alterlist = self.alterListGUI()
        if len(alterlist) > 0:
            print("\nData transforms:")
            for alter in alterlist:
                if isinstance(alter, list):
                    alter = AlterListItem(*alter)
                string = f"- {alter.label}: {alter.doc}"
                print(string)
        # List of functions
        if len(self._funclistgui_last) == 0:
            self.funcListGUI()  # NB: not complete output as no graph and graph_i kwargs
        if len(self._funclistgui_last) == 0:
            msg = (
                "ERROR Curve.printHelp. One should execute self._funclistgui_memorize()"
                "in funcListGUI, or override printHelp to override in subclass %s."
            )
            raise NotImplementedError(msg % type(self))
        if len(self._funclistgui_last) > 0:
            print("\nAnalysis functions:")
        not_document = [self.print_help]
        for entry in self._funclistgui_last:
            if not isinstance(entry, FuncGUI):
                entry = FuncGUI(None, None).init_legacy(entry)
            if entry.func in not_document:
                continue
            print("\n".join(entry.func_docstring_to_text()))
        return True

    # *** arithmetics with Curves
    def __add__(self, other):
        return math_on_curves(self, other, operator=MathOperator.ADD)

    def __radd__(self, other):
        return math_on_curves(self, other, operator=MathOperator.ADD)

    def __sub__(self, other):
        """substract operation, element-wise or interpolated"""
        return math_on_curves(self, other, operator=MathOperator.SUB)

    def __rsub__(self, other):
        """reversed substract operation, element-wise or interpolated"""
        return math_on_curves(self.__neg__(), other, operator=MathOperator.ADD)

    def __mul__(self, other, **kwargs):
        """multiplication operation, element-wise or interpolated"""
        return math_on_curves(self, other, operator=MathOperator.MUL)

    def __rmul__(self, other):
        """r-multiplication operation"""
        return math_on_curves(self, other, operator=MathOperator.MUL)

    def __div__(self, other):
        """division operation, element-wise or interpolated"""
        return math_on_curves(self, other, operator=MathOperator.DIV)

    def __rdiv__(self, other):
        """division operation, element-wise or interpolated"""
        tmp = Curve([self.x(), 1 / self.y()], self.get_attributes())
        inverse = tmp.cast_as(self.classNameGUI())
        return math_on_curves(inverse, other, operator=MathOperator.MUL)

    def __truediv__(self, other):
        """division operation, element-wise or interpolated"""
        return math_on_curves(self, other, operator=MathOperator.DIV)

    def __rtruediv__(self, other):
        """division operation, element-wise or interpolated"""
        tmp = Curve([self.x(), 1 / self.y()], self.get_attributes())
        inverse = tmp.cast_as(self.classNameGUI())
        return math_on_curves(inverse, other, operator=MathOperator.MUL)

    def __pow__(self, other):
        """power operation, element-wise or interpolated"""
        return math_on_curves(self, other, operator=MathOperator.POW)

    def __neg__(self):
        out = Curve([self.x(), 0 - self.y()], self.get_attributes())
        return out.cast_as(self.classNameGUI())

    # DEPRECATED METHODS
    def swapShowHide(self) -> bool:  # @pylint: disable=C0103
        """. Deprecated. Toogle on/off if Curve is displayed or not.

        :meta private:"""
        self.update({"linestyle": "none" if self.visible() else ""})
        return True

    def isHidden(self) -> bool:  # @pylint: disable=C0103
        """. Deprecated. True if Curve is hidden, False if displayed.

        :meta private:"""
        return True if self.attr("linestyle") in Curve.LINESTYLEHIDE else False

    def getAttribute(  # @pylint: disable=C0103
        self, key: str, default=MetadataContainer.VALUE_DEFAULT
    ):
        """. Deprecated. Legacy getter of attribute

        :meta private:"""
        return self.attr(key, default=default)

    def getAttributes(self, keys_list=None) -> dict:  # @pylint: disable=C0103
        """. Deprecated. Use get_attributes instead

        :meta private:"""
        return self.get_attributes(keys_list=keys_list)

    def updateValuesDictkeys(self, *args, **kwargs):  # @pylint: disable=C0103
        """Deprecated, use update_values_keys instead
        Performs update({key1: value1, key2: value2, ...}). Handy to call from GUI

        :param args: value1, value2, ...
        :param kwargs: keys=['key1', 'key2', ...]. list[str]. Same length as args.
        :returns: True if success, False otherwise
        :meta private:"""
        # Shortcut: handy to be used from GUI
        msg = "Deprecated, use curve.update_values_keys() instead"
        issue_warning(logger, msg, category=DeprecationWarning)
        return update_values_keys(self, *args, keys=kwargs["keys"])

    def castCurve(self, new_type_gui):  # @pylint: disable=C0103
        """:meta private:"""
        msg = "Deprecated, use curve.curve_cast() instead"
        issue_warning(logger, msg, category=DeprecationWarning)
        return self.cast_as(new_type_gui)

    def updateValuesDictkeysGraph(  # @pylint: disable=C0103
        self,
        *args,
        keys: Optional[list] = None,
        graph: "Graph" = None,
        also_attr: Optional[list] = None,
        also_vals: Optional[list] = None,
    ):
        """Depreated, use update_values_keys_graph_conditional instead.
        :meta private:"""
        return self.update_values_keys_graph_condition(
            graph, *args, keys=keys, also_attr=also_attr, also_vals=also_vals
        )

    def getData(self):  # @pylint: disable=C0103
        """Deprecated. Use 'get_data' instead.

        :meta private:
        """
        msg = "Deprecated getData, use curve.get_data() instead"
        issue_warning(logger, msg, category=DeprecationWarning)
        return self.get_data()

    def selectData(  # @pylint: disable=C0103
        self, xlim=None, ylim=None, alter="", offsets=False, data=None
    ):
        """Deprecated: Use 'select_data' instead.

        :meta private:
        """
        msg = "Deprecated selectData, use curve.select_data() instead."
        issue_warning(logger, msg, category=DeprecationWarning)
        return self.select_data(
            xlim=xlim, ylim=ylim, alter=alter, offsets=offsets, data=data
        )

    def setX(self, x, index=None):  # @pylint: disable=C0103
        """Deprecated. Use 'set_x' instead."""
        msg = "Deprecated, use 'set_x' instead."
        issue_warning(logger, msg)
        return self.set_x(x, index=index)

    def setY(self, y, index=None):  # @pylint: disable=C0103
        """Deprecated. Use 'set_y' instead."""
        msg = "Deprecated, use 'set_y' instead."
        issue_warning(logger, msg)
        return self.set_y(y, index=index)


def math_on_curves(
    curve_a: Curve,
    curve_b: Union[Curve, float, int],
    interpolate=-1,
    offsets: bool = False,
    operator=MathOperator.ADD,
) -> Curve:
    """
    Math operations on Curves, element-wise or interpolated along x data.

    :param curve_a: a Curve object
    :param curve_b: a Curve object, or e.g. a float
    :param interpolate: interpolate, or not, the data on the x-axis

           - 0: no interpolation, only consider y() values

           - 1: interpolation, x are the points contained in curve_a.x() or in
             curve_b.x()

           - 2: interpolation, x are the points of curve_a.x(), restricted to min&max of
             curve_b.x()

           - -1: 0 if same x values (gain time), otherwise 1

    :param offsets: if True, adds Curves after computing offsets and muloffsets.
           If False, adds on the raw data.
    :param operator: MathOperator.ADD, MathOperator.SUB, MathOperator.MUL,
           MathOperator.DIV
    :return: a Curve object
    """

    def finalize(out_: Curve, curve, offsets_):
        out_.update(curve.get_attributes())
        # remove offset information if use it during calculation
        if offsets_:
            out_.update({"offset": "", "muloffset": ""})
        return out_.cast_as(curve.classNameGUI())

    ca_x = curve_a.x_offsets if offsets else curve_a.x
    ca_y = curve_a.y_offsets if offsets else curve_a.y
    if not isinstance(curve_b, Curve):  # add something/number to a Curve
        out = Curve([ca_x(), MathOperator.operate(ca_y(), curve_b, operator)], {})
        return finalize(out, curve_a, offsets)

    # 'other' is a Curve
    cb_x = curve_b.x_offsets if offsets else curve_b.x
    cb_y = curve_b.y_offsets if offsets else curve_b.y
    # default mode -1: check if can gain time (avoid interpolating)
    r = range(0, min(len(ca_y()), len(cb_y())))
    if interpolate == -1:
        interpolate = 0 if np.array_equal(curve_a.x(index=r), curve_b.x(index=r)) else 1
    # avoiding interpolation
    if not interpolate:
        le = min(len(ca_y()), len(cb_y()))
        r = range(0, le)
        if le < len(ca_y()):
            msg = "math_on_curves: Curves not same length, clip to shortest ({}, {})"
            issue_warning(logger, msg.format(len(ca_y()), len(cb_y())))
        if not np.array_equal(curve_a.x(index=r), curve_b.x(index=r)):
            msg = (
                "math_on_curves ({}): Curves not same x axis values. Consider "
                "interpolation (interpolate=1)."
            )
            issue_warning(logger, msg.format(operator))
        datay = MathOperator.operate(curve_a.y(index=r), curve_b.y(index=r), operator)
        out = Curve([curve_a.x(index=r), datay], curve_b.get_attributes())
        return finalize(out, curve_a, offsets)

    # not elementwise : interpolate
    # construct new x -> all x which are in the range of the other curve
    datax = list(ca_x())
    if interpolate == 1:  # x from both self and other
        xmin = max(min(ca_x()), min(cb_x()))
        xmax = min(max(ca_x()), max(cb_x()))
        # no duplicates
        datax += [x for x in cb_x() if x not in datax]
    else:
        # interpolate 2: copy x from self, restrict to min&max of other
        xmin, xmax = min(cb_x()), max(cb_x())
    datax = [x for x in datax if xmax >= x >= xmin]
    revers_ = (curve_a.x()[0] > curve_a.x()[1]) if len(ca_x()) > 1 else False
    datax.sort(reverse=revers_)
    f0 = interp1d(ca_x(), ca_y(), kind="linear")
    f1 = interp1d(cb_x(), cb_y(), kind="linear")
    datay = [MathOperator.operate(f0(x), f1(x), operator) for x in datax]
    out = Curve([datax, datay], curve_b.get_attributes())
    return finalize(out, curve_a, offsets)


def _xy_offsets(curve: Curve, x_or_y, **kwargs):
    """Calls x() or y(), and apply the effects of offset and muloffset on output."""
    series = curve.x(**kwargs) if x_or_y == "x" else curve.y(**kwargs)
    reserved = [curve.OFFSET_MINMAX, curve.OFFSET_0MAX]
    special = None
    # offset
    offset_input = curve.attr("offset")
    if not curve.is_attr_value_default(offset_input):
        if isinstance(offset_input, list):
            offset = offset_input[0] if x_or_y == "x" else offset_input[1]
        else:
            offset = 0 if x_or_y == "x" else offset_input
        if isinstance(offset, str):
            if offset in reserved:
                special = offset
                offset = 0
            else:
                offset = fractionstr_to_float(offset)
        series = series + offset
    # muloffset
    muloffset_input = curve.attr("muloffset")
    if not curve.is_attr_value_default(muloffset_input):
        if isinstance(muloffset_input, list):
            muloffset = muloffset_input[0] if x_or_y == "x" else muloffset_input[1]
        else:
            muloffset = 1 if x_or_y == "x" else muloffset_input
        if isinstance(muloffset, str):
            if muloffset.replace(" ", "") in reserved:
                special = muloffset
                muloffset = 1
            else:
                muloffset = fractionstr_to_float(muloffset)
        series = series * muloffset
    # special
    if special is not None:
        min_, max_ = np.min(series), np.max(series)
        if special == curve.OFFSET_MINMAX:
            series = (series - min_) / (max_ - min_)
        elif special == curve.OFFSET_0MAX:
            series = series / max_
    return series
