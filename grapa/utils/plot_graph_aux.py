# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:21:57 2018

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
from copy import copy
import warnings
from inspect import signature
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes

from grapa.mathModule import is_number
from grapa.utils.error_management import issue_warning

logger = logging.getLogger(__name__)


class SubplotsAdjuster:
    """
    A utility class for managing and adjusting subplot_adjust parameter of matplotlib.

    Methods:
    --------
    default():
        Retrieves the default `subplots_adjust` parameters from Matplotlib's `rcParams`.

    merge(default, spa_user, fig):
        Merges default subplot parameters with user-defined parameters, handling
        relative adjustments and figure size scaling.

        Parameters:
        - default (dict): Dictionary of default subplot parameters.

        - spa_user (dict, or list): User-defined subplot adjustment parameters.
          Can be a string, dictionary, or list specifying adjustments.

        - fig (matplotlib.figure.Figure or tuple or lsit): Either a Matplotlib figure
          object or a tuple specifying the figure size.

        Returns:

        - dict: A dictionary containing the merged subplot adjustment parameters.
    """

    @classmethod
    def default(cls):
        """Retrieves matplotlib default subplots_adjust parameters in rcParams"""
        subplot_adjust = {}
        for key in ["bottom", "left", "right", "top", "hspace", "wspace"]:
            subplot_adjust.update({key: mpl.rcParams["figure.subplot." + key]})
        return subplot_adjust

    @classmethod
    def merge(cls, default, spa_user, fig):
        """
        :param default: dict of default parameters
        :param spa_user: dict, list, or float of user-defined parameters
                         if single float: interpreted as 'bottom'.
                         if spa_user[-1] is 'abs', all values are understood in terms of
                         absolute values and not relative as usually the case
        :param fig: either a figure, or a list or tuple of 2 elements 'figsize'
        """
        subplot_adjust = dict(default)
        if spa_user != "":
            if isinstance(spa_user, dict):
                subplot_adjust.update(spa_user)
            elif isinstance(spa_user, list):
                # handles relative in given in absolute
                if isinstance(spa_user[-1], str) and spa_user[-1] == "abs":
                    del spa_user[-1]
                    if hasattr(fig, "get_size_inches"):
                        fs = fig.get_size_inches()
                    else:
                        fs = fig
                    for i in range(len(spa_user)):
                        spa_user[i] = spa_user[i] / fs[i % 2]
                indic = ["left", "bottom", "right", "top", "wspace", "hspace"]
                for i in range(min(len(spa_user), len(indic))):
                    subplot_adjust.update({indic[i]: float(spa_user[i])})
            else:
                try:
                    subplot_adjust.update({"bottom": float(spa_user)})
                except ValueError as e:
                    msg = (
                        "'subplot_adjust' should be dict, list or numeric ({}). {}, {}."
                    )
                    issue_warning(logger, msg.format(spa_user, type(e), e))
                    subplot_adjust = {}
        return subplot_adjust


class ParserAxlines(ABC):
    """
    Possible inputs:
    - 3
    - [3, {'color':'r', 'xmin': 0.5]
    - [1, 2, 4, {'color':'b'}]
    - [[1, {'color':'c'}], [2, {'color':'m'}]]
    """

    def __init__(self, data):
        self._list = []
        if not isinstance(data, list):  # handles "3"
            data = [data]
        self._initialize(data)

    def _initialize(self, data: list):
        kwdef = {"color": "k"}
        if isinstance(data[-1], dict):  # handles [..., {'color':'m'}]
            kwdef.update(data[-1])
            data = data[:-1]
        for el in data:
            kw = copy(kwdef)
            if not isinstance(el, list):  # handles [1,2,3...]
                self._list.append({"args": [el], "kwargs": kw})
            else:
                el_ = el
                # handles [[1,2, {'color':'y'], [3, {'color': 'm'}]]
                if isinstance(el[-1], dict):
                    kw.update(el[-1])
                    el_ = el[:-1]
                for e in el_:
                    self._list.append({"args": [e], "kwargs": kw})

    def __str__(self):
        return "self\t" + "\n\t".join([str(a) for a in self._list])

    @abstractmethod
    def plot(self, ax, curvedummy, alter, type_plot):
        """Plots the lines on the given axes."""
        # abstract base method. implementations should take care of e.g. alter
        for el in self._list:
            ax.axhline(*el["args"], **el["kwargs"])
        return True


class ParserAxhline(ParserAxlines):
    """Parser for horizontal lines in a plot."""

    def plot(self, ax, curvedummy, alter, type_plot):
        for el in self._list:
            pos = curvedummy.y(
                alter=alter[1],
                xyValue=[1, el["args"][0]],
                errorIfxyMix=True,
                neutral=True,
            )
            if pos == 0 and type_plot in ["semilogy", "loglog"]:
                continue

            try:
                ax.axhline(pos, **el["kwargs"])
            except Exception as e:  # e.g. axhline  [1, {'color': 'e'}]
                msg = "Exception in ParserAxhline.plot: %s, %s. %s: %s."
                issue_warning(logger, msg, *(pos, el, type(e), e))
        return True


class ParserAxvline(ParserAxlines):
    """Parser for vertical lines in a plot."""

    def plot(self, ax, curvedummy, alter, type_plot):
        for el in self._list:
            pos = curvedummy.x(
                alter=alter[0],
                xyValue=[el["args"][0], 1],
                errorIfxyMix=True,
                neutral=True,
            )
            if pos == 0 and type_plot in ["semilogx", "loglog"]:
                continue

            try:
                ax.axvline(pos, **el["kwargs"])
            except Exception as e:  # e.g. axvline  [1, {'color': 'e'}]
                msg = "Exception ParserAxhline.plot: {}, {}. {}: {}."
                issue_warning(logger, msg.format(pos, el, type(e), e))
        return True


class ParserMisc:
    """Class to handle miscellaneous plot properties"""

    @classmethod
    def set_xyticklabels(cls, value, xyaxis):
        """
        Set ticks according to user wishes
        xyaxis: e.g. ax.xaxis
        """
        # value = self.graphInfo["xtickslabels"]
        # ensure proper formatting of input
        if not isinstance(value, list):
            value = [value, [], {}]
        value = list(value)  # let us not modify input object
        if len(value) == 0:
            value.append(None)
        if len(value) == 1:
            value.append(None)
        if len(value) == 2:
            value.append({})
        newticks = True
        # actual work
        ticksloc = value[0]
        if not isinstance(ticksloc, list):
            ticksloc = xyaxis.get_ticklocs()
            newticks = False  # later will be set to existing values
        labels = []
        kw = {}
        if value[1] is not None:
            labels = list(value[1])
        if isinstance(value[2], dict):
            kw = dict(value[2])
        if len(kw) > 0 and len(labels) != len(ticksloc):
            # mandatory kw labels if provide keywords
            if newticks:
                xyaxis.set_ticks(ticksloc)
            labels = [la.get_text() for la in xyaxis.get_ticklabels()]
        if len(labels) == len(ticksloc):
            proto = signature(xyaxis.set_ticks)
            if "labels" in proto.parameters:
                kw.update({"labels": labels})
            else:
                msg = (
                    "xtickslabels or ytickslabels: attempt to use of "
                    "ax.set_ticks() with keyword labels, but matplotlib version "
                    "too old. labels ignored, likely display errors."
                )
                issue_warning(logger, msg)
        try:
            xyaxis.set_ticks(ticksloc, **kw)
        except TypeError:
            for key in ["color", "rotation"]:
                if key in kw:
                    del kw[key]
            xyaxis.set_ticks(ticksloc, **kw)

        return ticksloc, kw

    @classmethod
    def set_axislabel(cls, method, label, graph):
        """Sets the axis label with the given method and label."""
        out = {"size": False}
        if isinstance(label, list) and len(label) == 2 and isinstance(label[-1], dict):
            method(label[0], **label[-1])
            if "size" in label[-1] or "fontsize" in label[-1]:
                out["size"] = True
        elif isinstance(label, list) and len(label) in [3, 4]:
            lbl = graph.formatAxisLabel(label[0:3])
            kwargs = label[-1] if isinstance(label[-1], dict) else {}
            method(lbl, **kwargs)
            if "size" in kwargs or "fontsize" in kwargs:
                out["size"] = True
        else:
            method(label)
        return out

    @classmethod
    def alter_lim(cls, ax, lim, xory, alter, curvedummy):
        """Alters the limits of the specified axis."""
        limauto = ax.get_xlim() if xory == "x" else ax.get_ylim()
        if not isinstance(lim, list):
            lim = list(lim)
        # limInput = [li if not isinstance(li, str) else np.inf for li in lim]
        liminput = [li if is_number(li) else np.inf for li in lim]
        lim = list(liminput)
        while len(lim) < 2:
            lim.append(np.inf)
        if xory == "x":
            fun = ax.set_xlim
        elif xory == "y":
            fun = ax.set_ylim
        else:
            return False

        try:
            if xory == "x":
                newlim = curvedummy.x(
                    alter=alter[0],
                    xyValue=[lim, [0] * len(lim)],
                    errorIfxyMix=True,
                    neutral=True,
                )
                lim = list(newlim)
            elif xory == "y":
                newlim = curvedummy.y(
                    alter=alter[1],
                    xyValue=[[0] * len(lim), lim],
                    errorIfxyMix=True,
                    neutral=True,
                )
                lim = list(newlim)
        except ValueError:
            msg = (
                "ValueError in ParserMisc alter_lim: cannot transform limit, keep"
                "initial input. lim {}, xory {}, xyValue {}, alter {}, curvedummy."
            )  # Presumably should change logging level. e.g. debug, print, or silent
            issue_warning(logger, msg.format(lim, xory, [[0] * len(lim), lim], alter))
        except TypeError:
            msg = (
                "TypeError in ParserMisc alter_lim TypeError, continue without axis"
                "limit. lim {}, xory {}, alter {}, curvedummy."
            )  # this is an error, to log.
            issue_warning(logger, msg.format(lim, xory, alter))
            lim = [np.inf, np.inf]
        # if inf (e.g. alter transform failed) whereas input was provided,
        # then take provided value
        if np.sum(np.isinf(lim)) == 2 and np.sum(np.isinf(liminput)) < 2:
            lim = [li if not np.isinf(li) else None for li in liminput]
        else:  # default matplotlib values if not provided
            lim = [
                limauto[i] if np.isnan(lim[i]) or np.isinf(lim[i]) else lim[i]
                for i in range(len(lim))
            ]
        if lim[0] != lim[1]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fun((lim[0], lim[1]))

    @classmethod
    def set_xytickstep(cls, val, xyaxis, axlim):
        """Set ticks according to user wishes"""
        arr = None
        if isinstance(val, list):
            arr = val
        elif val != 0:
            val = abs(val)
            # axlim = xyaxis.get_lim()  # for some reason get_lim does not exist
            ticks = [t for t in xyaxis.get_ticklocs() if min(axlim) <= t <= max(axlim)]
            start, end = min(ticks), max(ticks)
            while start - val >= min(axlim):
                start -= val
            while end + val <= max(axlim):
                end += val
            arr = np.arange(start, end + end / 1e10, val)
        # performs modification
        if arr is not None:
            xyaxis.set_ticks(arr)
            return True

        issue_warning(logger, "xticksstep or yticksstep invalid input {}.".format(val))
        return False


class GroupedPlotters:
    """class to handle boxplots, violin plots and related"""

    def __init__(self):
        self.plotters: Dict[str, Plotter] = {
            "boxplot": PlotterBoxplot(),
            "violinplot": PlotterViolin(),
        }
        Plotter.position = 0

    def add_curve(self, type_graph, curve, y, fmt, ax):
        """Add a curve to the appropriate plotter based on the type of graph."""
        for key, plotter in self.plotters.items():
            if type_graph == key:
                return plotter.add_curve(curve, y, fmt, ax)

    def plot(self, ax: Axes, ax_or_one_of_its_twins):
        """Plot all the added curves using their respective plotters."""
        for _key, plotter in self.plotters.items():
            if len(plotter) > 0:
                plotter.plot(ax, ax_or_one_of_its_twins)


class Plotter(ABC):
    """Base class for handling grouped plots like boxplots and violin plots."""

    position = 0
    IMPORTERROR = "ImportError"

    SEABORN_STRIPPLOT = "seaborn.stripplot"
    SEABORN_SWARMPLOT = "seaborn.swarmplot"
    SCATTER = "scatter"
    SCATTER_JITTER_DEFAULT = 0.15

    def __init__(self):
        self.y = []
        self.callkw: Dict[str, Any] = {"positions": [], "labels": []}
        self.colors = []
        # aim: warnings to be shown only once per graph plotting; Instances of plotter
        self._showwarnings = {"swarmplot_jitter": True}

    def __len__(self):
        """Returns the number of y datasets added to the plotter."""
        return len(self.y)

    def plot_addon_scatter(self, curve, y, position, ax):
        """Handles possible scatter addons to boxplot or violin plot"""
        addon = curve.attr("boxplot_addon", None)  # seaborn_stripplot_kw
        if not isinstance(addon, list) or len(addon) != 2:
            return None
        fun, kw = addon[0], dict(addon[1])
        if fun == Plotter.SCATTER:
            jitter = Plotter.SCATTER_JITTER_DEFAULT
            if "jitter" in kw:
                jitter = kw["jitter"]
                del kw["jitter"]
            if "size" in kw and "s" not in kw:
                kw["s"] = kw["size"]
                del kw["size"]
            x = position + jitter * 2 * (np.random.rand(len(y)) - 0.5)
            return ax.scatter(x, y, **kw)

        if fun == Plotter.SEABORN_STRIPPLOT:
            try:
                import seaborn as sns  # grapa must run even without seaborn
            except ImportError:
                msg = "GroupedPlotters ImportError with seaborn, cannot plot stripplot."
                issue_warning(logger, msg)
                return self.IMPORTERROR
            else:
                kw.update({"ax": ax, "native_scale": True})
                return sns.stripplot({position: y[~np.isnan(y)]}, **kw)

        if fun == Plotter.SEABORN_SWARMPLOT:
            try:
                import seaborn as sns  # grapa must run even without seaborn
            except ImportError:
                msg = "GroupedPlotters ImportError with seaborn, cannot plot swarmplot."
                print(msg)
                return self.IMPORTERROR
            else:
                kw.update({"ax": ax, "native_scale": True})
                if "jitter" in kw:
                    del kw["jitter"]
                    if self._showwarnings["swarmplot_jitter"]:
                        msg = "Plotter plot_addon_scatter swarmplot: removed kw jitter"
                        issue_warning(logger, msg)
                        self._showwarnings["swarmplot_jitter"] = False
                with warnings.catch_warnings(record=True) as warns:
                    handle = sns.swarmplot({position: y[~np.isnan(y)]}, **kw)
                    if len(warns) > 0:
                        for warn in warns:
                            msg = "%s %s: %s"
                            issue_warning(
                                logger, msg, warn.category, warn.filename, warn.message
                            )
                    return handle
        return None

    @classmethod
    def funcListGUI(cls, curve):
        """Generates a GUI function list for configuring boxplot addons."""
        current = curve.attr("boxplot_addon")
        color = curve.attr("color", "grey")
        val0 = ["", cls.SCATTER, cls.SEABORN_STRIPPLOT, cls.SEABORN_SWARMPLOT]
        val1 = [
            "",
            {},
            {"color": curve.attr("color", "k")},
            {"color": color, "jitter": Plotter.SCATTER_JITTER_DEFAULT},
            {"color": color, "size": 5, "jitter": Plotter.SCATTER_JITTER_DEFAULT},
        ]
        if not isinstance(current, list):
            def0, def1 = "", ""
        else:
            while len(current) < 2:
                current.append("")
            def0 = current[0]
            def1 = current[1]
            if def1 not in val1 and isinstance(def1, dict):
                val1.insert(0, def1)
        out = [
            [
                curve.update_plotter_boxplot_addon,
                "Save",
                ["also show data", "kw"],
                [def0, def1],
                {},
                [
                    {"field": "Combobox", "values": val0},
                    {"field": "Combobox", "values": val1, "width": 20},
                ],
            ]
        ]
        return out

    @abstractmethod
    def add_curve(self, curve, y, fmt, ax):
        """Add a curve to the boxplot or violin plot."""

    @abstractmethod
    def plot(self, ax: Axes, ax_or_one_of_its_twins: Axes):
        """Plots the boxplot or violinplot on the given axes."""


class PlotterBoxplot(Plotter):
    """class to handle boxplots"""

    def add_curve(self, curve, y, fmt, ax):
        """Add a curve to the boxplot."""
        handle = None
        # y: already computed from calling nearby environment
        if len(y) > 0 and not np.isnan(y).all():
            position = curve.attr("boxplot_position", Plotter.position)
            self.y.append(y[~np.isnan(y)])
            self.callkw["positions"].append(position)
            self.callkw["labels"].append(fmt["label"] if "label" in fmt else "")
            self.colors.append(fmt["color"] if "color" in fmt else "")
            # aggressive gather of possible parameters
            for key in fmt:
                if key not in self.callkw and key not in ["label", "color"]:
                    self.callkw[key] = fmt[key]
            # addon - seaborn swarmplot or stripplot
            handle = self.plot_addon_scatter(curve, y, position, ax)
            if handle == Plotter.IMPORTERROR:
                # do not mask fliers if seaborn import failed
                self.callkw["showfliers"] = True
                print("boxplot -> show fliers True")
                handle = None
            Plotter.position += 1
        return handle

    def plot(self, ax: Axes, _ax_or_one_of_its_twins: Axes):
        """Plots the boxplot on the given axes."""
        # NOTE: the handling of ax and ax_or_one_of_its_twins looks wrong
        # TODO: presumably the ax_twinx does not work well

        # change in matplotlib parameters according to version
        version = mpl.__version__
        if tuple(map(int, version.split(".")[:2])) >= (3, 9):
            if "labels" in self.callkw:
                self.callkw.update({"tick_labels": self.callkw["labels"]})
                del self.callkw["labels"]

        handle = ax.boxplot(self.y, **self.callkw)
        # handle coloring
        nb_el_boxplot = {"whiskers": 2, "caps": 2}
        for key in handle:
            for i in range(len(handle[key])):
                j = (
                    i
                    if key not in nb_el_boxplot
                    else int(np.floor(i / nb_el_boxplot[key]))
                )
                if j < len(self.colors) and self.colors[j] != "":
                    handle[key][i].set_color(self.colors[j])
                    if hasattr(handle[key][i], "set_markeredgecolor"):
                        handle[key][i].set_markeredgecolor(self.colors[j])
        return handle


class PlotterViolin(Plotter):
    """class to handle violin plots"""

    def __init__(self):
        super().__init__()
        del self.callkw["labels"]
        self.labels = []

    def add_curve(self, curve, y, fmt, ax):
        """Add a curve to the violin plot."""
        handle = None
        if len(y) > 0 and not np.isnan(y).all():
            position = curve.attr("boxplot_position", Plotter.position)
            self.y.append(y[~np.isnan(y)])
            self.callkw["positions"].append(position)
            self.labels.append(fmt["label"] if "label" in fmt else "")
            self.colors.append(fmt["color"] if "color" in fmt else "")
            # could use a more aggressive strategy to gather keywords
            for key in ["showmeans", "showmediansshowmedians", "showextrema"]:
                value = curve.attr(key)
                if value != "":
                    self.callkw[key] = value
            # addon - seaborn swarmplot or stripplot
            handle = self.plot_addon_scatter(curve, y, position, ax)
            if handle == Plotter.IMPORTERROR:
                handle = None
            Plotter.position += 1
        return handle

    def plot(self, ax: Axes, ax_or_one_of_its_twins: Axes):
        """Plots the violin plot on the given axes."""
        # NOTE: the handling of ax and ax_or_one_of_its_twins looks wrong
        # TODO presumably the ax_twinx does not work well
        # print("violinplot plot self.callkw", self.callkw)
        handle = ax.violinplot(self.y, **self.callkw)
        # set color
        for key in handle:
            if isinstance(handle[key], list):
                for i, handlekeyi in enumerate(handle[key]):
                    if self.colors[i] != "":
                        handlekeyi.set_color(self.colors[i])
            else:
                handle[key].set_color((0, 0, 0))
        # violinplots labels not set automatically, unlike boxplot
        for i, labelsi in enumerate(self.labels):
            xticks = list(ax_or_one_of_its_twins.get_xticks())
            xtklbl = list(ax_or_one_of_its_twins.get_xticklabels())
            xticks.append(self.callkw["positions"][i])
            xtklbl.append(labelsi)
            ax_or_one_of_its_twins.set_xticks(xticks)
            ax_or_one_of_its_twins.set_xticklabels(xtklbl)
        return handle
