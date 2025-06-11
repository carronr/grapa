# -*- coding: utf-8 -*-
"""
Collection of classes and functions to plot graphs.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import warnings
from copy import deepcopy
from re import split as resplit
import logging
import subprocess

import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as tck

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

from grapa.curve import Curve
from grapa.graph import Graph
from grapa.curve_inset import Curve_Inset
from grapa.curve_subplot import Curve_Subplot
from grapa.utils.string_manipulations import strToVar, TextHandler
from grapa.utils.graphIO import export_filesave_default  # GraphIO
from grapa.utils.plot_curve import plot_curve
from grapa.utils.plot_graph_aux import (
    ParserAxhline,
    ParserAxvline,
    ParserMisc,
    GroupedPlotters,
    SubplotsAdjuster,
)

logger = logging.getLogger(__name__)


def plot_graph(
    graph: Graph,
    filesave: str = "",
    img_format: str = "",
    figsize=(0, 0),
    if_save=True,
    if_export=True,
    fig_ax=None,
    if_subplot=False,
):
    """
    Plot the content of the object.

    :param graph: a Graph object
    :param filesave: (optional) filename for the saved graph
    :param img_format: by default image will be .png. Possible format are the ones
           accepted by plt.savefig
    :param figsize: default figure size is a class consant
    :param if_save: if True, save the plot as an image.
    :param if_export: if True, reate a human- and machine-readable .txt file
           containing all information of the graph
    :param fig_ax: [fig, ax]. Useful when wish to embedd graph in a GUI.
    :param if_subplot: False, unless fig to be drawn in a subplot. Not handled by the
           GUI, but might be useful in scripts. Prevents deletion of existing axes.
    """
    handles = []

    # treat filesave, and store info if not already done. required for
    # relative path to subplots or insets
    filename = export_filesave_default(graph, filesave)
    if not hasattr(graph, "filename"):
        graph.filename = filename

    # Store attributes which might be modified upon execution
    restore = {}
    for attr in ["text", "textxy", "textargs"]:
        restore.update({attr: graph.attr(attr)})

    # retrieve default axis positions subplotadjust_def
    subplotadjust_def = SubplotsAdjuster.default()
    for curve in graph:
        if curve.attr("colorbar", None) is not None:
            subplotadjust_def["right"] -= 0.1
            break

    # other data we want to retrieve now
    fontsize = Graph.DEFAULT["fontsize"]
    if "fontsize" in graph.graphinfo:
        fontsize = graph.graphinfo["fontsize"]
    alter = graph.get_alter()
    if not isinstance(alter, list):
        # CHECK WITH PRINTOUT. may lead to bug with data copy
        logger.warning("SOMETHING FISHY HERE graphIO alter", alter, type(alter))
    ignore_xlim = True if alter[0] != "" else False
    ignore_ylim = True if alter[1] != "" else False

    ax_twinx, ax_twiny, ax_twinxy = None, None, None

    # check
    if len(graph) == 0:
        if not if_subplot:
            print("Warning plot {}: no data to plot!".format(graph.filename))
            if fig_ax is not None:
                fig, ax = fig_ax[0], fig_ax[1]
                if ax is not None:
                    ScaErrorHandler.attempt(ax)
                # plt.cla()
                if ax is not None:
                    ax.cla()
            # return 1

    # retrieve figure size
    if figsize == (0, 0):  # if figure size not imposed at function call
        figsize = graph.FIGSIZE_DEFAULT
        if "figsize" in graph.graphinfo:
            figsize = graph.graphinfo["figsize"]

    # create graph, initialize axes
    to_unpack = _create_subplots_gridspec(graph, fig_ax, if_subplot=if_subplot)
    ([fig, ax], gs, matrix, subplotsid, subplotsngraphs) = to_unpack

    # HOPEFULLY SOLVED BUG - grapa fails to plot figures especially with subplots
    # seems that ax is sometimes not None whereas the Figure does not contain any ax
    # print("Graph.plot. {}. # {}.".format(fig, plt.get_fignums()), "ax: {}. {}".format
    # ax, len(fig.get_axes())), os.path.basename(self.attr("filename")))
    # if len(fig.get_axes()) == 0 and ax is not None:
    #    print("MAYBE FAILS!?!")
    #    print("   ", matrix, gs)

    # either ax is None and gs is not None, either the other way around
    fig.patch.set_alpha(0.0)
    if not if_subplot:  # set figure size if fig is not a subplot
        fig.set_size_inches(*figsize, forward=True)

    # adjust positions of the axis within the graph (subplots_adjust)
    attr_spa = graph.attr("subplots_adjust")
    subplot_adjust = SubplotsAdjuster.merge(subplotadjust_def, attr_spa, fig)
    if fig is not None:
        fig.subplots_adjust(**subplot_adjust)
    if gs is not None:
        gs.update(**subplot_adjust)

    # set default graph scales (lin, log, etc.)
    type_plot = graph.attr("typeplot")
    if type_plot.endswith(" norm."):
        type_plot = type_plot[:-6]
        # print("type_plot new", type_plot)
    _plot_set_axis_scale(type_plot, gs, ax)

    groupedplotters = GroupedPlotters()
    container = {
        "fig": fig,
        "gs": gs,
        "matrix": matrix,
        "subplotsid": subplotsid,
        "fontsize": fontsize,
        "ax_twinx": ax_twinx,
        "ax_twiny": ax_twiny,
        "ax_twinxy": ax_twinxy,
        "groupedplotters": groupedplotters,
    }

    # Start looping on the curves
    to_unpack2 = _plot_loop_over_curves(graph, container, ax, handles)
    # unpack results of loop. NB: groupedplotters was modified in the process
    ax_and_twins, ax_or_one_of_its_twins, axes, handles, graph_aux = to_unpack2
    ax, ax_twinx, ax_twiny, ax_twinxy = ax_and_twins
    # end of loop over Curves

    # generates boxplot and violinplot, after plotting of other curves
    groupedplotters.plot(ax, ax_or_one_of_its_twins)
    # final display of auxiliary axes/graph
    if graph_aux is not None:
        plot_graph(
            graph_aux, if_save=False, if_export=False, fig_ax=[fig, ax], if_subplot=True
        )
        del graph_aux  # not quite necessary. Clean-up after graph_aux is dealt with.
        # main graph: work on initial axis
        ax = axes[0]
        ScaErrorHandler.attempt(ax, "- 2")

    # graph cosmetics
    curvedummy = graph.curve(-1) if len(graph) > 0 else Curve([[0], [0]], {})
    fontsizeset = []
    ax_and_twins = (ax, ax_twinx, ax_twiny, ax_twinxy)
    # title, xlabel, ylabel, xticklabels, yticklabels (to do before xlim and ylim)
    _plot_add_title_labels_ticklabels(graph, fontsizeset, gs, ax_and_twins)
    # xlim must be after xtickslabels
    _plot_set_xylims(graph, ax_and_twins, alter, curvedummy)
    # xtickstep and ytickstep
    _plot_add_xytickssteps(graph, ax, ignore_xlim, ignore_ylim)
    # other
    _plot_add_axhvlines(graph, ax, curvedummy, alter, type_plot)
    # text annotations
    _plot_add_text(graph, ax)
    # create legend
    if gs is None:  # create legend on current ax
        _ = _plot_add_legend(graph, ax, handles)
    # arbitrary functions
    _plot_apply_arbitrary_functions(graph, ax, fig)
    # font sizes - one of the last things to do
    if gs is None:
        _plot_change_fontsize(fontsize, fontsizeset, ax_and_twins)

    # before saving image: restore self to its initial state
    graph.update(restore)

    # save the graph if an export format was provided
    export_format = ""
    if img_format in [".txt", ".xml"]:
        export_format = img_format
        img_format = ""
    if if_save:
        _plot_savefig(graph, fig, filename, img_format)
    if if_export:
        graph.export(filesave=filename + export_format)
    return [fig, axes]


def _create_subplots_gridspec(graph, fig_ax=None, if_subplot=False):
    """
    Generates a Figure (if not provided) with a gridpec axes matrix
    """
    # TODO: Handle subplotskwargs: sharex True, 'col'
    # set fig as the active figure, if provided. Else with current figure
    if fig_ax is not None:
        fig, ax = fig_ax
        plt.figure(fig.number)  # bring existing figure to front
        # delete all except the provided axis
        if not if_subplot:
            for ax_ in fig.get_axes():
                if ax_ is not ax:
                    plt.delaxes(ax_)
    else:
        fig = plt.figure()  # create a new figure
        ax = None
    # count number of graphs to be plotted
    axes = [{"ax": None, "activenext": True} for _curve in graph]
    ncurvesondefault = 0  # do not create default axis if no curve on it
    isthereasubplot = False
    ngraphs = 0
    for c in range(len(graph)):
        curve = graph[c]
        if isinstance(curve, Curve_Inset):
            axes[c].update({"ax": "inset", "activenext": False})
            # default can be overridden if Graph file has length 0
        elif isinstance(curve, Curve_Subplot) and curve.visible():
            rowspan = int(curve.attr("subplotrowspan", 1))
            colspan = int(curve.attr("subplotcolspan", 1))
            ngraphs += rowspan * colspan
            isthereasubplot = True
        elif ngraphs == 0:
            ncurvesondefault += 1
            ngraphs = 1
    ngraphs = max(1, ngraphs)
    # transpose?
    transpose = graph.attr("subplotstranspose", False)
    # determine axes matrix shape
    ncols = int(graph.attr("subplotsncols", (1 if ngraphs < 2 else 2)))
    nrows = int(np.ceil(ngraphs / ncols))
    # width, heigth ratios?
    gridspeckwargs = {}
    val = list(graph.attr("subplotswidth_ratios", ""))
    if len(val) > 0:
        target = ncols if not transpose else nrows
        if len(val) != target:
            val += [1] * max(0, target - len(val))
            while len(val) > target:
                del val[-1]
            msg = (
                "_create_subplots_gridspec: corrected width_ratios to match ncols {"
                "}: {}."
            )
            print(msg.format(ncols, val))
        gridspeckwargs.update({"width_ratios": val})
    val = list(graph.attr("subplotsheight_ratios", ""))
    if len(val) > 0:
        target = nrows if not transpose else ncols
        if len(val) != target:
            val += [1] * max(0, target - len(val))
            while len(val) > target:
                del val[-1]
            msg = (
                "_create_subplots_gridspec: corrected height_ratios to match nrows "
                "{}: {}."
            )
            print(msg.format(nrows, val))
        gridspeckwargs.update({"height_ratios": val})
    # generate axes matrix: either gs, either ax is created
    gs, matrix = None, np.array([[]])
    if ax is None:
        if ngraphs == 1 and not isthereasubplot:
            ax = fig.add_subplot(111)
            ax.ticklabel_format(useOffset=False)
            ax.patch.set_alpha(0.0)
            # hide axis if no Curve will be plotted on it
            if ncurvesondefault == 0:
                ax.axis("off")
        else:
            matrix = np.ones((nrows, ncols)) * (-1)
            if transpose:
                gs = GridSpec(ncols, nrows, **gridspeckwargs)
            else:
                gs = GridSpec(nrows, ncols, **gridspeckwargs)

    # coordinates of the plot id
    subplotsid = graph.attr("subplotsid", False)
    if subplotsid is not False and (
        not isinstance(subplotsid, (list, tuple)) or len(subplotsid) != 2
    ):
        subplotsid = (-0.03, 0.00)
    # misc adjustments to self
    if ngraphs > 1:  # do not want default values if multiple subplots
        graph.update({"xlabel": "", "ylabel": ""})
    # return
    return [fig, ax], gs, matrix, subplotsid, ngraphs


def _create_new_axis(
    graph, curve, fig, gs, matrix, subplotsid, subplotscounter, subplotidkwargs
):
    rowspan = int(curve.attr("subplotrowspan", 1))
    colspan = int(curve.attr("subplotcolspan", 1))
    gspos = gs.get_grid_positions(fig)
    flag = False
    ax = None
    for i_ in range(len(matrix)):
        for j_ in range(len(matrix[i_])):
            if matrix[i_, j_] == -1:
                # first free spot found: start there
                if rowspan > matrix.shape[0] - i_:
                    # calculation of number of rows probably faulty
                    msg = (
                        "WARNING plot: rowspan larger than can handled "
                        "for subplot {}, value coerced to {}."
                    )
                    print(msg.format(subplotscounter, matrix.shape[0] - i_))
                if colspan > matrix.shape[1] - j_:
                    # either faulty calculation of number of rows,
                    # either that colspan cannot fit in ncols
                    msg = (
                        "WARNING plot: colspan larger than can handled "
                        "for subplot {}, value coerced to {}."
                    )
                    print(msg.format(subplotscounter, matrix.shape[1] - j_))
                rowspan = min(rowspan, matrix.shape[0] - i_)
                colspan = min(colspan, matrix.shape[1] - j_)
                if graph.attr("subplotstranspose", False):
                    if subplotsid:
                        coords = (
                            gspos[2][i_] + subplotsid[0],
                            gspos[1][j_] + subplotsid[1],
                        )
                        txt = "(" + chr(ord("a") + subplotscounter) + ")"
                        graph.text_add(txt, coords, subplotidkwargs)
                    # ax = plt.subplot(gs[j_ : (j_ + colspan), i_ : (i_ + rowspan)])
                    ax = fig.add_subplot(gs[j_ : (j_ + colspan), i_ : (i_ + rowspan)])
                else:
                    if subplotsid:
                        coords = (
                            gspos[2][j_] + subplotsid[0],
                            gspos[1][i_] + subplotsid[1],
                        )
                        txt = "(" + chr(ord("a") + subplotscounter) + ")"
                        graph.text_add(txt, coords, subplotidkwargs)
                    # ax = plt.subplot(gs[i_ : (i_ + rowspan), j_ : (j_ + colspan)])
                    ax = fig.add_subplot(gs[i_ : (i_ + rowspan), j_ : (j_ + colspan)])
                matrix[i_ : (i_ + rowspan), j_ : (j_ + colspan)] = subplotscounter
                flag = True
                break
        if flag:
            break
    if ax is None:
        logger.error("plot _create_new_axis: ax is None. matrix {}.".format(matrix))
        raise Exception
    ax.ticklabel_format(useOffset=False)
    ax.patch.set_alpha(0.0)
    return ax


class ScaErrorHandler:
    """try plt.sca, handles possible matplotlib errors"""

    _PLOT_PRINTEDERROR_ATTRIBUTE = False
    _PLOT_PRINTEDERROR_VALUE = False

    @classmethod
    def attempt(cls, ax, _txt=""):
        """try plt.sca, handles possible matplotlib errors"""
        try:
            plt.sca(ax)
            # actually, show avoid using pyplot together with tkagg !!!
        except AttributeError:
            if not cls._PLOT_PRINTEDERROR_ATTRIBUTE:
                # print('WARNING sca(ax)', txt, '. AttributeError caught, cause to
                # investigate...')
                cls._PLOT_PRINTEDERROR_ATTRIBUTE = True
        except ValueError:
            if not cls._PLOT_PRINTEDERROR_VALUE:
                # print('WARNING sca(ax)', txt, '. ValueError caught, cause to
                # investigate...')
                cls._PLOT_PRINTEDERROR_VALUE = True


def _plot_set_axis_scale(typeplot, gs, ax):
    """Set default graph scales (lin, log, etc.)"""
    if typeplot != "" and gs is None:
        xarg = "log" if typeplot in ["semilogx", "loglog"] else "linear"
        try:
            ax.set_xscale(xarg)
        except (ValueError, AttributeError):
            msg = "_plot_set_axis_scale x: try no data transform? ({}, {})"
            logger.warning(msg.format(typeplot, xarg))
        yarg = "log" if typeplot in ["semilogy", "loglog"] else "linear"
        try:
            ax.set_yscale(yarg)
        except (ValueError, AttributeError):
            msg = "_plot_set_axis_scale y: try no data transform? ({}, {})"
            logger.warning(msg.format(typeplot, yarg))


def _plot_add_legend(graph, ax, handles):
    """Add a legend to the ax, using the list of handles, and formatting instructions in
    graph"""
    leg_prop_user = deepcopy(graph.attr("legendproperties", default="best"))
    if not isinstance(leg_prop_user, dict):
        leg_prop_user = {"loc": leg_prop_user}
    if "loc" in leg_prop_user:
        leg_prop_user["loc"] = str(leg_prop_user["loc"]).lower()
        rep = {
            "best": 0,
            "ne": 1,
            "nw": 2,
            "sw": 3,
            "se": 4,
            "right": 5,
            "w": 6,
            "e": 7,
            "s": 8,
            "n": 9,
            "center": 10,
        }
        if leg_prop_user["loc"] in rep:
            leg_prop_user["loc"] = rep[leg_prop_user["loc"]]
    prop = {}
    if "fontsize" in graph.graphinfo:
        prop = {"size": graph.graphinfo["fontsize"]}
    if "prop" in leg_prop_user:
        prop.update(leg_prop_user["prop"])
    leg_prop_user["prop"] = prop
    if "fontsize" in leg_prop_user:
        leg_prop_user["prop"].update({"size": leg_prop_user["fontsize"]})
        del leg_prop_user["fontsize"]
    leg_label_color = None
    if "color" in leg_prop_user:
        leg_label_color = leg_prop_user["color"].lower()
        del leg_prop_user["color"]
    leg_prop = {
        "fancybox": True,
        "framealpha": 0.5,
        "frameon": False,
        "numpoints": 1,
        "scatterpoints": 1,
    }
    leg_prop.update(leg_prop_user)
    labels = []
    for i in range(len(handles) - 1, -1, -1):
        label = (
            handles[i]["handle"].get_label()
            if hasattr(handles[i]["handle"], "get_label")
            else ""
        )  # not legend in case don't know how to find it
        if isinstance(handles[i]["handle"], mpl.image.AxesImage):
            label = None
        if label is None or len(label) == 0 or label.startswith("_"):
            del handles[i]  # delete curve handle if no label is to be shown
        else:
            labels.append(label)
    # labels is reversed by construction
    try:
        leg = ax.legend([h["handle"] for h in handles], labels[::-1], **leg_prop)
    except ValueError as e:
        logger.error("_plot_add_legend: ValueError {}".format(e))
        return None

    # color of legend
    if leg_label_color is not None:
        if leg_label_color == "curve":
            # special value: same color for text and lines
            lines, texts = leg.get_lines(), leg.get_texts()
            if len(texts) > len(lines):  # issue with errorbar, notably
                lines = []
                for h in handles:
                    if isinstance(h["handle"], tuple):
                        lines.append(h["handle"][0])
                    else:
                        lines.append(h["handle"])
            for line, text in zip(lines, texts):
                try:
                    text.set_color(line.get_color())
                except Exception:
                    pass
        else:
            for text in leg.get_texts():
                text.set_color(leg_label_color)
    # legend title
    leg_title = deepcopy(graph.attr("legendtitle"))
    # legendTitle can be of the form ['some title', {'fontsize':24}]
    if leg_title != "":
        setfunc = []
        if isinstance(leg_title, list):
            for key in ["color", "position", "fontsize", "align"]:
                if key in leg_title[1]:
                    setfunc.append([key, leg_title[1][key]])
                    del leg_title[1][key]
            prop.update(leg_title[1])
            leg_title = leg_title[0]
        leg.set_title(leg_title, prop=prop)
        for setf in setfunc:
            if setf[0] == "align":
                leg._legend_box.align = setf[1]
            else:
                if hasattr(leg.get_title(), "set_" + setf[0]):
                    getattr(leg.get_title(), "set_" + setf[0])(setf[1])
                else:
                    msg = "_plot_add_legend: what to do with keyword {} ?"
                    logger.warning(msg.format(setf[0]))
    # in legend, set color of scatter correctly
    for i, handle in enumerate(handles):
        if "setlegendcolormap" in handle and handle["setlegendcolormap"]:
            try:
                leg.legendHandles[i].set_color(handle["handle"].get_cmap()(0.5))
                # plt.cm.afmhot(.5))
            except AttributeError:
                logger.error("error setlegendcolormap", exc_info=True)
    return leg


def _plot_apply_arbitrary_functions(graph, ax, fig):
    """Apply arbitrary functions to the ax of figure
    Keyword arbitraryfunctions:
    list[list["attr.attr...", list[argval0, ...], dict[kw0: kwval0, ...]], ...]
    The functions are accessed though ax if nothing seppcifed, or"""
    if "arbitraryfunctions" not in graph.graphinfo:
        return

    msg_error = "Exception in function Graph.plot arbitrary functions."
    special_locators_formatters = [
        "set_major_locator",
        "set_minor_locator",
        "set_major_formatter",
        "set_minor_formatter",
    ]
    for fun in graph.graphinfo["arbitraryfunctions"]:
        # print ('Graph plot arbitrary func', fun, type(fun))
        f, arg, opt = fun[0], fun[1], fun[2]
        fsplit = f.split(".")
        # print ('   ', ax, fsplit, len(fsplit))
        # first-level
        obj = ax
        if len(fsplit) > 0 and fsplit[0] == "fig":
            obj = fig
            del fsplit[0]
        # going down the chain of attributes/methods
        for subf in fsplit:
            if subf.startswith("_") or subf == "getattr":
                # To prevent possible security vulnerability.
                # Not sure if it is "safe".
                msg = (
                    "arbitraryfunctions does not accept functions or "
                    "attributes starting with character '_'. {}"
                )
                logger.error(msg.format(fsplit))
                raise RuntimeError(msg.format(fsplit))
            try:
                if hasattr(obj, "__call__"):
                    obj = getattr(obj(), subf)
                else:
                    obj = getattr(obj, subf)
            except AttributeError:
                logger.error(msg_error, exc_info=True)
                continue
        # print ('   ', obj, type(obj))

        # handle ticks locators and formatters, take objects as arguments
        if fsplit[-1] in special_locators_formatters:
            if len(arg) > 0 and isinstance(arg[0], str):
                try:
                    asplit = resplit("[()]", arg[0])
                    args = [strToVar(a) for a in asplit[1:] if a != ""]
                    arg = [getattr(tck, asplit[0])(*args)]
                except Exception:
                    pass  # just continue with unmodified arg
        # execute desired function
        try:
            obj(*arg, **opt)
        except Exception:
            logger.error(msg_error, exc_info=True)


def _plot_add_text(graph: Graph, ax) -> None:
    """Add Annotations to ax, according to information stored within the Graph object"""
    if "text" not in graph.graphinfo:
        return

    textxy_default = (0.05, 0.95)
    texts, texys, args = TextHandler.check_valid(graph)
    for i in range(len(texts)):
        text, texy, arg = texts[i], texys[i], dict(args[i])
        if text != "":
            if texy != "" and texy is not None and texy not in [["", ""], ("", "")]:
                if "xytext" in arg:
                    msg = "Graph plot annotate: {} textxy {} replaces {}"
                    logger.warning(msg.format(text, texy, arg["xytext"]))
                arg.update({"xytext": texy})
            if "xytext" not in arg:
                if "xy" in arg:
                    arg.update({"xytext": arg["xy"]})
                else:
                    arg.update({"xytext": textxy_default})
            if "xy" not in arg:
                arg.update({"xy": arg["xytext"]})
            if "xycoords" not in arg and "textcoords" in arg:
                arg.update({"xycoords": arg["textcoords"]})
            if "xycoords" in arg and "textcoords" not in arg:
                arg.update({"textcoords": arg["xycoords"]})
            if "textcoords" not in arg:
                arg.update({"textcoords": "figure fraction"})
            if "xycoords" not in arg:
                arg.update({"xycoords": "figure fraction"})
            if "fontsize" not in arg and "fontsize" in graph.graphinfo:
                arg.update({"fontsize": graph.graphinfo["fontsize"]})
            # set_clip_box to draw all that can be shown, to couple with
            # 'annotation_clip'=False
            arrow_set_clip_box = False
            if "arrowprops" in arg and "set_clip_box" in arg["arrowprops"]:
                arrow_set_clip_box = arg["arrowprops"]["set_clip_box"]
                del arg["arrowprops"]["set_clip_box"]
            # print("Graph plot annotate", text, "args", arg)
            try:
                ann = ax.annotate(text, **arg)
                if arrow_set_clip_box:
                    ann.arrow_patch.set_clip_box(ax.bbox)
            except Exception:
                msg = "Exception during ax.annotate _plot_add_text"
                logger.error(msg, exc_info=True)


def _plot_change_fontsize(fontsize, fontsizeset, ax_and_twins):
    """Change the fontsize on the different textual elements of the graph,
    except for these listed in fontsizeset which should have received ad-hoc fontsize"""
    ax, ax_twinx, ax_twiny, ax_twinxy = ax_and_twins
    list_labels = [ax.title, ax.xaxis.label, ax.yaxis.label]
    if "ax.get_xticklabels()" not in fontsizeset:
        list_labels += ax.get_xticklabels()
    if "ax.get_yticklabels()" not in fontsizeset:
        list_labels += ax.get_yticklabels()
    # maybe an automatic detection of all the axis would be more robust
    # than trying to list all possible existing axis?
    if ax_twinx is not None:
        list_labels += [ax_twinx.yaxis.label] + ax_twinx.get_yticklabels()
    if ax_twiny is not None:
        list_labels += [ax_twiny.xaxis.label] + ax_twiny.get_xticklabels()
    if ax_twinxy is not None:
        list_labels += (
            [ax_twinxy.xaxis.label, ax_twinxy.yaxis.label]
            + ax_twinxy.get_xticklabels()
            + ax_twinxy.get_yticklabels()
        )
    for item in list_labels:
        if item not in fontsizeset:
            item.set_fontsize(fontsize)


def _plot_add_title_labels_ticklabels(graph, fontsizeset, gs, ax_and_twins):
    """add title, xlabel, ylabel, their coordinates, xtickslabels, ytickslabels
    onto ax.
    Note: xtickslabels, ytickslabels must take place be before xlim and ylim"""
    ax, ax_twinx, ax_twiny, ax_twinxy = ax_and_twins
    if "title" in graph.graphinfo:
        out = ParserMisc.set_axislabel(ax.set_title, graph.graphinfo["title"], graph)
        if out["size"]:
            fontsizeset.append(ax.title)
    if "xlabel" in graph.graphinfo and gs is None:
        # gs check because xlabel could be automatically set upon file parsing
        out = ParserMisc.set_axislabel(ax.set_xlabel, graph.graphinfo["xlabel"], graph)
        if out["size"]:
            fontsizeset.append(ax.xaxis.label)
    if "ylabel" in graph.graphinfo and gs is None:
        # gs check because ylabel could be automatically set upon file parsing
        out = ParserMisc.set_axislabel(ax.set_ylabel, graph.graphinfo["ylabel"], graph)
        if out["size"]:
            fontsizeset.append(ax.yaxis.label)
    # labels twin axis
    if "twinx_ylabel" in graph.graphinfo:
        if ax_twinx is not None:
            val = graph.graphinfo["twinx_ylabel"]
            out = ParserMisc.set_axislabel(ax_twinx.set_ylabel, val, graph)
            if out["size"]:
                fontsizeset.append(ax_twinx.yaxis.label)
        elif ax_twinxy is not None:
            val = graph.graphinfo["twinx_ylabel"]
            out = ParserMisc.set_axislabel(ax_twinxy.set_ylabel, val, graph)
            if out["size"]:
                fontsizeset.append(ax_twinxy.yaxis.label)
    if "twiny_xlabel" in graph.graphinfo:
        if ax_twiny is not None:
            val = graph.graphinfo["twiny_xlabel"]
            out = ParserMisc.set_axislabel(ax_twiny.set_xlabel, val, graph)
            if out["size"]:
                fontsizeset.append(ax_twiny.xaxis.label)
        elif ax_twinxy is not None:
            val = graph.graphinfo["twiny_xlabel"]
            out = ParserMisc.set_axislabel(ax_twinxy.set_xlabel, val, graph)
            if out["size"]:
                fontsizeset.append(ax_twinxy.xaxis.label)

    # lcoation of labels
    if "xlabel_coords" in graph.graphinfo:
        val = graph.graphinfo["xlabel_coords"]
        if isinstance(val, list):
            ax.xaxis.set_label_coords(val[0], val[1])
        else:
            ax.xaxis.set_label_coords(0.5, val)
    if "ylabel_coords" in graph.graphinfo:
        val = graph.graphinfo["ylabel_coords"]
        if isinstance(val, list):
            ax.yaxis.set_label_coords(val[0], val[1])
        else:
            ax.yaxis.set_label_coords(val, 0.5)

    # xlim, ylim. Start with xtickslabels as this guy would override xlim
    if "xtickslabels" in graph.graphinfo:
        val = graph.graphinfo["xtickslabels"]
        _ticklocs, kw = ParserMisc.set_xyticklabels(val, ax.xaxis)
        if "size" in kw:
            fontsizeset.append("ax.get_xticklabels()")
    if "ytickslabels" in graph.graphinfo:
        val = graph.graphinfo["ytickslabels"]
        _ticklocs, kw = ParserMisc.set_xyticklabels(val, ax.yaxis)
        if "size" in kw:
            fontsizeset.append("ax.get_yticklabels()")
    return fontsizeset


def _plot_set_xylims(graph, ax_and_twins, alter, curvedummy):
    """Set xlim and ylim onto the axes.
    Note: xlim must be after xtickslabels and ytickslabels, these would try to set
    axis limits by themselves"""
    ax, ax_twinx, ax_twiny, ax_twinxy = ax_and_twins
    aac = [alter, curvedummy]
    if "xlim" in graph.graphinfo:
        ParserMisc.alter_lim(ax, graph.graphinfo["xlim"], "x", *aac)
    if "ylim" in graph.graphinfo:
        ParserMisc.alter_lim(ax, graph.graphinfo["ylim"], "y", *aac)
    if "twinx_ylim" in graph.graphinfo:
        if ax_twinxy is not None:
            ParserMisc.alter_lim(ax_twinxy, graph.graphinfo["twinx_ylim"], "y", *aac)
        if ax_twinx is not None:
            ParserMisc.alter_lim(ax_twinx, graph.graphinfo["twinx_ylim"], "y", *aac)
    if "twiny_xlim" in graph.graphinfo:
        if ax_twinxy is not None:
            ParserMisc.alter_lim(ax_twinxy, graph.graphinfo["twiny_xlim"], "x", *aac)
        if ax_twiny is not None:
            ParserMisc.alter_lim(ax_twiny, graph.graphinfo["twiny_xlim"], "x", *aac)


def _plot_add_xytickssteps(graph, ax, ignore_xlim, ignore_ylim):
    """add xticksstep and yticksstep
    Note: must(or should?) be after xlim and ylim"""
    if "xticksstep" in graph.graphinfo and not ignore_xlim:
        val = graph.graphinfo["xticksstep"]
        ParserMisc.set_xytickstep(val, ax.xaxis, ax.get_xlim())
    if "yticksstep" in graph.graphinfo and not ignore_ylim:
        val = graph.graphinfo["yticksstep"]
        ParserMisc.set_xytickstep(val, ax.yaxis, ax.get_ylim())


def _plot_add_axhvlines(graph, ax, curvedummy, alter, type_plot):
    """add axhlines and axvlines onto ax
    curvedummy to deal with alter"""
    if "axhline" in graph.graphinfo:
        lst = ParserAxhline(graph.graphinfo["axhline"])
        lst.plot(ax.axhline, curvedummy, alter, type_plot)
    if "axvline" in graph.graphinfo:
        lst = ParserAxvline(graph.graphinfo["axvline"])
        lst.plot(ax.axvline, curvedummy, alter, type_plot)


def _plot_loop_over_curves(graph, container, ax, handles):
    """A loop over the Curves within a Graph to plot them onto axes
    A few objects within container get modified during execution: ax_twinx, ax_twiny,
    ax_twinxy, groupedplotters
    """
    # clean-up, definition
    if handles is None or not isinstance(handles, list):
        handles = []
    subplot_adjust_def = SubplotsAdjuster.default()
    subplot_colorbar = [
        0.90,
        subplot_adjust_def["bottom"],
        0.05,
        subplot_adjust_def["top"] - subplot_adjust_def["bottom"],
    ]

    # variable retrieval
    fig = container["fig"]
    gs = container["gs"]
    matrix = container["matrix"]
    subplotsid = container["subplotsid"]
    fontsize = container["fontsize"]
    ax_twinx = container["ax_twinx"]
    ax_twiny = container["ax_twiny"]
    ax_twinxy = container["ax_twinxy"]
    groupedplotters = container["groupedplotters"]

    subplotidkwargs = {
        "textcoords": "figure fraction",
        "fontsize": fontsize,
        "horizontalalignment": "right",
        "verticalalignment": "top",
    }
    kw_graphsubplot = {"if_save": False, "if_export": False, "if_subplot": True}

    # prepare output and loop variables
    ax_or_one_of_its_twins = ax
    axes = []  # contains only main ax, not twins
    if ax is not None:
        axes.append(ax)
    colorbar_ax = []
    pairs_ax_curve = []
    subplotscounter = 0
    ignore_next = 0
    # auxiliary graph, to be created when new axis, filled with curves and
    # plotted when new axis is created
    graph_aux = None  # auxiliary Graph, calling plot() when creating new axis
    graph_aux_kw = {"config": graph.config_all()["filename"]}
    # start loop
    for curve_i, curve in enumerate(graph):
        if ignore_next > 0:
            ignore_next -= 1
            continue  # ignore this curve, go the next one...
            # for example if previous curve type was scatter
        if not curve.visible():
            continue  # go to next curve

        handle = None
        attr = curve.get_attributes()

        # Inset: if curve contains information for an inset in the Graph?
        if isinstance(curve, Curve_Inset):
            val = curve.attr("insetfile")
            inset = Graph(graph.filenamewithpath(val), **graph_aux_kw)
            coords = (
                list(attr["insetcoords"])
                if "insetcoords" in attr
                else [0.3, 0.2, 0.4, 0.3]
            )
            if "insetupdate" in attr and isinstance(attr["insetupdate"], dict):
                inset.update(attr["insetupdate"])
            ax_inset = fig.add_axes(coords)
            ax_inset.ticklabel_format(useOffset=False)
            ax_inset.patch.set_alpha(0.0)
            if curve.attr("insetfile") in ["", " "] or len(inset) == 0:
                # nothing in provided Graph -> created axis becomes active one
                if graph_aux is not None:
                    # if there was already auxiliary graph: display it, create anew
                    plot_graph(graph_aux, fig_ax=[fig, ax], **kw_graphsubplot)
                graph_aux = inset
                ax = ax_inset
                ScaErrorHandler.attempt(ax)
                ax_twinx, ax_twiny, ax_twinxy = None, None, None
            else:
                # found a Graph, place it in inset. Next Curve in existing
                # axes. No change for graph_aux
                plot_graph(inset, fig_ax=[fig, ax_inset], **kw_graphsubplot)
                continue  # go to next Curve

        # Subplots: if more than 1 subplot is expected
        if gs is not None:
            # if required, create the new axis
            if ax is None or isinstance(curve, Curve_Subplot):
                if graph_aux is not None:
                    plot_graph(graph_aux, fig_ax=[fig, ax], **kw_graphsubplot)
                    graph_aux = None
                ax = _create_new_axis(
                    graph,
                    curve,
                    fig,
                    gs,
                    matrix,
                    subplotsid,
                    subplotscounter,
                    subplotidkwargs,
                )
                ScaErrorHandler.attempt(ax)
                ax_twinx, ax_twiny, ax_twinxy = None, None, None
                subplotscounter += 1
                axes.append(ax)
            else:  # we go on, not enough information to create subplot
                pass

        # shall we plot a Graph object instead of a Curve in this new axis?
        if isinstance(curve, Curve_Subplot):
            # new axes, so create new graph_aux (is None by construction)
            val = curve.attr("subplotfile")
            if val not in [" ", "", None]:
                graph_aux = Graph(graph.filenamewithpath(val), **graph_aux_kw)
            else:
                graph_aux = Graph("", **graph_aux_kw)
            upd = {"subplots_adjust": ""}
            if isinstance(curve.attr("subplotupdate"), dict):
                upd.update(curve.attr("subplotupdate"))
            graph_aux.update(upd)
            if val not in [" ", "", None]:
                continue  # go to next Curve
                # if no file then try to plot data of the Curve

        if graph_aux is not None:
            # if active axis is not main one, place Curve in graph_aux
            if isinstance(curve, (Curve_Inset, Curve_Subplot)):
                graph_aux.append(Curve(curve.data, curve.get_attributes()))
            else:
                graph_aux.append(curve)
            continue  # go to next Curve

        # Use usual axis. But maybe one of the twin axes
        ax_or_one_of_its_twins = ax
        if_twinx = curve.attr("ax_twinx")  # if True
        if_twiny = curve.attr("ax_twiny")
        if if_twinx and if_twiny:
            if ax_twinxy is None:
                ax_twinxy = ax.twinx().twiny()
            ax_or_one_of_its_twins = ax_twinxy
        elif if_twinx:
            if ax_twinx is None:
                ax_twinx = ax.twinx()
            ax_or_one_of_its_twins = ax_twinx
        elif if_twiny:
            if ax_twiny is None:
                ax_twiny = ax.twiny()
            ax_or_one_of_its_twins = ax_twiny

        # do the actual plotting
        try:
            handle, ignore_next = plot_curve(
                ax_or_one_of_its_twins,
                groupedplotters,
                (graph, curve_i, curve),
                ignore_next=ignore_next,
            )
        except Exception:
            # need to catch exceptions as we want to proceed with graph
            msg = "Exception occured in Curve.plot(), Curve no {}."
            logger.error(msg.format(curve_i), exc_info=True)

        # remind pairs ax - curve
        pairs_ax_curve.append({"ax": ax_or_one_of_its_twins, "curve": curve})

        # Add colorbar if required
        if handle is not None and curve.attr("colorbar", False):
            kwargs = {}
            if isinstance(attr["colorbar"], dict):
                kwargs = deepcopy(attr["colorbar"])
            adjust = deepcopy(subplot_colorbar)
            if "adjust" in kwargs:
                adjust = kwargs["adjust"]
                del kwargs["adjust"]
            if isinstance(adjust[-1], str):
                if adjust[-1] == "ax":
                    # coordinates relative to the ax and not to figure
                    axbounds = ax_or_one_of_its_twins.get_position().bounds
                    adjust[0] = adjust[0] * axbounds[2] + axbounds[0]
                    adjust[1] = adjust[1] * axbounds[3] + axbounds[1]
                    adjust[2] = adjust[2] * axbounds[2]
                    adjust[3] = adjust[3] * axbounds[3]
                else:
                    logger.warning(
                        "WARNING plot colorbar: cannot interpret last keyword "
                        "(must be numeric, or 'ax')."
                    )
                del adjust[-1]
            if "labelsize" in kwargs:
                pass  # TODO
            colorbar_ax.append({"ax": fig.add_axes(adjust), "adjusted": False})
            try:
                colorbar_ax[-1]["cbar"] = fig.colorbar(
                    handle, cax=colorbar_ax[-1]["ax"], **kwargs
                )
            except AttributeError as e:
                msg = "_plot_loop_over_curves in fig.colorbar. {}, cax={}, kwargs {}). {}: {}."
                logger.error(
                    msg.format(handle, colorbar_ax[-1]["ax"], kwargs, type(e), e)
                )

            try:
                colorbar_ax[-1]["cbar"].solids.set_rasterized(True)
                colorbar_ax[-1]["cbar"].solids.set_edgecolor("face")
            except Exception:
                pass
            subplot_colorbar[0] -= 0.11  # default for next colorbar
            # set again ax_ as the current plt active ax, not the colorbar
            ScaErrorHandler.attempt(ax_or_one_of_its_twins, "ax_ after colorbar")

        # store handle in list
        if handle is not None:
            if isinstance(handle, list):
                for h in handle:
                    handles.append({"handle": h})
            else:
                handles.append({"handle": handle})
                if curve.attr("type") == "scatter":
                    handles[-1].update({"setlegendcolormap": True})
    # end loop over curves
    ax_and_twins = (ax, ax_twinx, ax_twiny, ax_twinxy)
    return ax_and_twins, ax_or_one_of_its_twins, axes, handles, graph_aux


def _plot_savefig(graph, fig, filename, img_format):
    """Saves a matplotlib figure in the required file format.
    Uses matplotlib fig.savefig"""
    save_dpi = 300 if "dpi" not in graph.graphinfo else graph.graphinfo["dpi"]

    if len(img_format) == 0:  # default file format
        img_format = graph.config("save_imgformat", ".png")

    if not isinstance(img_format, list):
        img_format = [img_format]
    for img_forma_ in img_format:
        img_format_target = ""
        if img_forma_ == ".emf":
            # special: we save svg and convert into emf using inkscape
            img_format_target = ".emf"
            img_forma_ = ".svg"

        filename_ = filename + img_forma_
        if not graph.attr("saveSilent"):
            print("Graph saved as " + filename_.replace("/", "\\"))

        try:
            fig.savefig(filename_, transparent=True, dpi=save_dpi)
        except PermissionError:
            msg = "PermissionError in _plot_savefig, filename_ {}."
            logger.error(msg.format(filename_))  # maybe just print?
            return

        graph.filename = filename_
        if img_format_target == ".emf":
            convert_svg_to_emf(graph, filename_, img_format, img_format_target)


def convert_svg_to_emf(graph, filename, img_format, img_format_target):
    """Converts a svg file into an emf"""
    success = False
    inkscapepath = graph.config("inkscape_path", [])
    if isinstance(inkscapepath, str):
        inkscapepath = [inkscapepath]
    inkscapepath += ["inkscape"]

    for p in inkscapepath:
        if not os.path.exists(p):
            continue  # cannot find inkscape executable
        try:
            fileemf = filename[: -len(img_format)] + img_format_target
            commandstr = '"{}" --without-gui --export-emf="{}" "{}"'
            command = commandstr.format(p, fileemf, filename)
            out = subprocess.call(command)
            if out == 0:
                print("Graph saved as " + fileemf.replace("/", "\\"))
                success = True
                break

            print("Graph save as .emf: likely error (return value:", out, ")")
        except Exception:
            logger.error("Exception during save in .emf format", exc_info=True)
    if not success:
        print("Could not save image in .emf format. Please check the following:")
        print(" - A version of inkscape is available,")
        print(" - file config.txt in grapa directory,")
        print(
            "- in file config.txt a line exists, similar as that, and indicate a "
            "valid inkscape executable e.g.: inkscape_path",
            r'["C:\Program Files\Inkscape\inkscape.exe"]',
        )


def image_to_clipboard(graph, folder=""):
    """copy the image output of a Graph to the clipboard - Windows only"""
    # save image, because we don't have pixel map at the moment
    print("Copying graph image to clipboard")
    selffilename = graph.filename if hasattr(graph, "filename") else None
    file_clipboard = "_grapatoclipboard"
    if len(folder) > 0:
        file_clipboard = os.path.join(folder, file_clipboard)
    tmp = graph.attr("saveSilent")
    graph.update({"saveSilent": True})
    graph.plot(if_save=True, if_export=False, filesave=file_clipboard)
    graph.update({"saveSilent": tmp})
    if selffilename is not None:  # restore self.filename
        graph.filename = selffilename
    from io import BytesIO
    from PIL import Image

    try:
        import win32clipboard
    except ImportError:
        msg = (
            "Module win32clipboard not found, cannot copy to clipboard. Image was "
            "created: {}. Try the following:\npip install pywin32"
        )
        logger.error(msg.format(file_clipboard), exc_info=True)
        return False

    def send_to_clipboard(clip_type, content):
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(clip_type, content)
        win32clipboard.CloseClipboard()

    # read image -> get pixel map, convert into clipboard-readable format
    output = BytesIO()
    img = Image.open(file_clipboard + ".png")

    # new version, try to copy png into clipboard.
    # Maybe people would complain, then revert to legacy
    img.save(output, "PNG")
    data = output.getvalue()
    output.close()
    send_to_clipboard(win32clipboard.RegisterClipboardFormat("PNG"), data)
    return True
