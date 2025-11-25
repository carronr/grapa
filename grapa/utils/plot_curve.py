"""functions to plot a curve onto and ax"""

import inspect
import logging
from dataclasses import dataclass
from typing import Tuple, Any, TYPE_CHECKING

from matplotlib.axes import Axes
import numpy as np

from grapa import KEYWORDS_CURVE
from grapa.colorscale import Colorscale
from grapa.curve_image import Curve_Image
from grapa.utils.plot_graph_aux import GroupedPlotters
from grapa.utils.error_management import IncorrectInputError, issue_warning

if TYPE_CHECKING:
    from grapa.graph import Graph
    from grapa.curve import Curve

logger = logging.getLogger(__name__)


@dataclass
class DataStruct:
    """To pass parameters around.

    :meta private
    """

    ax_x_y_fmt: Tuple[
        Axes, np.ndarray, np.ndarray, dict
    ]  # [matplotlib.axes.Axes, np.ndarray, np.ndarray, dict]
    alter: list
    plot_method: str
    graph: "Graph"
    graph_i: int
    curve: "Curve"


def plot_curve(
    ax: Axes,
    groupedplotters: GroupedPlotters,
    graph: "Graph",
    graph_i: int,
    curve: "Curve",
    ignore_next: int = 0,
):
    """
    Plot a Curve on some axis

    :param ax: a matplotlib.axes.Axes
    :param groupedplotters: to handle boxplots, violinplots and such (see
        plot_graph_aux.py)
    :param graph_i_curve: (graph, graph_i, curve), such that curve == graph[graph_i]
        Required to properly plot scatter with scatter_c, etc.
    :param ignore_next: int, counter to decide whether the next curves shall not be
        plotted (multi-Curve plotting such as scatter)
    :return: handle, ignore_next
    """
    handle = None
    # any reason to not plot anything?
    if curve.attr("curve") == "subplot":
        return handle, ignore_next
    if not curve.visible():
        return handle, ignore_next

    # check default arguments
    if graph is None:
        graph_i = None
    else:
        if graph[graph_i] != curve:
            graph_i = None
        if graph_i is None:
            try:
                graph_i = graph.index(curve)
            except ValueError:
                pass  # graph_i remains None
        if graph_i is None:
            graph = None  # curve was not found in graph
            msg = "Warning Curve.plot: Curve not found in provided Graph"
            issue_warning(logger, msg)

    # retrieve basic information, data after transform including of offset and muloffset
    attr = curve.get_attributes()
    linespec = curve.attr("linespec")
    alter = graph.get_alter() if graph is not None else ["", ""]
    x = curve.x_offsets(alter=alter[0])
    y = curve.y_offsets(alter=alter[1])
    graph_typeplot = graph.attr("typeplot")
    plot_method = curve.attr("type", "plot")
    if graph_typeplot.endswith(" norm."):
        # graph_typeplot = graph_typeplot[:-6]
        y = y / max(y)

    # Construct dict of keywords fmt based on curves attributes, in a restrictive way
    # Some attributes are commands for plotting, some are metadata, and no obvious way
    # to discriminate between the 2
    fmt = _build_fmt(attr, ax, plot_method)

    # Start plotting
    struct = DataStruct((ax, x, y, fmt), alter, plot_method, graph, graph_i, curve)

    # No support for the following methods (either 2D data, or complicated
    # to implement):
    #    hlines, vlines, broken_barh, polar,
    #    pcolor, pcolormesh, streamplot, tricontour, tricontourf,
    #    tripcolor
    # Partial support for:
    #    imgshow, contour, contourf (part of Curve_Image)
    handle, ignore_next, attr_ignore = _choose_plot_method(
        struct, attr, groupedplotters, linespec, ignore_next
    )

    _apply_attr_to_handle(handle, attr, fmt, attr_ignore)

    return handle, ignore_next


def _choose_plot_method(
    struct: DataStruct, attr, groupedplotters, linespec, ignore_next
):
    """Choose the appropriate matplotlib method and organize the call to it"""
    attr_ignore = [
        "label",
        "plot",
        "linespec",
        "type",
        "ax_twinx",
        "ax_twiny",
        "offset",
        "muloffset",
        "labelhide",
        "colorbar",
    ]

    (ax, x, y, fmt) = struct.ax_x_y_fmt
    curve = struct.curve
    plot_method = struct.plot_method
    handle = None

    # "simple" plotting methods, with prototype similar to plot()
    if plot_method in [
        "semilogx",
        "semilogy",
        "loglog",
        "plot_date",
        "stem",
        "step",
        "triplot",
    ]:
        handle = getattr(ax, plot_method)(x, y, linespec, **fmt)
    elif plot_method in ["fill"]:
        if curve.attr("fill_padto0", False):
            x = [x[0]] + list(x) + [x[-1]]
            y = [0] + list(y) + [0]
        handle = ax.fill(x, y, linespec, **fmt)
    # plotting methods not accepting formatting string as 3rd argument
    elif plot_method in [
        "barbs",
        "cohere",
        "csd",
        "hexbin",
        "hist2d",
        "quiver",
        "xcorr",
    ]:
        handle = getattr(ax, plot_method)(x, y, **fmt)
    elif plot_method in ["bar", "barh"]:
        handle = _plot_bar_barh(struct)
    elif plot_method in ["fill_between", "fill_betweenx"]:
        handle, ignore_next = _plot_fillbetweenetc(struct, ignore_next)
    #  plotting of single vector data
    elif plot_method in [
        "acorr",
        "angle_spectrum",
        "eventplot",
        "hist",
        "magnitude_spectrum",
        "phase_spectrum",
        "pie",
        "psd",
        "specgram",
    ]:
        # careful with eventplot, the Curve data are modified
        handle = getattr(ax, plot_method)(y, **fmt)
    # a more peculiar plotting
    elif plot_method in ["spy"]:
        handle = getattr(ax, plot_method)([x, y], **fmt)
    elif plot_method == "stackplot":
        handle, ignore_next = _plot_stackplot(struct, attr_ignore, ignore_next)
    elif plot_method == "errorbar":
        handle, ignore_next = _plot_errorbar(struct, attr, linespec, ignore_next)
    elif plot_method == "scatter":
        handle, ignore_next = _plot_scatter(struct, ignore_next)
    elif plot_method in ["boxplot", "violinplot"]:
        handle = groupedplotters.add_curve(plot_method, curve, y, fmt, ax)
    elif plot_method in ["imshow", "contour", "contourf"]:
        handle, ignore_next = _plot_imshowetc(struct, attr, ignore_next)
    else:
        # default is plot (lin-lin) # also valid if no information is
        # stored, aka returned ''
        try:
            handle = ax.plot(x, y, linespec, **fmt)
        except (TypeError, ValueError, AttributeError) as e:
            msg = "Exception _choose_plot_method ax.plot. linespec %s. fmt %s. %s: %s."
            msa = (linespec, fmt, type(e), e)
            logger.error(msg, *msa)
            raise IncorrectInputError(msg % msa) from None
    return handle, ignore_next, attr_ignore


def _apply_attr_to_handle(handle, attr: dict, fmt: dict, attr_ignore: list) -> None:
    """apply attr to handle, if not in fmt (already done) and not in attr_ignore"""
    handles = handle if isinstance(handle, list) else [handle]
    for key in attr:
        if key not in fmt and key not in attr_ignore:
            for h in handles:
                if hasattr(h, "set_" + key):
                    try:
                        getattr(h, "set_" + key)(attr[key])
                    except ValueError as e:
                        msg = "plot_curve _apply_attr_to_handle() .set_xxx, {}. {}"
                        issue_warning(logger, msg.format(key, e))
                    except Exception as e:
                        msg = "plot_curve _apply_attr_to_handle() .set_xxx, {}. {}: {}"
                        msgfull = msg.format(key, type(e), e)
                        issue_warning(logger, msgfull, exc_info=True)


def _build_fmt(attr: dict, ax: Axes, plot_method: str) -> dict:
    """internal details: construction of the fmt keyword used in ax.plot"""
    fmt = {}
    for key in attr:
        if not isinstance(key, str):
            msg = "TO CHECK THIS {} {} {}"
            issue_warning(logger, msg.format(type(key), key, attr[key]))
        keys_not_fmt = ["plot", "linespec", "type", "ax_twinx", "ax_twiny", "colorbar"]
        keys_not_fmt += ["offset", "muloffset", "labelhide", "xerr", "yerr"]
        if (
            (not isinstance(attr[key], str) or attr[key] != "")
            and key in KEYWORDS_CURVE["keys"]
            and key not in keys_not_fmt
        ):
            fmt[key] = attr[key]

    # some renaming of keywords, etc
    if "legend" in fmt:
        fmt["label"] = fmt["legend"]
        del fmt["legend"]
    if "cmap" in fmt and not isinstance(fmt["cmap"], str):
        # convert Colorscale into matplotlib cmap
        fmt["cmap"] = Colorscale(fmt["cmap"]).cmap()
    if "vminmax" in fmt:
        if isinstance(fmt["vminmax"], list) and len(fmt["vminmax"]) > 1:
            if (
                fmt["vminmax"][0] != ""
                and not np.isnan(fmt["vminmax"][0])
                and not np.isinf(fmt["vminmax"][0])
            ):
                fmt.update({"vmin": fmt["vminmax"][0]})
            if (
                fmt["vminmax"][1] != ""
                and not np.isnan(fmt["vminmax"][1])
                and not np.isinf(fmt["vminmax"][1])
            ):
                fmt.update({"vmax": fmt["vminmax"][1]})
        del fmt["vminmax"]

    # add keyword arguments which are in the plot method prototypes
    try:
        sig = inspect.signature(getattr(ax, plot_method))
        for key in sig.parameters:
            if key in attr and key not in fmt:
                fmt.update({key: attr[key]})
    except AttributeError:
        msg = "plot_curve: desired plotting method not found ({}). Going for default."
        issue_warning(logger, msg.format(plot_method))
        # for example 'errorbar_yerr' after suppression of previous Curve 'errorbar'.
        # To be 'plot' anyway.
    except Exception as e:
        msg = "Exception in plot_curve while identifying keyword arguments. %s. %s: %s."
        msa = (plot_method, type(e), e)
        logger.error(msg, *msa, exc_info=True)
        raise IncorrectInputError(msg % msa) from e

    if "labelhide" in attr and attr["labelhide"]:
        if "label" in fmt:
            del fmt["label"]
    return fmt


def _identify_curve_within_range_same_x(x_ref, range_, graph: "Graph", alter, where):
    """function used for bar, bah. Possibly others could make good use of it"""
    if graph is None:
        return None, None, None
    for j in range_:
        if graph[j].visible():
            flag = True
            for key_, value_ in where.items():
                if graph[j].attr(key_) != value_:
                    flag = False
            if not flag:
                continue
            x_j = graph[j].x_offsets(alter=alter[0])
            if np.array_equal(x_ref, x_j):
                y_j = graph[j].y_offsets(alter=alter[1])
                return j, x_j, y_j
    return None, None, None


def _plot_imshowetc(struct: DataStruct, attr: dict, ignore_next: int) -> tuple:
    """Plot imshow, contour, contourf"""
    (ax, _, _, fmt) = struct.ax_x_y_fmt
    (graph, graph_i, curve) = struct.graph, struct.graph_i, struct.curve
    alter, plot_method = struct.alter, struct.plot_method

    handle = None
    img, ignore_next, xdata, ydata = Curve_Image.get_image_data(
        curve, graph, graph_i, alter, ignore_next
    )
    if "label" in fmt:
        del fmt["label"]
    if plot_method in ["contour", "contourf"]:
        for key in [
            "corner_mask",
            "colors",
            "alpha",
            "cmap",
            "norm",
            "vmin",
            "vmax",
            "levels",
            "origin",
            "extent",
            "locator",
            "extend",
            "xunits",
            "yunits",
            "antialiased",
            "nchunk",
            "linewidths",
            "linestyles",
            "hatches",
        ]:
            if key in attr and key not in fmt:
                fmt.update({key: attr[key]})
        # TODO: remove linewidths, linestyles for contourf, hatches for contour
    args = [img]
    if (
        xdata is not None
        and ydata is not None
        and plot_method in ["contour", "contourf"]
    ):
        args = [xdata, ydata] + args
    try:
        handle = getattr(ax, plot_method)(*args, **fmt)
    except (TypeError, ValueError, IndexError) as e:
        msg = "Curve plot {}. {}: {}."
        msa = (plot_method, type(e), e)
        logger.error(msg, *msa, exc_info=True)
        raise IncorrectInputError(msg % msa) from e
    return handle, ignore_next


def _plot_scatter(struct: DataStruct, ignore_next: int) -> tuple:
    """Plot curve using scatter"""
    (ax, x, y, fmt) = struct.ax_x_y_fmt
    graph, graph_i = struct.graph, struct.graph_i
    alter = struct.alter

    handle = None
    convert = {"markersize": "s", "markeredgewidth": "linewidths"}
    for key, key_new in convert.items():
        if key in fmt:
            fmt.update({key_new: fmt[key]})
            del fmt[key]
    if graph is not None:
        for j in range(graph_i + 1, min(graph_i + 3, len(graph))):
            typenext = graph[j].attr("type")
            if typenext not in ["scatter_c", "scatter_s"]:
                break

            if "s" not in fmt and typenext == "scatter_s":
                fmt.update({"s": graph[j].y_offsets(alter=alter[1])})
                ignore_next += 1
                continue

            if "c" not in fmt and (
                typenext == "scatter_c"
                or np.array_equal(x, graph[j].x_offsets(alter=alter[0]))
            ):
                fmt.update({"c": graph[j].y_offsets(alter=alter[1])})
                ignore_next += 1
                if "color" in fmt:
                    del fmt["color"]  # there cannot be both c and color keywords
                continue
            break
    try:
        handle = ax.scatter(x, y, **fmt)
    except (ValueError, Exception) as e:  # Exception to remove once dummy-proof
        msg = "Exception occured in curve _plot_scatter during scatter. %s: %s"
        msa = (type(e), e)
        logger.error(msg, *msa)
        raise IncorrectInputError(msg % msa) from e
    return handle, ignore_next


def _plot_errorbar(
    struct: DataStruct, attr: dict, linespec: str, ignore_next: int
) -> tuple:
    """errorbar. look for next Curves, maybe xerr/yerr was provided"""
    (ax, x, y, fmt) = struct.ax_x_y_fmt
    graph, graph_i = struct.graph, struct.graph_i

    if "xerr" in attr:
        fmt.update({"xerr": attr["xerr"]})
    if "yerr" in attr:
        fmt.update({"yerr": attr["yerr"]})
    if graph is not None:
        for j in range(graph_i + 1, min(graph_i + 3, len(graph))):
            if len(graph[j].y()) == len(y):
                typenext = graph[j].attr("type")
                if typenext not in ["errorbar_xerr", "errorbar_yerr"]:
                    break
                if typenext == "errorbar_xerr":
                    fmt.update({"xerr": graph[j].y_offsets()})
                    ignore_next += 1
                    continue
                if typenext == "errorbar_yerr":
                    fmt.update({"yerr": graph[j].y_offsets()})
                    ignore_next += 1
                    continue
            break
    handle = ax.errorbar(x, y, fmt=linespec, **fmt)
    return handle, ignore_next


def _plot_stackplot(struct: DataStruct, attr_ignore: list, ignore_next: int):
    """Plot stackplot. look for next Curves with type == 'stackplot', and same x
    attr_ignore gets modified (also fmt but that is expected"""
    (ax, x, y, fmt) = struct.ax_x_y_fmt
    graph, graph_i, curve = struct.graph, struct.graph_i, struct.curve
    alter, plot_method = struct.alter, struct.plot_method

    nexty = []

    fmt["labels"], fmt["colors"] = [""], [""]
    if "label" in fmt:
        fmt["labels"] = ["" if curve.attr("labelhide") else fmt["label"]]
        del fmt["label"]
    if "color" in fmt:
        fmt["colors"] = [fmt["color"]]
        del fmt["color"]
    attr_ignore.append("color")
    if graph is not None:
        for j in range(graph_i + 1, len(graph)):
            if graph[j].attr("type") == plot_method and np.array_equal(
                x, graph[j].x_offsets(alter=alter[0])
            ):
                ignore_next += 1
                if graph[j].visible():
                    nexty.append(graph[j].y_offsets(alter=alter[1]))
                    lbl = graph[j].attr("label")
                    fmt["labels"].append("" if graph[j].attr("labelhide") else lbl)
                    fmt["colors"].append(graph[j].attr("color"))
                    continue
            else:
                break
    if np.all([(c == "") for c in fmt["colors"]]):
        del fmt["colors"]
    handle = ax.stackplot(x, y, *nexty, **fmt)
    return handle, ignore_next


def _plot_bar_barh(struct: DataStruct):
    """Plot bar and barh"""
    (ax, x, y, fmt) = struct.ax_x_y_fmt
    graph, graph_i, curve = struct.graph, struct.graph_i, struct.curve
    alter, plot_method = struct.alter, struct.plot_method

    if graph is not None:
        key = "bottom" if plot_method == "bar" else "left"
        value = curve.attr(key)
        range_ = None
        if value == "bar_first":
            range_ = range(graph_i)
        elif value == "bar_previous":
            range_ = range(graph_i - 1, -1, -1)
        elif value == "bar_next":
            range_ = range(graph_i + 1, len(graph))
        elif value == "bar_last":
            range_ = range(len(graph) - 1, graph_i, -1)
        if range_ is not None:
            argssamex = [graph, alter, {"type": plot_method}]
            j, _x2, y2 = _identify_curve_within_range_same_x(x, range_, *argssamex)
            if j is not None:
                fmt.update({key: y2})
            else:
                msg = "plot_curve {}: no suitable Curve found ({}, {}, {})"
                issue_warning(logger, msg.format(plot_method, graph_i, key, value))
    handle = getattr(ax, plot_method)(x, y, **fmt)
    return handle


def _plot_fillbetweenetc(struct: DataStruct, ignore_next: int) -> tuple:
    """plot fill_between and fill_betweenx"""
    (ax, x, y, fmt) = struct.ax_x_y_fmt
    graph, graph_i = struct.graph, struct.graph_i
    alter, plot_method = struct.alter, struct.plot_method

    handle = None
    success = False
    if graph is not None and len(graph) > graph_i + 1:
        x2 = graph[graph_i + 1].x_offsets(alter=alter[0])
        y2 = graph[graph_i + 1].y_offsets(alter=alter[1])
        if not np.array_equal(x, x2):
            msg = (
                "Curve {} and {}: fill_between, fill_betweenx: x series must be "
                "equal. Fill to 0."
            )
            issue_warning(logger, msg.format(graph_i, graph_i + 1))
        else:
            ignore_next += 1
            success = True
            handle = getattr(ax, plot_method)(x, y, y2, **fmt)
    if not success:
        handle = getattr(ax, plot_method)(x, y, **fmt)
    return handle, ignore_next
