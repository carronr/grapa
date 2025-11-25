"""
Provides a script that performs correlation analysis on a data input.
possibly, SCAPS 2-D parameter sweeps
"""

import os
import sys
import copy
import fnmatch
import itertools
import logging
from typing import Optional, Union, List

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.datatypes.graphScaps import GraphScaps
from grapa.colorscale import Color
from grapa.utils.error_management import issue_warning

logger = logging.getLogger(__name__)


class CONF:
    """Configuration constants"""

    AS_DATATABLE = "AS_DATATABLE"
    AS_SCAPS = "AS_SCAPS"
    AUTO = "AUTO"

    HIGHER = ">"
    LOWER = "<"

    pltclose = True


def _writefile_datatable(filename, pkeys, pvals):
    """Write the content of the data matix into a text file"""
    with open(filename, "w") as file:
        file.write("\t".join(pkeys) + "\n")
        data = np.transpose(np.array(pvals))
        for line in data:
            file.write("\t".join([str(li) for li in line]) + "\n")


def plot_parameters_1d(pkeys, pvals, new_graph_kwargs={}):
    """array of normal plots"""
    xlim = values_to_lim(pvals[0])
    graphaux = Graph(**new_graph_kwargs)
    graphaux.append(Curve([[0], [0]], {}))
    graphaux.castCurve("subplot", 0, silentSuccess=True)
    graphaux[0].update({"subplotupdate": {"xlabel": pkeys[0], "xlim": xlim}})
    graphshow = Graph(**new_graph_kwargs)
    for i in range(1, pvals.shape[0]):
        graphshow.append(copy.deepcopy(graphaux[0]))
        spu = graphshow[-1].attr("subplotupdate")
        spu.update({"title": pkeys[i]})
        graphshow[-1].update({"subplotupdate": spu})
        graphshow.append(Curve([pvals[0], pvals[i, :]], {"label": pkeys[i]}))
    graphshow.update(
        {"subplots_adjust": [0.1, 0.1, 0.9, 0.9, 0.5, 0.5], "figsize": [8, 8]}
    )
    return graphshow


def plot_parameters_2d(
    pkeys, pvals, idx_pattern=None, idx_labels=None, new_graph_kwargs={}
):
    """array of scatter plots"""
    if idx_pattern is None:
        idx_pattern = [0, 1]
    if idx_labels is None:
        idx_labels = range(2, len(pkeys))

    # preparatory works
    islog = [guess_is_logarithm_quantity(pvals[i, :]) for i in idx_pattern]
    TYPEPLOTS = {0: "", 1: "semilogy", 10: "semilogx", 11: "loglog"}
    typeplot = TYPEPLOTS[islog[0] * 10 + islog[1]]

    s_a = [0.1, 0.1, 0.9, 0.9, 0.5, 0.5]
    figsize = [8, 8]
    spanx = figsize[0] * (s_a[2] - s_a[0]) / (2 * 1 + 1 * s_a[4]) * 72  # in points
    npntsx = np.sqrt(pvals.shape[1]) * 1.5  # *2 safety margin, symbols not too large
    markersize = (spanx / npntsx) ** 2

    graphaux = Graph(**new_graph_kwargs)
    dataaux = [[1], [1]]
    graphaux.append(Curve(dataaux, {}))
    attr0 = {
        "type": "scatter",
        "marker": "o",
        "markersize": markersize,
        "colorbar": {"adjust": [1.00, 0, 0.05, 1, "ax"]},
    }
    x0, x1 = pvals[idx_pattern[0], :], pvals[idx_pattern[1], :]
    graphaux.append(Curve([x0, x1], attr0))
    graphaux.castCurve("subplot", 0, silentSuccess=True)
    xlim = values_to_lim(x0, islog[0])
    ylim = values_to_lim(x1, islog[1])
    su = {
        "xlabel": pkeys[idx_pattern[0]],
        "ylabel": pkeys[idx_pattern[1]],
        "xlim": xlim,
        "ylim": ylim,
    }
    if typeplot != "":
        su.update({"typeplot": typeplot})
    graphaux[0].update({"subplotupdate": su})

    graph = Graph(**new_graph_kwargs)
    flag_onlynan = True
    for i in idx_labels:
        values, key = pvals[i, :], pkeys[i]
        if pkeys[i] in ["FF", "eta"]:
            values = np.array(values) * 100
            key = key + " (%)"
        graph.append(copy.deepcopy(graphaux[0]))
        # text annotations
        text, textxy, textargs = [], [], []
        colorthres = (np.min(values) + np.max(values)) / 2
        for j in range(len(x0)):
            color = "k" if values[j] > colorthres else "w"  # [0.5,0.5,0.5]
            text.append("{:#.3g}".format(values[j]))
            textxy.append("")
            textargs.append(
                {
                    "xytext": (x0[j], x1[j]),
                    "textcoords": "data",
                    "verticalalignment": "center",
                    "horizontalalignment": "center",
                    "fontsize": 8,
                    "color": color,
                }
            )  # , 'zorder':0
        spu = graph[-1].attr("subplotupdate")
        spu.update({"title": key, "text": text, "textxy": textxy, "textargs": textargs})
        graph[-1].update({"subplotupdate": spu})
        graph.append(graphaux[1])
        # add data
        graph.append(Curve([x0, values], {"type": "scatter_c", "label": key}))
        if not np.isnan(values).all():
            flag_onlynan = False
    graph.update({"subplots_adjust": s_a, "figsize": figsize})
    if flag_onlynan:
        return None
    return graph


def plot_correlations(
    pkeys,
    pvals,
    idx_pattern=None,
    idx_labels=None,
    groupbyparam=True,
    new_graph_kwargs={},
):
    """
    seriesX, seriesy: tuple of column to consider
    groupbyparam: if True, organize data as series where possible
    """
    if idx_pattern is None:
        idx_pattern = range(pvals.shape[0])
    if idx_labels is None:
        idx_labels = range(pvals.shape[1])

    # preparatory works
    islogx = [guess_is_logarithm_quantity(pvals[i, :]) for i in idx_pattern]
    islogy = [guess_is_logarithm_quantity(pvals[j, :]) for j in idx_labels]
    xlim = []
    for k in range(len(idx_pattern)):
        d = pvals[idx_pattern[k], :]
        lim = list(np.exp(values_to_lim(np.log(d)))) if islogx[k] else values_to_lim(d)
        xlim.append(lim)
    ylim = []
    for k in range(len(idx_labels)):
        d = pvals[idx_labels[k], :]
        lim = list(np.exp(values_to_lim(np.log(d)))) if islogy[k] else values_to_lim(d)
        ylim.append(lim)
    # axis label & ticks removval. arbitraryfunc.: matplotlib bug with minor tick labels
    sburemoverx = {
        "xtickslabels": [None, []],
        "xlabel": "",
        "arbitraryfunctions": [["xaxis.set_minor_formatter", ["NullFormatter()"], {}]],
    }
    sburemovery = {
        "ytickslabels": [None, []],
        "ylabel": "",
        "arbitraryfunctions": [["yaxis.set_minor_formatter", ["NullFormatter()"], {}]],
    }

    lenx, leny = len(idx_pattern), len(idx_labels)
    graph = Graph(**new_graph_kwargs)
    plts = [3 * (lenx + (lenx - 1) * 0.0), 3 * (leny + (leny - 1) * 0.0)]
    fs = [plts[0] + 1 + 0.5, plts[1] + 1 + 0.5]
    sa = [1 / fs[0], 1 / fs[0], 1 - 0.5 / fs[0], 1 - 0.5 / fs[1], 0, 0]
    graph.update({"subplotsncols": lenx, "figsize": fs, "subplots_adjust": sa})
    pearson = np.zeros((lenx, leny, 2))
    pearson.fill(np.nan)
    for j_, j in enumerate(idx_labels):
        pvalsj = pvals[j, :]
        if np.isnan(pvalsj).all():
            continue  # no point to proceed if only nan in data

        for i_, i in enumerate(idx_pattern):
            pvalsi = np.array(pvals[i, :])
            # Curve subplot
            attri = {
                "label": "subplot {}; {}".format(pkeys[i], pkeys[j]),
                "labelhide": 1,
            }
            graph.append(Curve([[0], [0]], attri))
            graph.castCurve("subplot", -1, silentSuccess=True)

            # curve with data
            sbu = {"xlabel": pkeys[i], "ylabel": pkeys[j]}
            sbu.update({"xlim": xlim[i_], "ylim": ylim[j_]})
            # remove axis labels, except for end graph.
            # NB: grapa not yet compatible with sharex and sharey axes
            if i != idx_pattern[0]:
                sbu.update(sburemovery)
            if j != idx_labels[-1]:
                sbu.update(sburemoverx)
            if islogx[i_] and islogy[j_]:
                sbu.update({"typeplot": "loglog"})
            elif islogx[i_]:
                sbu.update({"typeplot": "semilogx"})
            elif islogy[j_]:
                sbu.update({"typeplot": "semilogy"})
            graph[-1].update({"subplotupdate": sbu})

            # if group by subseries with same values of other parameters
            flagappend = False
            if groupbyparam and i != j:
                signatures = []
                for m in range(len(pvalsi)):
                    signatures.append(
                        ";".join([str(pvals[k, m]) for k in idx_pattern if k != i])
                    )
                signatures = np.array(signatures)
                signunique = np.unique(signatures)
                if len(signunique) < len(pvalsi):
                    for sign in signunique:
                        test = signatures == sign
                        attrs = {"linespec": "o-", "label": sign, "labelhide": 1}
                        graph.append(Curve([pvalsi[test], pvalsj[test]], attrs))
                        flagappend = True
            if not flagappend:  # not group by parameter signature: data cloud
                graph.append(Curve([pvalsi, pvalsj], {"linespec": "o"}))
            mask = ~np.logical_or(np.isnan(pvalsi), np.isnan(pvalsj))
            try:
                pearson[i_, j_, :] = pearsonr(pvalsi[mask], pvalsj[mask])
            except ValueError:
                pass  # maybe not >= 2 numbers to compute correlation

    if np.isnan(pearson).all():
        msg = "plot_correlations: only Nan in Pearson matrix. Script end."
        issue_warning(logger, msg)
        return None, None
    return graph, pearson


def crosscorrelations_stats(pvals, pattern_idx, label_idx, also_transpose=True):
    """
    Computes cross-correlation statistics for given parameter indices.

    :param pvals: 2D array of parameter values.
    :param pattern_idx: List of indices for parameters to analyze.
    :param label_idx: Index of the label parameter.
    :param also_transpose: Whether to include transposed pairs in the output.
    :return: Dictionary of statistics for parameter pairs.
    """
    if not pattern_idx or label_idx >= pvals.shape[0]:
        raise ValueError("Invalid indices provided for pattern_idx or label_idx.")

    stats = {}
    for col1, col2 in itertools.combinations(pattern_idx, 2):
        unique_col1 = np.unique(pvals[col1, :])
        unique_col2 = np.unique(pvals[col2, :])
        if len(unique_col1) == 0 or len(unique_col2) == 0:
            continue  # Skip if no unique values

        pair_stats = {}
        for val1 in unique_col1:
            mask_col1 = pvals[col1, :] == val1
            for val2 in unique_col2:
                mask = mask_col1 & (pvals[col2, :] == val2)
                if np.any(mask):
                    avg = np.nanmean(pvals[label_idx, mask])
                    std = np.nanstd(pvals[label_idx, mask])
                    pair_stats[(val1, val2)] = {"avg": avg, "std": std}
        stats[(col1, col2)] = pair_stats

    if also_transpose:
        transposed = {}
        for (col1, col2), data in stats.items():
            transposed[(col2, col1)] = {
                tuple(reversed(key)): value for key, value in data.items()
            }
        stats.update(transposed)
    return stats


def crosscorrelations_plot(
    stats, pkeys, pvals, pattern_idx, label_idx, new_graph_kwargs={}
):
    """
    Plots cross-correlations between parameters.

    :param stats: Dictionary containing statistics for parameter pairs.
    :param pkeys: List of parameter names.
    :param pvals: Array of parameter values.
    :param pattern_idx: Indices of parameters to analyze.
    :param label_idx: Index of the label parameter.
    :param new_graph_kwargs: Additional arguments for the Graph object.
    :return: Graph object with cross-correlation plots.
    """
    # constants - maybe could be modified later on
    SUBPLOTDIM = [2.5, 2.5]
    MARGIN = [1, 1, 1, 1, 1.5, 1]

    # for log plots
    TYPEPLOTS = {0: "", 1: "semilogy", 10: "semilogx", 11: "loglog"}
    islog = {i: guess_is_logarithm_quantity(pvals[i, :]) for i in pattern_idx}

    # Calculate z-axis limits
    avgmin, avgmax = np.inf, -np.inf
    for _, data in stats.items():
        for _, values in data.items():
            avgmin = min(avgmin, values["avg"])
            avgmax = max(avgmax, values["avg"])
    zlim = [avgmin, avgmax]
    zstd = np.nanstd(pvals[label_idx])

    # initialize graph
    graph = Graph(**new_graph_kwargs)
    empty_spu = {"arbitraryfunctions": [["set_axis_off", [], {}]]}

    # generate plot for each pair, also empty diagonal
    pairs = []
    for cy in pattern_idx:
        for cx in pattern_idx:
            pairs.append((cx, cy))
    for pair in pairs:
        if pair not in stats:
            # along the diagonal
            graph.append(Curve([[0], [0]], {}))
            graph.castCurve("subplot", -1, silentSuccess=True)
            graph[-1].update({"subplotupdate": empty_spu, "label": str(pair)})
        else:
            data = stats[pair]
            x, y, avg, std = [], [], [], []
            for (valx, valy), values in data.items():
                x.append(valx)
                y.append(valy)
                avg.append(values["avg"])
                std.append(values["std"])
            conf = np.clip(1 / (np.array(std) / max(zstd, 1e-300)), 0, 10)
            # create subplot
            typeplot = TYPEPLOTS[islog[pair[0]] * 10 + islog[pair[1]]]
            spu = {
                "xlim": values_to_lim(x, is_log=islog[pair[0]]),
                "ylim": values_to_lim(y, is_log=islog[pair[1]]),
                "xlabel": pkeys[pair[0]],
                "ylabel": pkeys[pair[1]],
                "typeplot": typeplot,
            }
            graph.append(Curve([[0], [0]], {}))
            graph.castCurve("subplot", -1, silentSuccess=True)
            graph[-1].update({"subplotupdate": spu, "label": pair})
            # create scatter plot
            attrs = {
                "type": "scatter",
                "vminmax": zlim,
                "colorbar": {
                    "label": pkeys[label_idx],
                    "adjust": [1.0, 0, 0.05, 1, "ax"],
                },
            }
            graph.append(Curve([x, y], attrs))
            graph.append(Curve([x, avg], {"type": "scatter_c"}))
            graph.append(Curve([x, conf * 50], {"type": "scatter_s"}))

    # cosmetics
    kw_spa = {"graph": graph, "ncols": len(pattern_idx), "nrows": len(pattern_idx)}
    graph[0].update_spa_figsize_abs(SUBPLOTDIM, MARGIN, **kw_spa)
    return graph


def plot_pearson(pearson, keysx, keysy, new_graph_kwargs={}):
    """
    Plots a Pearson correlation heatmap.

    :param pearson: A 3D NumPy array containing Pearson correlation coefficients.
                    The first two dimensions represent the correlation matrix,
                    and the third dimension contains the coefficient and p-value.
    :param keysx: List of labels for the x-axis, corresponding to the columns of the
                  correlation matrix.
    :param keysy: List of labels for the y-axis, corresponding to the rows of the
                  correlation matrix.
    :param new_graph_kwargs: Dictionary of additional keyword arguments to customize the
                             Graph object.
    :return: A Graph object representing the Pearson correlation heatmap.
    """
    lenx, leny = pearson.shape[0], pearson.shape[1]
    # plot Pearson correlation coefficient
    graph = Graph(**new_graph_kwargs)
    for c in range(1, pearson.shape[1]):
        graph.append(Curve([pearson[:, 0, 0], pearson[:, c, 0]], {}))
    graph.castCurve("image", 0, silentSuccess=True)
    one = [0.5, 0.5]
    mrgn = [2, 2.5, 1.5, 0.5]
    plts = [one[0] * lenx, one[1] * leny]
    fs = [mrgn[0] + mrgn[2] + plts[0], mrgn[1] + mrgn[3] + plts[1]]
    sa = [mrgn[0] / fs[0], mrgn[1] / fs[1], 1 - mrgn[2] / fs[0], 1 - mrgn[3] / fs[1]]
    graph.update(
        {
            "xtickslabels": [
                list(range(lenx)),
                keysx,
                {"rotation": 45, "ha": "right", "fontsize": 8},
            ],
            "ytickslabels": [list(range(leny)), keysy],
            "figsize": fs,
            "subplots_adjust": sa,
            "xlabel": " ",
            "ylabel": " ",
        }
    )
    graph[0].update(
        {
            "cmap": "RdBu",
            "vmin": -1,
            "vmax": 1,
            "aspect": "auto",  # may avoid discrepancy allocated and actual axes positio
            "colorbar": {
                "label": "Pearson correlation coefficient",
                "adjust": [1 + one[0] / 4 / plts[0], 0, one[0] / 2 / plts[0], 1, "ax"],
            },
        }
    )
    return graph


def guess_is_logarithm_quantity(series):
    sunique = np.unique(series)
    if 0 in sunique or (sunique < 0).any():
        return False
    if len(sunique) > 2:
        ratios = sunique[1:] / sunique[:-1]
        ratiosda = np.abs(ratios - ratios[0])
        # print("ratiosda", ratios, ratiosda)
        if np.max(ratiosda) < 1e-4:  # 1e-15
            # print("logarithm TRUE")
            return True
    return False


def values_to_lim(values, is_log=False):
    """Returns limits [min, max] for a list of values."""
    if is_log:
        return list(np.exp(values_to_lim(np.log(values))))
    mi, ma = np.nanmin(values), np.nanmax(values)
    s = ma - mi
    lim = [mi - s / 10, ma + s / 10]
    if lim[0] == lim[1]:
        return ""
    return lim


def filter_pvals(pkeys, pvals, filters: Optional[list] = None):
    """Filter pvals according to filters."""
    # cleanup of data input
    if filters is None:
        filters = []

    # filter data
    flagged = []
    pvalsshape = list(pvals.shape)
    for j in range(pvals.shape[1]):
        flag = True
        for fil in filters:
            i = None
            if isinstance(fil[0], int):
                i = fil[0]
            else:
                try:
                    i = pkeys.index(fil[0])
                except ValueError:
                    for i in range(len(pkeys)):
                        if fnmatch.fnmatch(pkeys[i], fil[0]):
                            break  # True
            if fil[1] == CONF.HIGHER and not pvals[i, j] > fil[2]:
                flag = False
            elif fil[1] == CONF.LOWER and not pvals[i, j] < fil[2]:
                flag = False
        if not flag:
            flagged.append(j)
    # print(pvals)
    # print(pvals.shape)
    # print(flagged)
    # print(pkeys)
    pvals = np.delete(pvals, flagged, axis=1)
    # print text message
    if len(filters) > 0:
        criteria = " Filters: " + ", ".join(["{} {} {}".format(*f) for f in filters])
        msg = "Filtering data: from {} to {} rows in input data.{}"
        print(msg.format(pvalsshape[1], pvals.shape[1], criteria))
    else:
        print("Filtering data: no filtering.")
    return pvals


def process_datatable(pkeys, pvals, idx_pattern=None, idx_labels=None, **ngkwargs):
    """Main function to process a datatable type of input"""
    graphshow = None
    if idx_pattern is not None and len(idx_pattern) == 1:
        graphshow = plot_parameters_1d(pkeys, pvals, **ngkwargs)
    if idx_pattern is not None and len(idx_pattern) == 2:
        graphshow = plot_parameters_2d(
            pkeys, pvals, idx_pattern, idx_labels, **ngkwargs
        )

    # correlation graph
    graphtable, pearson = plot_correlations(
        pkeys, pvals, idx_pattern=idx_pattern, idx_labels=idx_labels, **ngkwargs
    )

    # Pearson summary
    graphpearson = None
    if graphtable is not None:
        # pearson graph
        graphpearson = plot_pearson(
            pearson,
            [pkeys[p] for p in idx_pattern],
            [pkeys[p] for p in idx_labels],
            **ngkwargs
        )
    return graphtable, graphpearson, pearson, graphshow


def colorize_graph(graph, idx_pattern, pvals):
    """Colorize a graph based on parameter values.
    Modifies object graph
    for 2D parameter sweeps: colorize in sweeps in hls colorspace, with

    - hue between 0 and 1 according to the first parameter, starting with red,
      with additional small increment according to 2nd parameter,

    - luminance from 0.25 to 0.75 according to the second parameter
    """
    if idx_pattern is None:
        return

    if len(idx_pattern) == 1:
        graph.colorize("viridis")
        return

    if len(idx_pattern) == 2:
        x0, x1 = pvals[idx_pattern[0], :], pvals[idx_pattern[1], :]
        lookup = [[x0[i], x1[i]] for i in range(len(x0))]
        p0 = list(np.unique(x0))
        p1 = list(np.unique(x1))
        print("Colorize graph in a 2D fashion")
        hues = list(0.91 + np.linspace(0, -1, len(p0) + 1))[:-1]
        huestep = np.abs(hues[1] - hues[0]) * 0.5 if len(hues) > 1 else 0.2
        lums = np.linspace(0.25, 0.75, len(p1))
        for c in range(len(graph)):
            hue, lum = lookup[c]
            p1i = p1.index(lum)
            hls = [hues[p0.index(hue)] + huestep * p1i / (len(p1) - 1), lums[p1i], 1]
            # print("hls", hls)
            # rgb = colorsys.hls_to_rgb(*hls)
            rgb = Color(hls, space="hls").get()
            graph[c].update({"color": list(rgb)})
        return


class Helper:
    """Base helper class for parameter sweeps"""

    @classmethod
    def idx_pattern_labels(cls, pkeys):
        """Returns indices of pattern and label parameters"""
        idx_pattern = range(len(pkeys))
        idx_labels = range(len(pkeys))
        return idx_pattern, idx_labels

    @classmethod
    def as_labels_datatable(cls, graph, keys):
        """Transforms a Graph into a list of labels pkeys, data table pvals"""
        pkeys = keys
        pvals = []
        for key in keys:
            pvals.append([])
            for curve in graph:
                pvals[-1].append(curve.attr(key, np.nan))
        return pkeys, np.array(pvals), len(pkeys)


class HelperScaps(Helper):
    """Specialized helper for SCAPS 2-D parameter sweeps"""

    @classmethod
    def idx_pattern_labels(cls, pkeys):
        nresults = 6
        nparams = len(pkeys) - nresults
        nresults = 4  # hack, ignore last 2 results
        idx_pattern = range(nparams)
        idx_labels = range(nparams, nparams + nresults)
        return idx_pattern, idx_labels

    @classmethod
    def as_labels_datatable(cls, graph, *args):
        pkeys = []
        pvals = []
        keys_of_interest = []
        nparam = 0
        # retrieve parameters. number not known a priori
        while True:
            test = "Batch parameters " + str(nparam) + " key"
            if graph[0].attr(test, "") != "":
                key = graph[0].attr(test)
                pkeys.append(key)
                keys_of_interest.append(key)
                nparam += 1
                continue
            break
        # known keys for PV values
        jv_params = [
            "solar cell parameters deduced from calculated IV-curve Voc (Volt)",
            "solar cell parameters deduced from calculated IV-curve Jsc (mA/cm2)",
            "solar cell parameters deduced from calculated IV-curve FF",
            "solar cell parameters deduced from calculated IV-curve eta",
            "solar cell parameters deduced from calculated IV-curve V_MPP (Volt)",
            "solar cell parameters deduced from calculated IV-curve J_MPP (mA/cm2)",
        ]
        keys_of_interest += jv_params
        pkeys += [
            p.replace("solar cell parameters deduced from calculated IV-curve ", "")
            for p in jv_params
        ]
        for key in keys_of_interest:
            pvals.append([])
            for curve in graph:
                pvals[-1].append(curve.attr(key, np.nan))
        # print(pkeys)
        # print(pvals)
        return pkeys, np.array(pvals), nparam


class HelperDatatable(Helper):
    """Specialized helper for datatable type of input"""

    @classmethod
    def as_labels_datatable(cls, graph, *args):
        """Transforms a Graph into a list of labels pkeys, data table pvals"""
        pkeys = graph.attr("collabelsdetail", None)
        if isinstance(pkeys, list):
            if (
                len(pkeys) == 1
                and isinstance(pkeys[0], list)
                and len(pkeys[0]) == len(graph) + 1
            ):
                pkeys = pkeys[0]
            if len(pkeys) != len(graph) + 1:
                print(
                    "HelperDatatable collabelsdetail not good pick.",
                    len(pkeys),
                    len(graph),
                )
                print(graph.attr("collabelsdetail", None))
                pkeys = None
        else:  # let's try something else
            pkeys = None

        if pkeys is None:
            pkeys = graph.attr("collabels")
            if len(pkeys) == len(graph):  # missing first columns
                pkeys.insert(0, graph.attr("xlabel"))
            if len(pkeys) < len(graph) + 1:
                msg = "HelperDatatable.as_labels_datatable: Alternative guess for pkeys"
                print(msg)
                pkeys = [graph.attr("xlabel")]
                for curve in graph:
                    pkeys.append(curve.attr("label"))
        # cleanup grapa "salting" of column labels
        attrs = graph[0].get_attributes()
        for key in attrs:
            if pkeys[0].endswith(key):
                flag = True
                for c in range(len(graph)):
                    if not pkeys[c + 1].endswith(graph[c].attr(key)):
                        flag = False
                if flag:
                    le = len(pkeys[0]) - len(key)
                    pkeys = [p[le:] for p in pkeys]
                    print("HelperDatatable.as_labels_datatable: pkeys cleaned up!")
                    break
        # extract data
        pvals = [graph[0].x()]
        for curve in graph:
            pvals.append(curve.y())
        if len(pkeys) != len(pvals):
            msg = (
                "ERROR HelperDatatable.as_labels_datatable size issue. len pkeys {"
                "}, len pvals {}."
            )
            issue_warning(logger, msg.format(len(pkeys), len(pvals)))
            print(pkeys)

        # pad with nan, in case pvals have different lengths
        max_len = max(len(row) for row in pvals)
        padded_pvals = [list(row) + [np.nan] * (max_len - len(row)) for row in pvals]
        return pkeys, np.array(padded_pvals), len(pkeys)


def _choice_helper_datakeys(datakeys, graph):
    helper = Helper
    helper_dict = {CONF.AS_SCAPS: HelperScaps, CONF.AS_DATATABLE: HelperDatatable}
    if isinstance(datakeys, str) and datakeys in helper_dict:
        helper = helper_dict[datakeys]
        print("Interpret input data {}.".format(datakeys.lower().replace("_", " ")))

    # identification of data type
    if datakeys == CONF.AUTO:
        meastype = graph.attr("meastype")
        if meastype == GraphScaps.FILEIO_GRAPHTYPE:
            print("Datatype AUTO, detected Scaps.")
            datakeys = CONF.AS_SCAPS
            helper = HelperScaps

        meastype_possible = ["Undetermined data type", "Database"]
        if len(str(graph.attr("collabels", ""))) > 0 and meastype in meastype_possible:
            print("Datatype AUTO, detected data table.")
            datakeys = CONF.AS_DATATABLE
            helper = HelperDatatable
    return helper, datakeys


def process_file(
    filename: str,
    datakeys: Union[str, List[str]] = CONF.AUTO,
    filters: Optional[list] = None,
    idx_pattern=None,
    idx_labels=None,
    newGraphKwargs={},
) -> Graph:
    """Process a file contaiing correleation data. e.g. a SCAPS output file of a
    2-parameter sweep. Creates different graph files and images in the same folder,
    and returns one of the graphs.

    :param filename: file to process
    :param datakeys: how to interpret the file content.

           - CONF.AS_DATATABLE: open the file as a datatable

           - CONF.AS_SCAPS: assumes this is the output of Scaps simulations. also
             preselect series of interest

           - CONF.AUTO: first open the graph, then auto detect

           - [key1, key2, ...]: to retrieve from graph Graph, each Curve is one
              "experiment"

    :param filters: list of conditions to exclude specific "experiments" (e.g. rows in
           a table) should be excluded. e.g. [["Jsc_mApcm2", HIGHER, 10]]
    :param idx_pattern: list/range of data to consider for the correlation plots
           By default, whow all columns
    :param idx_labels: list/range of data to consider for the correlation plots
           By default, whow all columns
    :param newGraphKwargs: specific to Grapa, to e.g. have consistent config file
    :return: a Graph object
    """
    print("Processing file", filename)
    ngkwargs = {"new_graph_kwargs": newGraphKwargs}

    # open input data
    graph = Graph(filename, **newGraphKwargs)
    if len(graph) == 0:
        msg = "File seems contains not data: {}. Script end."
        issue_warning(logger, msg.format(filename))
        return False

    helper, datakeys = _choice_helper_datakeys(datakeys, graph)
    fnamebase = os.path.splitext(filename)[0]
    dataext = graph[0].attr("curve").replace("Curve ", "")
    if len(graph) > 10:
        graph.update({"legendproperties": {"fontsize": 6}})

    print("Running script assuming data", datakeys)

    # specific to file format to retrieve the parameters of interest
    # SCAPS: data as data table: parameter0, parameter1, ..., Voc, Jsc, FF, Eff
    # otherwise: datakeys as list of keys to retrieve info, e.g. from a grapa file
    pkeys, pvals, nparams = helper.as_labels_datatable(graph, datakeys)

    # filter data
    pvals = filter_pvals(pkeys, pvals, filters=filters)
    if np.isnan(pvals).all():
        graph.plot(fnamebase + "_parseddata" + dataext)  # before coloring
        if CONF.pltclose:
            plt.close()
        msg = "Data table of parameters contains only NaN. Script end."
        msg += "\npkeys: {}\npvals.shape: {}"
        issue_warning(logger, msg.format(pkeys, pvals.shape))
        return graph

    # export and plot results
    _writefile_datatable(fnamebase + "_correlation_table.txt", pkeys, pvals)

    # by default, full correlation matrix
    idx_pattern_, idx_labels_ = helper.idx_pattern_labels(pkeys)
    if idx_pattern is None:
        idx_pattern = idx_pattern_
        print("Automatic choice of seriesx: {}.".format(idx_pattern))
    if idx_labels is None:
        idx_labels = idx_labels_
        print("Automatic choice of seriesy: {}.".format(idx_labels))

    # if datakeys != CONF.AS_DATATABLE:
    colorize_graph(graph, idx_pattern, pvals)
    graph.plot(fnamebase + "_parseddata" + dataext)
    if CONF.pltclose:
        plt.close()

    # process datatable
    msg = "script correlation, reached datatable {}"
    print(msg.format(len(idx_pattern)))
    graphtable, graphpearson, pearson, graphshow = process_datatable(
        pkeys, pvals, idx_pattern=idx_pattern, idx_labels=idx_labels, **ngkwargs
    )
    print("script correlation, reached datatable save graphs")
    if isinstance(graphshow, Graph):
        graphshow.plot(fnamebase + "_summary")
        if CONF.pltclose:
            plt.close()
    if isinstance(graphtable, Graph):
        graphtable.plot(fnamebase + "_correlation_data")
        if CONF.pltclose:
            plt.close()
    if isinstance(graphpearson, Graph):
        graphpearson.plot(fnamebase + "_correlation_pearson")
        if CONF.pltclose:
            plt.close()

    # cross-correlations
    msg = "script correlation, reached crosscorrelation len(idx_pattern) {}"
    print(msg.format(len(idx_pattern)))
    if len(idx_pattern) > 1:
        for idx_l in idx_labels:
            if idx_l in idx_pattern:
                msg = "Cross-correlation: skip {} {} because part of the pattern."
                print(msg.format(idx_l, pkeys[idx_l]))
                continue
            if len(np.unique(pvals[idx_l])) < 2:
                msg = "Cross-correlation: skip {} {} because has less than 2 values."
                print(msg.format(idx_l, pkeys[idx_l]))
                continue

            averages = crosscorrelations_stats(pvals, idx_pattern, idx_l)
            graphcc = crosscorrelations_plot(
                averages, pkeys, pvals, idx_pattern, idx_l, **ngkwargs
            )
            graphcc.update({"dpi": 100})
            graphcc.plot(fnamebase + "_corrcross" + str(idx_l))
            if CONF.pltclose:
                plt.close()

    print("Script ended successfully")
    if graphtable is not None:
        return graphtable
    return graph


def run_standalone():
    """Run the script as a standalone program."""
    datakeys = CONF.AUTO
    idx_pattern = None
    idx_labels = None
    filters = []

    # Example
    filename = r"..\examples\JV\SAMPLE_B_3layerMo\I-V_SAMPLE_B_3LayerMo_Param.txt"
    datakeys = CONF.AS_DATATABLE  # not necessarily needed, autodetection should work
    filters = [["Jsc_mApcm2", CONF.HIGHER, 10]]
    idx_pattern = range(11)
    idx_labels = range(11)

    # # Example  -  please first run script JV first to generate file ...summary_allJV
    # filename = r"..\examples\JV\SAMPLE_B_3layerMo\export_SAMPLE_B_3LayerMo_summary_allJV.txt"
    # datakeys = ["Voc", "Jsc", "FF", "area"]  # , 'Eff', 'Rp', 'acquis soft rs']
    # filters = [["Jsc", CONF.HIGHER, 10]]

    # # Example
    # filename = r'G:\CIGS\RC\_simulations\20230508_Scaps_windowlayers\CIGS_RC\test\CdSX_CdSd_CdSn_ZnOd_ZnOX_AZOX.iv'
    # # datakeys = ['batch parameters 0 value', 'batch parameters 1 value']  # , 'temperature [k]']
    # filters = [["i_ZnO*affinit*", CONF.HIGHER, 4.45], [1, '>', 0.03]]

    process_file(
        filename,
        datakeys=datakeys,
        filters=filters,
        idx_pattern=idx_pattern,
        idx_labels=idx_labels,
    )


if __name__ == "__main__":
    CONF.pltclose = False
    run_standalone()
    if not CONF.pltclose:
        plt.show()
