# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:46:38 2016

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import glob
import os
import copy
import sys
from typing import Optional, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.curve_subplot import Curve_Subplot

try:
    from grapa.scripts.script_processJV import writeFileAvgMax
except ImportError:
    pass


@contextmanager
def temporary_chdir(new_path):
    prev_cwd = os.getcwd()  # Save current working directory
    os.chdir(new_path)  # Change to new directory
    try:
        yield  # Run the code inside the `with` block
    finally:
        os.chdir(prev_cwd)  # Restore original directory


@dataclass
class ManyArgs:
    """for nicer_looking code. These variables do not change over time."""

    folder: str
    export_prefix: str
    plotkwargs: dict
    pltclose: bool


def label_from_graph(graph, silent=False, file=""):
    # label will be per priority:
    # 1. if a 'label' field is present at beginning of file (so all curve labels are
    #    identical)
    # 2. a 'sample' field of 'sample name' field at beginning of file
    # 3. processed filename

    # if label set at beginning of the file -> same label for each curve
    lbl = graph[0].attr("label")
    for curve in graph:
        if not curve.attr("label").startswith(lbl):
            lbl = None
            break
    if lbl is not None:
        if not silent:
            msg = "label: {} ('label' attribute in file header, or unexpected input)"
            print(msg.format(lbl))
    if lbl is None:
        # if not every label is identical: look for 'sample' and 'sample
        # name' fields overrides if finds 'sample' keyword in the file headers
        if graph[0].attr("sample name") != "":
            lbl = graph[0].attr("sample name")
            if isinstance(lbl, list):
                lbl = lbl[0]
                for curve in graph:  # clean a bit the mess
                    if isinstance(curve.attr("sample name"), list):
                        curve.update({"sample name": curve.attr("sample name")[0]})
            if not silent:
                print("label:", lbl, "('sample name' attribute in file header)")
        elif graph.attr("sample") != "":
            lbl = graph.attr("sample")
            if not silent:
                print("label:", lbl, "('sample' attribute in file header)")
    if lbl is None:
        lbl = (
            os.path.basename(file)
            .replace("export", "")
            .replace("summary", "")
            .replace("_", " ")
            .replace("  ", " ")
        )
        if not silent:
            msg = (
                "label: {} (from file name - no 'label', 'sample' or 'sample name' "
                "attribute found in file header)"
            )
            print(msg.format(lbl))
    return lbl


def generate_file_list(folder, export_prefix):
    globpath = os.path.join(folder, "*.txt")
    file_list = sorted(glob.glob(globpath))
    files = []
    for file in file_list:
        filename = os.path.basename(file)
        # do not want to open previously created export files
        if len(export_prefix) > 0 and filename[: len(export_prefix)] != export_prefix:
            files.append(file)
    return files


def _is_jv(files, newgraphkwargs):
    print("Tasting file", files[0])
    graph = Graph(files[0], **newgraphkwargs)
    collabels = graph.attr("collabels")
    if len(collabels) < 8:
        return False

    jv = True
    cols = [0, 1, 2, 3, 8, 9]  # required columns
    titles = [["Voc"], ["Jsc"], ["FF"], ["Eff"], ["Rp"], ["Rs"]]
    for col, title_ in zip(cols, titles):
        flag = False
        for tmp in title_:
            if tmp in collabels[col]:
                flag = True
                break
        if not flag:
            jv = False
    return jv


def JVSummaryToBoxPlots(
    folder: str,
    exportPrefix="boxplot_",
    replace: Optional[List[Tuple[str, str]]] = None,
    plotkwargs: Optional[dict] = None,
    silent=False,
    pltClose=True,
    newGraphKwargs: Optional[dict] = None,
):
    """Generate box plots from JV or general data files in a given folder.

    :param folder: folder where to look for data files (default: current folder)
    :param exportPrefix: prefix for the output box plot files
    :param replace: list of string replacements to be made in the labels
    :param plotkwargs: dict of keyword arguments to be passed to calls to graph.plot()
    :param silent: if True, suppress console output (default: False)
    :param pltClose: if True, close the plots after saving
    :param newGraphKwargs: dict to be passed to the Graph constructor

    :return: Graph object containing the summary of box plots
    """
    if replace is None:
        replace = []  # possible input: [['Oct1143\n', ''], ['\nafterLS','']]
    if plotkwargs is None:
        plotkwargs = {}
    if newGraphKwargs is None:
        newGraphKwargs = {}
    ngkwargs = copy.deepcopy(newGraphKwargs)
    ngkwargs.update({"silent": silent})

    # establish list of files
    files = generate_file_list(folder, exportPrefix)
    if len(files) == 0:
        print("Script JV summary to boxplot: Cold not find suitable data files. Abort.")
        return None

    # default mode: every column is shown, grouped by columns
    mode = "all"
    # test if JV mode: only a subset is shown, and some cosmetics is performed
    jv_titles = []  # to silence linters
    if _is_jv(files, ngkwargs):
        mode = "JV"
        jv_cols = [0, 1, 2, 3, 8, 9, 10, 11]  # all columns of interest
        jv_titles = [
            ["Voc"],
            ["Jsc"],
            ["FF"],
            ["Eff"],
            ["Rp"],
            ["Rs"],
            ["n"],
            ["J0", "I0"],
        ]
        jv_updates = [{}] * len(jv_cols)
        jv_updates[-1] = {"typeplot": "semilogy"}  # J0

    graphs: List[Graph] = []
    titles = []

    n_carriagereturn_label = 0
    boxplot_positions = []
    str_statistics = ""

    for file in files:
        # open the file corresponding to 1 given sample
        if not silent:
            print("open file", file)
        graph = Graph(file, **ngkwargs)
        if len(graph) == 0:
            print("Cannot interpret file data. Go to next one. File:", file)
            continue

        if mode == "JV":
            str_statistics += jv_extract_stats(graph, len(str_statistics))

        complement = copy.deepcopy(graph[0].get_attributes())
        upd_boxp_sc = {"color": "grey", "size": 5}
        if "color" in complement:
            upd_boxp_sc = {"color": complement["color"], "alpha": 0.33, "size": 5}
        attr_complement = {
            "type": "boxplot",
            "showfliers": False,
            "boxplot_addon": ["scatter", upd_boxp_sc],
        }
        complement.update(attr_complement)
        if "boxplot_position" in complement:
            boxplot_positions.append(complement["boxplot_position"])

        label = label_from_graph(graph, silent=silent, file=file)
        for rep in replace:  # replace provided by user
            label = label.replace(rep[0], rep[1])
        complement.update({"label": label})
        # needed to estimate the graph height
        n_carriagereturn_label = max(n_carriagereturn_label, label.count("\n"))
        # construct one column of each box plot
        collabels = graph.attr("collabels")
        curves_of_interest = jv_cols if mode == "JV" else range(len(graph))

        for i, c in enumerate(curves_of_interest):
            if c >= len(graph):
                print("graph[c].getData()", len(graph), c)
                continue

            data = graph[c].getData()
            # prepare graph title
            while i >= len(titles):
                titles.append("")
            while len(collabels) <= c:
                print("Maybe bug JVSummaryToBoxPlots len", len(graph), collabels)
                collabels.append("")

            title = collabels[c] if titles[i] == "" else ""

            # if input not match JV file, discard it
            if mode == "JV":
                title, data = jv_title_data(
                    title, data, c, collabels, i, jv_titles[i], file
                )

            # prepare graph
            if title != "":
                titles[i] = graph.formatAxisLabel(title)
            while i >= len(graphs):
                graphs.append(Graph("", **ngkwargs))
            graphs[i].append(Curve(data, complement))
        # go to next file

    manyargs = ManyArgs(folder, exportPrefix, plotkwargs, pltClose)

    graph_size(graphs, files, n_carriagereturn_label, boxplot_positions)
    filesaves = plot_individual_graphs(manyargs, graphs, titles, mode, jv_updates)
    graphsum = graph_summary(manyargs, filesaves)
    save_statistics(manyargs, str_statistics)
    print("JVSummaryToBoxPlots completed.")
    return graphsum


def jv_extract_stats(graph, len_str_statistics):
    # gather statistics data to be placed in a text file
    try:
        add = writeFileAvgMax(
            graph,
            colSample=True,
            filesave=False,
            ifPrint=False,
            withHeader=(True if len_str_statistics == 0 else False),
        )
        if isinstance(add, str):
            return add
    except Exception as e:
        print("Exception += ", type(e), e)
    return ""


def jv_title_data(title, data, c, collabels, i, jv_titlei, file):
    flag = False
    if c < len(collabels):
        for tit in jv_titlei:
            if collabels[c].startswith(tit):
                flag = True
                break
    if not flag:
        msg = "Wrong column name, no data plotted, file {} i {} column {}"
        print(msg.format(file, i, c))
        data = [[np.nan], [np.nan]]
        title = ""

    repl = {
        "Voc_V": "Voc [V]",
        "Jsc_mApcm2": "Jsc [mA/cm2]",
        "FF_pc": "FF [%]",
        "Eff_pc": "Eff. [%]",
        "_Ohmcm2": " [Ohmcm2]",
        "A/cm2": "A cm$^{-2}$",
        "J0 ": "J$_0$ ",
        "Ohmcm2": "$\\Omega$ cm$^2$",
    }
    for old, new in repl.items():
        title = title.replace(old, new)
    return title, data


def graph_size(graphs, files, n_carriagereturn_label, boxplot_positions):
    """Set the figure size and margin according to number of files and
    length of the labels"""
    margin = [1.1, 0.5 + (n_carriagereturn_label + 1) * 12 / 72 * 1.2, 0.5, 0.5]  # inch
    # default font size 12, 72 dpi, 1.2 interline spacing
    deltax = len(files) + 1
    if len(boxplot_positions) == len(files):
        deltax = np.max(boxplot_positions) - np.min(boxplot_positions) + 1
    figsize = (margin[0] + margin[2] + 0.75 * deltax, margin[1] + margin[3] + 3)
    subplots_adjust = [
        margin[0] / figsize[0],
        margin[1] / figsize[1],
        1 - margin[2] / figsize[0],
        1 - margin[3] / figsize[1],
    ]
    for graph in graphs:
        graph.update({"figsize": figsize, "subplots_adjust": subplots_adjust})


def plot_individual_graphs(manyargs: ManyArgs, graphs, titles, mode, jv_updates):
    """Plot and save individual graphs"""
    filesaves = []
    for i, graph in enumerate(graphs):
        if len(graph) == 0:
            continue
        # only plot if # of non-NaN elements in each curve is > 0
        if sum([sum(~np.isnan(curve.x())) for curve in graph]) == 0:
            continue

        title = titles[i]
        graph.update({"ylabel": title})
        filesave = os.path.join(manyargs.folder, manyargs.export_prefix + str(i))
        if mode == "JV":
            jv_update = jv_updates[i]
            if jv_update is not {}:
                graph.update(jv_update)
            tit = title.split(" ")[0].replace("$", "").replace(".", "").replace("_", "")
            filesave = os.path.join(manyargs.folder, manyargs.export_prefix + tit)
        graph.plot(filesave=filesave, **manyargs.plotkwargs)
        if manyargs.pltclose:
            plt.close()
        filesaves.append(filesave + ".txt")
    return filesaves


def graph_summary(args: ManyArgs, filesaves):
    """Make a big graph with one panel per individual graph created previsouly"""
    graphsum = Graph()
    if len(filesaves) == 0:
        return graphsum

    for filesave in filesaves:
        fname = os.path.basename(filesave)
        quantity = fname.replace(args.export_prefix, "").replace(".txt", "")
        curve = Curve_Subplot([[0], [0]], {})
        curve.update({"subplotfile": fname, "label": "Subplot " + quantity})
        graphsum.append(curve)

    graphsum.update({"subplotsncols": 2, "subplotstranspose": 1})
    panelsize = [3, 3]
    margin = [1, 1, 1, 1, 1, 1]
    nrows = 2
    ncols = np.ceil(len(graphsum) / nrows)
    graphsum[0].update_spa_figsize_abs(
        panelsize, margin, ncols=ncols, nrows=nrows, graph=graphsum
    )
    filesave = os.path.realpath(
        os.path.join(args.folder, args.export_prefix + "summary")
    )

    with temporary_chdir(args.folder):
        print("filesave", filesave)
        graphsum.plot(filesave=filesave, **args.plotkwargs)
        if args.pltclose:
            plt.close()
    return graphsum


def save_statistics(args: ManyArgs, str_statistics):
    """Save statistics text file if any"""
    if len(str_statistics) > 0:
        print(str_statistics)
        filesave = os.path.join(args.folder, args.export_prefix + "statistics.txt")
        print("filesave", filesave)
        with open(filesave, "w") as f:
            f.write(str_statistics)


def demo():
    folder_ = "./../examples/boxplot/"
    JVSummaryToBoxPlots(folder_, pltClose=False, silent=True)
    plt.show()


if __name__ == "__main__":
    demo()
