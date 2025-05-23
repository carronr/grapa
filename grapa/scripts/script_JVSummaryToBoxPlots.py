# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:46:38 2016

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import glob
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import sys

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


def label_from_graph(graph, silent=False, file=""):
    # label will be per priority:
    # 1. if a 'label' field is present at beginning of file (so all curve labels are
    #    identical)
    # 2. a 'sample' field of 'sample name' field at beginning of file
    # 3. processed filename

    # if label set at beginning of the file -> same label for each curve
    lbl = graph[0].attr("label")
    for c in range(len(graph)):
        if graph[c].attr("label") != lbl:
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
                for c in range(len(graph)):  # clean a bit the mess
                    if isinstance(graph[c].attr("sample name"), list):
                        graph[c].update(
                            {"sample name": graph[c].attr("sample name")[0]}
                        )
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


def JVSummaryToBoxPlots(
    folder=".",
    exportPrefix="boxplot_",
    replace=None,
    plotkwargs={},
    silent=False,
    pltClose=True,
    newGraphKwargs={},
):
    if replace is None:
        replace = []  # possible input: [['Oct1143\n', ''], ['\nafterLS','']]
    newGraphKwargs = copy.deepcopy(newGraphKwargs)
    newGraphKwargs.update({"silent": silent})

    # establish list of files
    path = os.path.join(folder, "*.txt")
    fileList = sorted(glob.glob(path))
    files = []
    for file in fileList:
        filename = os.path.basename(file)
        # do not want to open previously created export files
        if len(exportPrefix) > 0 and filename[: len(exportPrefix)] != exportPrefix:
            files += [file]
    if len(files) == 0:
        return None

    # default mode: every column is shown, grouped by columns
    mode = "all"
    # test if JV mode: only a subset is shown, and some cosmetics is performed
    isJV = True
    graph = Graph(files[0], **newGraphKwargs)
    collabels = graph.attr("collabels")
    if len(collabels) < 8:
        isJV = False
    else:
        collabels = graph.attr("collabels")
        cols = [0, 1, 2, 3, 8, 9]  # required columns
        titles = [["Voc"], ["Jsc"], ["FF"], ["Eff"], ["Rp"], ["Rs"]]
        for c in range(len(cols)):
            flag = False
            for tmp in titles[c]:
                if tmp in collabels[cols[c]]:
                    flag = True
                    break
            if not flag:
                isJV = False
    if isJV:
        mode = "JV"
        JVcols = [0, 1, 2, 3, 8, 9, 10, 11]  # all columns of interest
        JVtitles = [
            ["Voc"],
            ["Jsc"],
            ["FF"],
            ["Eff"],
            ["Rp"],
            ["Rs"],
            ["n"],
            ["J0", "I0"],
        ]
        JVupdates = [{}] * len(JVcols)
        JVupdates[-1] = {"typeplot": "semilogy"}  # J0

    graphs = []
    titles = []

    nbCarriageReturnLabel = 0
    strStatistics = ""

    for file in files:
        # open the file corresponding to 1 given sample
        if not silent:
            print("open file", file)
        graph = Graph(file, **newGraphKwargs)
        if len(graph) == 0:
            print("Cannot interpret file data. Go to next one.")
            continue

        complement = copy.deepcopy(graph[0].get_attributes())
        complement.update(
            {
                "type": "boxplot",
                "showfliers": False,
                "boxplot_addon": ["scatter", {"color": "grey", "size": 5}],
            }
        )
        if mode == "JV":
            # gather statistics data to be placed in a text file
            try:
                add = writeFileAvgMax(
                    graph,
                    colSample=True,
                    filesave=None,
                    ifPrint=False,
                    withHeader=(True if len(strStatistics) == 0 else False),
                )
                if isinstance(add, str):
                    strStatistics += add
                # print('strStatistics increment', len(strStatistics))
            except Exception as e:
                print("Exception += ", type(e), e)
                pass

        label = label_from_graph(graph, silent=silent, file=file)
        # "clean" label according to replacement pairs provided by user
        for rep in replace:
            label = label.replace(rep[0], rep[1])
        complement.update({"label": label})

        # needed to estimate the graph height
        nbCarriageReturnLabel = max(nbCarriageReturnLabel, label.count("\n"))

        # construct one column of each box plot
        collabels = graph.attr("collabels")
        curvesOfInterest = JVcols if mode == "JV" else range(len(graph))
        print("size graph", len(graph), curvesOfInterest)
        for i in range(len(curvesOfInterest)):
            c = curvesOfInterest[i]
            if c >= len(graph):
                print("graph[c]c.getData()", len(graph), c)
                continue
            data = graph[c].getData()

            # prepare graph title
            while i >= len(titles):
                titles.append("")
            title = collabels[c] if titles[i] == "" else ""

            # if input not match JV file, discard it
            if mode == "JV":
                flag = False
                if c < len(collabels):
                    for tit in JVtitles[i]:
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

            # prepare graph
            if title != "":
                titles[i] = graph.formatAxisLabel(title)
            while i >= len(graphs):
                graphs.append(Graph("", **newGraphKwargs))
            graphs[i].append(Curve(data, complement))
        # go to next file

    # cosmetics, then save plots
    margin = [1.1, 0.5 + (nbCarriageReturnLabel + 1) * 12 / 72 * 1.2, 0.5, 0.5]  # inch
    # default font size 12, 72 dpi, 1.2 interline spacing
    figsize = (margin[0] + margin[2] + 0.9 * len(files), margin[1] + margin[3] + 3)
    subplots_adjust = [
        margin[0] / figsize[0],
        margin[1] / figsize[1],
        1 - margin[2] / figsize[0],
        1 - margin[3] / figsize[1],
    ]
    filesaves = []
    for i in range(len(graphs)):
        graph = graphs[i]
        if len(graph) > 0:
            # counts number of non-NaN element in each curve, only plot if is > 0
            num = sum([sum(~np.isnan(c.x())) for c in graph])
            if num > 0:
                graph.update(
                    {
                        "figsize": figsize,
                        "ylabel": titles[i],
                        "subplots_adjust": subplots_adjust,
                    }
                )
                filesave = os.path.join(folder, exportPrefix + str(i))
                if mode == "JV":
                    if JVupdates[i] is not {}:
                        graph.update(JVupdates[i])
                    tit = titles[i]
                    filesave = os.path.join(
                        folder,
                        exportPrefix
                        + tit.split(" ")[0]
                        .replace("$", "")
                        .replace(".", "")
                        .replace("_", ""),
                    )
                graph.plot(filesave=filesave, **plotkwargs)
                if pltClose:
                    plt.close()
                filesaves.append(filesave + ".txt")

    # summary graph
    graphsum = Graph()
    for filesave in filesaves:
        graphsum.append(Curve_Subplot([[0], [0]], {}))
        fname = os.path.basename(filesave)
        quantity = fname.replace(exportPrefix, "").replace(".txt", "")
        graphsum[-1].update({"subplotfile": fname, "label": "Subplot " + quantity})
    if len(graphsum) > 0:
        graphsum.update({"subplotsncols": 2, "subplotstranspose": 1})
        panelsize = [3, 3]
        margin = [1, 1, 1, 1, 1, 1]
        nrows = 2
        ncols = np.ceil(len(graphsum) / nrows)
        graphsum[0].update_spa_figsize_abs(
            panelsize, margin, ncols=ncols, nrows=nrows, graph=graphsum
        )
        filesave = os.path.realpath(os.path.join(folder, exportPrefix + "summary"))
        cwd = os.getcwd()
        os.chdir(folder)
        print("filesave", filesave)
        graphsum.plot(filesave=filesave, **plotkwargs)
        if pltClose:
            plt.close()
        os.chdir(cwd)

    # print and save statistics
    if len(strStatistics) > 0:
        print(strStatistics)
        filesave = os.path.join(folder, exportPrefix + "statistics.txt")
        print("filesave", filesave)
        with open(filesave, "w") as f:
            f.write(strStatistics)

    # return last plot
    print("JVSummaryToBoxPlots completed.")
    return graphsum


if __name__ == "__main__":
    folder_ = "./../examples/boxplot/"
    #    folder_ = './../examples/boxplot/notJVspecific/'
    JVSummaryToBoxPlots(
        folder=folder_, exportPrefix="boxplots_", pltClose=True, silent=True
    )
    plt.show()
