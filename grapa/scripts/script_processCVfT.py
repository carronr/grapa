# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics,
Romain Carron
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import warnings

path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.curve import Curve
from grapa.curve_image import Curve_Image
from grapa.curve_subplot import Curve_Subplot
from grapa.mathModule import roundSignificant
from grapa.colorscale import Colorscale
from grapa.datatypes.graphCf import GraphCf
from grapa.datatypes.curveCV import CurveCV


class WarningCollector:
    def __init__(self):
        self.content = []

    def append(self, message):
        self.content.append(message)
        print(message)

    def __len__(self):
        return len(self.content)


class CurveGet:
    KEY_AREA = "cell area (cm2)"

    @classmethod
    def temperature(cls, curve):
        # retrieve temperature from a Curve
        value = curve.attr("temperature")
        if value == "":
            value = curve.attr("temperature [k]")
        return value

    @classmethod
    def volt(cls, curve):
        # retrieve bias voltage from a Curve
        value = curve.attr("bias [v]")
        if value == "":
            value = curve.attr("bias")
            value = 0 if value == "OFF" else ""
        return value

    @classmethod
    def sample(cls, curve):
        # retrieve sample name from a Curve
        value = curve.attr("sample")
        return value

    @classmethod
    def area(cls, curve):
        # retrieve sample name from a Curve
        value = curve.attr(cls.KEY_AREA)
        return value


def close_enough(value, array, tolerance=3):
    """
    if value is close enough from an existing array value, that we round it to the
    existing one
    tolerance: presumably, provided by calling script
    """
    for val in array:
        if np.abs(value - val) < tolerance:
            return val
    return value


def compute_levels(dataset, numberapprox=15, minvalue="auto", minfractionpositive=None):
    """
    Determines human-friendly levels in view of contourf plot, imposing a minvalue and
    approximate number of bins.
    minfractionpositive: minimal fraction of the colorscale with positive value, in
       case min and maxhave different signs. None: no effect. e.g. 2/3, 1/2
    The returned level values should be human-friendly
    """
    p = [0.01, 0.05, 0.95, 0.99]
    quant = np.quantile(dataset, p)
    m, M = quant[0], quant[3]  # presumably use 1% and 99% quantile
    # Calculation to ignore few extreme points. Idea: quantile 95% at least at
    # 75% of colorscale (i.e. not 95% of data within first 10% of colorscale
    M = min(M, quant[2] + 0.5 * (quant[2] - quant[1]))
    m = max(m, quant[1] - 0.5 * (quant[2] - quant[1]))
    # at least a certain fraction of colorscale > 0
    if minfractionpositive is not None and M > 0 > m:
        ratio = 1 - 1 / minfractionpositive
        if m < ratio * M:
            m = ratio * M
    # if minvalue provided
    if minvalue != "auto":
        m = minvalue
    # calculate levels
    space = roundSignificant((M - m) / numberapprox, 1)
    start = m
    if m < 0 < M:  # make sure 0 is a level
        start = -np.ceil(-m / space) * space
    levels = list(np.arange(start, M + space, space))
    return levels


def is_file_suitable(folder, file):
    """
    Is a file suitable to be considered as input for the script
    """
    fileName, fileExt = os.path.splitext(file)
    fileExt = fileExt.lower()
    if fileExt in [".png", ".py"]:
        return False  # to speed up by avoid actually opening the files
    line1, line2, line3 = GraphIO.readDataFileLine123(os.path.join(folder, file))
    if GraphCf.isFileReadable(fileName, fileExt, line1=line1, line2=line2, line3=line3):
        return True
    return False


def fs_spa_spnc(margin, panelsize, ncols, nrows):
    """
    Returns as dict with figsize, subplots_adjust and subplotsncols to set a plot
    dimensions according to margin and panelisze in inches
    """
    figsize = [
        margin[0] + margin[2] + (ncols - 1) * margin[4] + ncols * panelsize[0],
        margin[1] + margin[3] + (nrows - 1) * margin[5] + nrows * panelsize[1],
    ]
    subplots_adjust = [
        margin[0] / figsize[0],
        margin[1] / figsize[1],
        1 - margin[2] / figsize[0],
        1 - margin[3] / figsize[1],
        margin[4] / panelsize[0],
        margin[5] / panelsize[1],
    ]
    return {
        "figsize": figsize,
        "subplots_adjust": subplots_adjust,
        "subplotsncols": ncols,
    }


def parse_folder(folder, errorsmsg, tolerance_temperature="default"):
    """
    Input:
    folder: a folder
    errorsmsg: to log issues encountered by the script
    Output:
    dataset: a dict of dicts : dataset[temperature][volt] contains a CurveCf object
    volts: sorted  list of voltages found in dataset
    temperatures: sorted list of temperatures found in dataset
    sample: if sample == "", try to find information inside measurement files
    area: if area == -1, tries to find information inside measurement files
    """
    sample = ""
    area = -1
    temperatures = []
    volts = []
    dataset = {}

    kw_tol = {}
    if tolerance_temperature != "default":
        kw_tol.update({"tolerance": tolerance_temperature})

    # list possible data files, within folder and subfolders limited to 1 level below
    files = []
    for item in os.listdir(folder):
        sub = os.path.join(folder, item)
        # print("scanning through", sub)
        if os.path.isfile(sub):
            if is_file_suitable(folder, item):
                files.append(os.path.join(folder, item))
        elif os.path.isdir(sub):
            for file in os.listdir(sub):
                if os.path.isfile(os.path.join(sub, file)):
                    if is_file_suitable(sub, file):
                        files.append(os.path.join(sub, file))
    if len(files) == 0:
        errorsmsg.append("Found no suitable file in folder {}.".format(folder))
        return

    # gather Curve objects, according to temperature and voltage
    for file in files:
        graph = Graph(file, complement={"_CfLoadPhase": True})
        if len(graph) > 0:
            curve = graph[0]
            phase = graph[1] if len(graph) > 1 else None
            temperature = CurveGet.temperature(curve)
            if temperature == "":
                msg = "Cannot identify Temperature. File ignored. {}"
                errorsmsg.append(msg.format(file))
                continue
            volt = CurveGet.volt(curve)
            if volt == "":
                errorsmsg.append("Cannot identify volt. File ignored. {}".format(file))
                continue
            if sample == "":
                sample = str(CurveGet.sample(curve))
            if area == -1:
                value = CurveGet.area(curve)
                if isinstance(value, (float, int)):
                    area = value

            temperature = int(temperature)
            temperature = close_enough(temperature, temperatures, **kw_tol)

            if temperature not in temperatures:
                temperatures.append(temperature)
            if volt not in volts:
                volts.append(volt)

            if temperature not in dataset:
                dataset[temperature] = {}
            if volt not in dataset[temperature]:
                dataset[temperature][volt] = {"c": curve, "phase": phase}
            else:
                msg = "ISSUE multiple data source {} K, {} V. Ignore {}, keep {}."
                f2 = dataset[temperature][volt]["c"].attr("filename")
                msgfrm = msg.format(temperature, volt, file, f2)
                errorsmsg.append(msgfrm)
    # sort, prepare output
    volts.sort()
    temperatures.sort()
    if area == -1:
        area = 1
    return dataset, temperatures, volts, sample, area


def generate_graphs(
    dataset,
    temperatures,
    volts,
    sample,
    area,
    folder=None,  # unused, to catch default keyword
    errorsmsg=None,
    newgraphkwargs=None,
):
    graphds = {}
    graphcs = {}
    graphfs = {}

    for temperature in temperatures:
        # for volt in volts:
        #    print(temperature, volt, volt in dataset[temperature])
        volts_act = []
        matrixd, matrixc, matrixp = [], [], []
        for volt in volts:
            if volt in dataset[temperature]:
                curve = dataset[temperature][volt]["c"]
                phase = dataset[temperature][volt]["phase"]
                frequency = curve.x()
                if len(matrixd) == 0:
                    matrixd.append(frequency)
                    matrixc.append(frequency)
                    matrixp.append(frequency)
                elif len(matrixd[0]) != len(frequency):
                    msg = (
                        "ISSUE dataset discarded: different number of points. {} K, "
                        "{} V"
                    )
                    errorsmsg.append(msg.format(temperature, volt))
                    continue
                elif not np.array(matrixd[0] == frequency).all():
                    msg = (
                        "ISSUE dataset discarded: different frequency values. {} K, "
                        "{} V"
                    )
                    errorsmsg.append(msg.format(temperature, volt))
                    continue
                derivative = curve.y(alter="CurveCf.y_mdCdlnf")
                if phase is None:
                    phasey = [np.nan] * len(derivative)
                else:
                    phasey = phase.y()
                    if len(phasey) != len(derivative):
                        msg = (
                            "ISSUE phase, not same number of datapoints. Phase data "
                            "discarded. {} K, {} V"
                        )
                        errorsmsg.append(msg.format(temperature, volt))
                        phasey = [np.nan] * len(derivative)
                matrixd.append(derivative)
                matrixc.append(curve.y())
                matrixp.append(phasey)
                volts_act.append(volt)

                if volt not in graphfs:
                    graphfs[volt] = Graph("", **newgraphkwargs)
                graphfs[volt].append(curve)

        matrixd = np.insert(np.transpose(matrixd), 0, 1, axis=0)
        matrixc = np.insert(np.transpose(matrixc), 0, 1, axis=0)
        matrixp = np.insert(np.transpose(matrixp), 0, 1, axis=0)
        print("Plot temperature {}, volts {}".format(temperature, volts_act))
        matrixd[0, 1:] = volts_act
        matrixc[0, 1:] = volts_act
        matrixp[0, 1:] = volts_act

        # plot CVf-T
        graphd = Graph("", **newgraphkwargs)  # graph derivative
        for line in matrixd[1:]:
            graphd.append(Curve_Image([matrixd[0, :], line], {}))
        data = [curve.y()[1:] for curve in graphd]
        levels = compute_levels(data, minfractionpositive=3 / 4)
        graphd.update(
            {
                "typeplot": "semilogy",
                "title": "{}, {} K".format(sample, temperature),
                "xlabel": "Bias voltage [V]",
                "ylabel": "Frequency [Hz]",
                "xtickslabels": [list(volts_act), list(volts_act)],
            }
        )
        graphd[0].update(
            {
                "datafile_xy1rowcol": 1,
                "cmap": "viridis",
                "type": "contourf",
                "colorbar": {
                    "label": "Derivative -f dC/df",  # '-dC / dln(f)',
                    "adjust": [1.05, 0, 0.05, 1, "ax"],
                },
                "levels": levels,
                "extend": "both",
            }
        )
        graphd[0].update(
            {CurveGet.KEY_AREA: area, "sample": sample, "temperature [K]": temperature}
        )
        # indicate phase on plot, as contour
        # Idea: indicate problematic phase value e.g. <5degree, <20degree.
        # Possibly problematic for the hardware to determine accurately
        graphd.append(
            Curve(
                [[0], [0]],
                {"label": "Separator: end derivative, start phase", "labelhide": 1},
            )
        )
        le = len(graphd)
        for line in matrixp[1:]:
            graphd.append(Curve_Image([matrixp[0, :], line], {}))
        if len(graphd) > le:
            graphd[le].update(
                {
                    "label": "Phase [degrees] for hatches contourf",
                    "datafile_xy1rowcol": 1,
                    "type": "contourf",
                    "colors": "none",
                    "levels": [5, 20],
                    "hatches": ["///", "/", None],
                    "extend": "both",
                }
            )
        graphds[temperature] = graphd

        # plot CV-T
        graphc = Graph("", **newgraphkwargs)  # graph c -> C-V slices
        for line in matrixc[1:]:
            attrs = {
                "sample": sample,
                "frequency [hz]": line[0],
                "temperature [K]": temperature,
                "label": "{} K {:.2e} Hz".format(temperature, line[0]),
            }
            graphc.append(CurveCV([matrixc[0, 1:], line[1:]], attrs))
            graphc[-1].update({CurveGet.KEY_AREA: area})
        colorscale = Colorscale([[0.91, 0.25, 1], [2.09, 0.75, 1], "hls"], invert=True)
        graphc.colorize(colorscale)
        graphc.update({"legendtitle": "{} K".format(temperature)})
        graphcs[temperature] = graphc
    return graphds, graphcs, graphfs


def plot_graphs_cvf(
    graphds, frm="export_map_{}_K{}", folder=None, errorsmsg=None, newgraphkwargs=None
):
    sample = ""
    # decide if data warrant a usual or narrow plot
    panelsize = [4, 3]  # in inch
    ncolsmax = 3
    for key, graph in graphds.items():
        if len(graph[0].x()) == 2:
            for curve in graph:
                curve.appendPoints([curve.x()[-1] + 1e-4], [curve.y()[-1]])
            msg = (
                "ISSUE: Only 1 voltage at {} K, duplicate and add small x delta to "
                "show contour."
            )
            errorsmsg.append(msg.format(key))
            panelsize[0] = 1
            ncolsmax = 6
    # set axis limits
    xlim = [np.inf, -np.inf]
    ylim = [np.inf, -np.inf]
    for key, graph in graphds.items():
        if sample == "":
            sample = CurveGet.sample(graph[0])
        v = graph[0].x()[1:]
        xlim[0] = min(xlim[0], np.min(v))
        xlim[1] = max(xlim[1], np.max(v))
        ylim[0] = min(ylim[0], graph[0].y()[0], graph[-1].y()[0])
        ylim[1] = max(ylim[1], graph[0].y()[0], graph[-1].y()[0])
    for key, graph in graphds.items():
        graph.update({"xlim": xlim, "ylim": ylim})
    # save individual plots of CVf-T derivative
    graphmain = Graph("", **newgraphkwargs)
    for key, graph in graphds.items():
        fname = frm.format(sample, key)
        graph.export(os.path.join(folder, fname))
        # append into summary plot
        attrs = {"label": "Subplot {} K".format(key), "subplotfile": fname + ".txt"}
        graphmain.append(Curve_Subplot([[0], [0]], attrs))
    # main plot cosmetics
    margin = [1, 1, 1, 1, 1.5, 0.75]  # in inch
    ncols = min(ncolsmax, len(graphmain))
    nrows = np.ceil(len(graphds) / ncols)
    attrs = fs_spa_spnc(margin, panelsize, ncols, nrows)
    graphmain.update(attrs)
    return graphmain


def plot_graphs_cv(
    graphcs,
    # volts,
    # temperatures,
    frmk="export_CV_{}_K{}",
    frmhz="export_CV_{}_Hz{}",
    folder=None,
    errorsmsg=None,  # ok if not used
    newgraphkwargs=None,
):
    number_graphs = 9
    sample = ""
    graphcts = {}
    attrscv = {
        "alter": ["CurveCV.x_CVdepth_nm", "CurveCV.y_CV_Napparent"],
        "xlabel": "Apparent depth [nm]",
        "ylabel": "Apparent doping N$_\\mathrm{CV}$ [cm$^{-3}$]",
        "typeplot": "semilogy",
    }
    # CV for various T
    graphchzmain = Graph("", **newgraphkwargs)
    for key, graph in graphcs.items():
        if sample == "":
            sample = CurveGet.sample(graph[0])
        if len(graph) > number_graphs:
            modulo = int(np.ceil(len(graph) / number_graphs))
            for c in range(len(graph)):
                if c % modulo:
                    graph[c].swapShowHide()
                else:
                    curve = graph[c]
                    frequency = curve.attr("frequency [hz]")
                    if frequency not in graphcts:
                        graphcts[frequency] = Graph("", **newgraphkwargs)
                    graphcts[frequency].append(curve)
        graph.update(attrscv)
        xflat = []
        for curve in graph:
            if not curve.isHidden():
                x = curve.x(alter=attrscv["alter"][0])
                xflat.append(x)
        xflat = np.sort(np.array(xflat).flatten())
        if len(xflat) > 2:
            xlim = [np.min([0, xflat[1]]), xflat[-2]]
            graph.update({"xlim": xlim})
        # save individual plots CV at given T
        fname = frmk.format(sample, key)
        graph.export(os.path.join(folder, fname))
        attrs = {
            "label": "Subplot {} K".format(key),
            "subplotfile": fname + ".txt",
        }
        graphchzmain.append(Curve_Subplot([[0], [0]], attrs))

    panelsize = [4, 3]  # in inch
    margin = [1, 1, 1, 1, 1, 0.75]  # in inch
    ncols = min(3, len(graphchzmain))
    nrows = np.ceil(len(graphchzmain) / ncols)
    attrs = fs_spa_spnc(margin, panelsize, ncols, nrows)
    graphchzmain.update(attrs)
    # CV for various Hz
    graphcmain = Graph("", **newgraphkwargs)
    for key, graph in graphcts.items():
        if len(graph) > 0:
            for curve in graph:
                temperature = curve.attr("temperature [k]")
                curve.update({"label": "{:.0f} K".format(temperature)})
            if np.max([len(curve.x()) for curve in graph]) == 1:
                # if only 1 datapoint - cannot plot de doping as there is no derivative
                for curve in graph:
                    curve.update({"linespec": "o"})
                attrscv = {
                    "xlabel": "Bias voltage [V]",
                    "ylabel": "Capacitance [nF cm$^{-2}$]",
                }
            sample = graph[0].attr("sample")
            graph.update(attrscv)
            graph.update({"legendtitle": "{}, {:.0f} Hz".format(sample, key)})
            colorscale = Colorscale([[1, 0.43, 0], [0, 0, 1]], invert=True)
            graph.colorize(colorscale)
            # save graph CV at given frequency
            fname = frmhz.format(sample, int(key))
            graph.export(os.path.join(folder, fname))
            attrs = {
                "label": "Subplot {} Hz".format(key),
                "subplotfile": fname + ".txt",
            }
            graphcmain.append(Curve_Subplot([[0], [0]], attrs))

    panelsize = [4, 3]  # in inch
    margin = [1, 1, 1, 1, 1, 0.75]  # in inch
    ncols = min(3, len(graphcmain))
    nrows = np.ceil(len(graphcmain) / ncols)
    attrs = fs_spa_spnc(margin, panelsize, ncols, nrows)
    graphcmain.update(attrs)
    graphcmain.update(
        {"title": "VALUES TO CONFIRM BECAUSE CELL AREA MAY BE NOT PROPERLY SET"}
    )
    return graphcmain, graphchzmain


def plot_graphs_cf(
    graphfs,
    # volts,
    frm="export_Cf_{}_V{}",
    folder=None,
    errorsmsg=None,  # ok if not used
    newgraphkwargs=None,
):
    # Cf plots including cosmetics
    graphfmain = Graph("", **newgraphkwargs)
    volts = list(graphfs.keys())
    volts = [x for _, x in sorted(zip([float(v) for v in volts], volts))]
    for volt in volts:
        graph = graphfs[volt]
        if len(graph) > 0:
            sample = graph[0].attr("sample")
            attrs = {
                "typeplot": "semilogx",
                "ylim": [0, ""],
                "ylabel": "Capacitance [nF cm$^{-2}$]",
                "xlabel": "Frequency [Hz]",
                "legendtitle": "{}, {} V".format(sample, volt),
            }
            graph.update(attrs)
            for curve in graph:
                temperature = curve.attr("temperature [k]")
                curve.update({"label": "{:.0f} K".format(temperature)})
            colorscale = Colorscale([[1, 0.43, 0], [0, 0, 1]], invert=True)
            graph.colorize(colorscale)
            fname = frm.format(sample, volt)
            graph.export(os.path.join(folder, fname))
            attrs = {
                "label": "Subplot {} V".format(volt),
                "subplotfile": fname + ".txt",
            }
            graphfmain.append(Curve_Subplot([[0], [0]], attrs))

    panelsize = [4, 3]  # in inch
    margin = [1, 1, 1, 1, 1, 0.75]  # in inch
    ncols = min(3, len(graphfmain))
    nrows = np.ceil(len(graphfmain) / ncols)
    attrs = fs_spa_spnc(margin, panelsize, ncols, nrows)
    graphfmain.update(attrs)
    return graphfmain


def script_process_cvft(
    folder, tolerance_temperature="default", pltClose=True, newGraphKwargs={}
):
    newgraphkwargs = copy.deepcopy(newGraphKwargs)
    errorsmsg = WarningCollector()
    kw = {"newgraphkwargs": newgraphkwargs, "errorsmsg": errorsmsg, "folder": folder}

    # retrieve data from folder
    kwtol = {}
    if tolerance_temperature != "default":
        kwtol.update({"tolerance_temperature": tolerance_temperature})
    dataset, temperatures, volts, sample, area = parse_folder(
        folder, errorsmsg, **kwtol
    )

    # construct graphs for individual temperatures
    graphds, graphcs, graphfs = generate_graphs(
        dataset, temperatures, volts, sample, area, **kw
    )

    # plot individual CVf-T maps and make summary, including cosmetics
    graphmain = plot_graphs_cvf(graphds, frm="CVfT_map_{}_K{}", **kw)
    # plot individual CV-T graphs and make summary, including cosmetics
    graphcmain, graphchzmain = plot_graphs_cv(
        graphcs, frmk="CVfT_CV_{}_K{}", frmhz="CVfT_CV_{}_Hz{}", **kw
    )
    # plot individual Cf-T graphs and make summary, including cosmetics
    graphfmain = plot_graphs_cf(graphfs, frm="CVfT_Cf_{}_V{}", **kw)  # , volts

    # temporarily change cwd so subplots are properly displayed
    # first calculate filenames - necessary if folder is a relative path
    fnamemap = os.path.realpath(
        os.path.join(folder, "CVfT_map_{}_{}".format(sample, "summary"))
    )
    fnamec = os.path.realpath(
        os.path.join(folder, "CVfT_CV_T_{}_{}".format(sample, "summary"))
    )
    fnamechz = os.path.realpath(
        os.path.join(folder, "CVfT_CV_Hz_{}_{}".format(sample, "summary"))
    )
    fnamef = os.path.realpath(
        os.path.join(folder, "CVfT_Cf_{}_{}".format(sample, "summary"))
    )

    cwd = os.getcwd()
    os.chdir(folder)
    graphmain.plot(fnamemap)
    if pltClose:
        plt.close()
    with warnings.catch_warnings():  # if not enough volts to compute doping
        warnings.simplefilter("ignore", category=RuntimeWarning)
        graphcmain.plot(fnamec)
        graphchzmain.plot(fnamechz)
    if pltClose:
        plt.close()
    graphfmain.plot(fnamef)
    if pltClose:
        plt.close()
    os.chdir(cwd)

    # end
    if len(errorsmsg) == 0:
        print("Script ended succesfully without issues.")
    else:
        msg = "Script ended succesfully. CAUTION: encountered {} issues, details above."
        print(msg.format(len(errorsmsg)))
    return graphmain


# TODO: VERIFY cell area correctly implemented

if __name__ == "__main__":
    """
    From a set of C-f data acquired at different voltages and temperatures,
    provides C-V-f maps for each temperatures, as well as C-V and C-f plots.
    """

    folder_ = "./../examples/Cf/"
    script_process_cvft(folder_, pltClose=False)  # , tolerance_temperature=5)

    plt.show()
