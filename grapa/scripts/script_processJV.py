# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics,
Romain Carron
"""

import numpy as np
import os
import sys

import glob
from copy import deepcopy
from re import search as research
import matplotlib.pyplot as plt

path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.database import Database
from grapa.mathModule import is_number, roundSignificant, strToVar
from grapa.colorscale import Colorscale
from grapa.curve import Curve
from grapa.curve_subplot import Curve_Subplot

from grapa.datatypes.graphJV import GraphJV
from grapa.datatypes.curveJV import CurveJV
from grapa.datatypes.graphJVDarkIllum import GraphJVDarkIllum


# prompt for folder
def promptForFolder():
    # Start->Execute, cmd, type "pip install tkinter"
    from tkinter.filedialog import askdirectory

    path = askdirectory()
    return path


# auxiliary function
def dictToListSorted(d):
    lst = []
    for key in d:
        lst.append(key)
    lst.sort()
    return lst


class areaDBHandler:
    def __init__(self):
        self._dict = {}

    def get(self, p1, p2):
        print("areaDBHandler", p1, p2)
        if p1 not in self._dict:
            self._dict.update({p1: {}})
        if p2 not in self._dict[p1]:
            self._dict[p1].update({p2: areaDB(p1, p2)})
        return self._dict[p1][p2]


#  auxiliary class - add some useful tools to the database class
class areaDB(Database):
    def __init__(self, folder, sample):
        self.colIdx = 0
        self.folder = folder
        self.sample = sample
        # different possible names for area databases
        test = []
        test.append([os.path.join(folder, sample + ".txt"), ""])
        test.append([os.path.join(folder, sample + "_area.txt"), ""])
        test.append([os.path.join(folder, sample + "_areas.txt"), ""])
        test.append([os.path.join(folder, "area.txt"), ""])
        test.append([os.path.join(folder, "area.xlsx"), sample])
        test.append([os.path.join(folder, "areas.txt"), ""])
        test.append([os.path.join(folder, "areas.xlsx"), sample])
        self.flag = False
        self.flagCheat = False
        for t in test:
            print("Area database: look for file ", t[0])
            graph = Graph(t[0], complement=t[1], silent=True, config=None)
            if len(graph) == 0:
                continue
            else:
                try:
                    # try convert it in database
                    Database.__init__(self, graph)
                    # identify suitable column
                    for col in self.colLabels:
                        if col.find("area") > -1:
                            self.colIdx = col
                    self.flag = True
                    print("Database of cell area parsed and area identified.")
                except Exception:
                    print("-> database file data could not be interpreted correctly.")
                    continue
                break  # break if success
        if not self.flag:
            print("areaDB: cannot find area database file.")

    def getArea(self, cell):
        if not self.flag:  # if database could not be opened
            if self.flagCheat:
                if cell in self.data:
                    return self.data[cell]
            return np.nan
        out = self.value(self.colIdx, cell)
        if np.isnan(out):
            out = self.value(self.colIdx, self.sample + " " + cell)
        if np.isnan(out):
            msg = "areaDB getArea: row {} not found ({})"
            print(msg.format(cell, self.rowLabels))
        #        print ('area cell',cell,':', out, '(',self.sample,')')
        return out

    def setArea(self, cell, value):
        if self.flag:
            self.setValue(self.colIdx, cell, value, silent=True)
        else:
            self.flagCheat = True
            if not hasattr(self, "data"):
                self.data = {}
            self.data.update({cell: value})


# main function
def processJVfolder(
    folder,
    ylim=[-50, 150],
    sampleName="",
    fitDiodeWeight=0,
    groupCell=True,
    figAx=None,
    pltClose=True,
    newGraphKwargs={},
):
    msg = "Script processJV folder initiated. Data processing can last a few seconds."
    print(msg)
    newGraphKwargs = deepcopy(newGraphKwargs)
    newGraphKwargs.update({"silent": True})
    if figAx is not None:
        pltClose = False
    path = os.path.join(folder, "*.txt")

    cellDict = {}
    sampleAreaDict = {}
    outFlag = True
    out0 = ""

    for file in glob.glob(path):
        fileName, fileExt = os.path.splitext(file)
        fileExt = fileExt.lower()
        line1, line2, line3 = GraphIO.readDataFileLine123(filename=file)
        if GraphJV.isFileReadable(
            fileName, fileExt, line1=line1, line2=line2, line3=line3
        ):
            try:
                graphTmp = Graph(file, **newGraphKwargs)
            except IndexError:
                continue  # next file
        else:
            continue
        try:
            sample = graphTmp[-1].sample()
            cell = graphTmp[-1].cell()
            dark = graphTmp[-1].darkOrIllum(ifText=True)
            measId = graphTmp[-1].measId()
        except Exception:
            continue  # go to next file

        print("File", os.path.basename(file))
        if sample == "" or cell == "":
            print(
                "WARNING: cannot identify sample (", sample, ") or cell (", cell, ")."
            )

        if sample != "" and cell != "":
            if sample not in cellDict:
                cellDict.update({sample: {}})
            if cell not in cellDict[sample]:
                cellDict[sample].update({cell: {}})
            if dark not in cellDict[sample][cell]:
                cellDict[sample][cell].update({dark: {}})
            if sample not in sampleAreaDict:
                areaHandler = areaDBHandler()
                sampleAreaDict.update({sample: areaHandler.get(folder, sample)})
            # open JV file again to correct for cell area
            area = sampleAreaDict[sample].getArea(cell)
            if (
                np.isnan(area)
                and is_number(graphTmp[0].attr("Acquis soft Cell area"))
                and not np.isnan(graphTmp[0].attr("Acquis soft Cell area"))
            ):
                sampleAreaDict[sample].setArea(
                    cell, graphTmp[0].attr("Acquis soft Cell area")
                )
                area = sampleAreaDict[sample].getArea(cell)
            cellDict[sample][cell][dark].update({measId: file})

            areaformer = graphTmp[0].attr("Acquis soft Cell area")
            msg = "  Sample {}, cell {}, area {} cm2 (acquired as {})"
            print(msg.format(sample, cell, area, areaformer))

        if outFlag:
            out0 = out0 + graphTmp[-1].printShort(header=True)
            outFlag = False

    # sweep through files, identify pairs
    listSample = dictToListSorted(cellDict)
    graphAllJV = Graph("", **newGraphKwargs)  # in case listSample is empty
    for s in listSample:
        graphAllJV = Graph("", **newGraphKwargs)

        s_str = "Sample " + str(s) if is_number(s) else str(s)
        out = (
            "Sample\t"
            + s_str
            + "\n"
            + "label\t"
            + s_str.replace("_", "\\n")
            + "\n"
            + deepcopy(out0)
        )
        outIllum = (
            "Sample\t"
            + s_str
            + "\n"
            + "label\t"
            + s_str.replace("_", "\\n")
            + "\n"
            + deepcopy(out0)
        )
        outDark = (
            "Sample\t"
            + s_str
            + "\n"
            + "label\t"
            + s_str.replace("_", "\\n")
            + "\n"
            + deepcopy(out0)
        )
        listCell = dictToListSorted(cellDict[s])
        for c in listCell:
            listDarkIllum = dictToListSorted(cellDict[s][c])
            # if want to process each emasureemnt independently
            if not groupCell:
                for d in listDarkIllum:
                    listGraph = dictToListSorted(cellDict[s][c][d])
                    for m in listGraph:
                        filesave = (
                            "export_"
                            + s
                            + "_"
                            + c
                            + "_"
                            + m
                            + ("_" + d).replace("_" + CurveJV.ILLUM, "")
                        )
                        print("Graph saved as", filesave)
                        graph = GraphJVDarkIllum(
                            cellDict[s][c][d][m],
                            "",
                            area=sampleAreaDict[s].getArea(c),
                            complement={
                                "ylim": ylim,
                                "saveSilent": True,
                                "_fitDiodeWeight": fitDiodeWeight,
                            },
                            **newGraphKwargs
                        )
                        out = out + graph.printShort()
                        filesave = os.path.join(folder, filesave)
                        graph.plot(filesave, figAx=figAx, pltClose=pltClose)
                        graphAllJV.append(graph.returnDataCurves())
            else:
                # if want to restrict 1 msmt dark + 1 illum per cell
                if len(listDarkIllum) > 2:
                    print("test WARNING sorting JV files.")
                # only dark on only illum measurement
                if len(listDarkIllum) == 1:
                    d = listDarkIllum[0]
                    listGraph = dictToListSorted(cellDict[s][c][d])
                    m = listGraph[0]
                    filesave = (
                        "export_"
                        + s
                        + "_"
                        + c
                        + "_"
                        + m
                        + ("_" + d)
                        .replace("_" + CurveJV.ILLUM, "")
                        .replace("_" + CurveJV.DARK, "")
                    )
                    print("Graph saved as", filesave)
                    fileDark = (
                        cellDict[s][c][d][m] if listDarkIllum[0] == CurveJV.DARK else ""
                    )
                    fileIllum = (
                        cellDict[s][c][d][m]
                        if listDarkIllum[0] == CurveJV.ILLUM
                        else ""
                    )
                    # create Graph file
                    graph = GraphJVDarkIllum(
                        fileDark,
                        fileIllum,
                        area=sampleAreaDict[s].getArea(c),
                        complement={
                            "ylim": ylim,
                            "saveSilent": True,
                            "_fitDiodeWeight": fitDiodeWeight,
                        },
                        **newGraphKwargs
                    )
                    filesave = os.path.join(folder, filesave)
                    graph.plot(filesave=filesave, figAx=figAx, pltClose=pltClose)
                    # prepare output summary files
                    out = out + graph.printShort()
                    if listDarkIllum[0] == CurveJV.DARK:
                        outDark = outDark + graph.printShort()
                    else:
                        outIllum = outIllum + graph.printShort()
                    # if len(listGraph) > 1 :
                    #    msg = '.'.join([cellDict[s][c][d][m2] for m2 in cellDict[s][c][d]][1:])
                    #    print ('test WARNING: other files ignored (,',msg,')')
                    graphAllJV.append(graph.returnDataCurves())

                # can identify pair of dark-illum files
                if len(listDarkIllum) == 2:
                    listGraph = dictToListSorted(cellDict[s][c][listDarkIllum[0]])
                    filesave = "export_" + s + "_" + c + "_" + listGraph[0]
                    fileDark = cellDict[s][c][listDarkIllum[0]][listGraph[0]]

                    listGraph = dictToListSorted(cellDict[s][c][listDarkIllum[1]])
                    filesave = filesave + "-" + listGraph[0]
                    fileIllum = cellDict[s][c][listDarkIllum[1]][listGraph[0]]

                    print("Graph saved as", filesave)
                    # create Graph file
                    graph = GraphJVDarkIllum(
                        fileDark,
                        fileIllum,
                        area=sampleAreaDict[s].getArea(c),
                        complement={
                            "ylim": ylim,
                            "saveSilent": True,
                            "_fitDiodeWeight": fitDiodeWeight,
                        },
                        **newGraphKwargs
                    )
                    filesave = os.path.join(folder, filesave)
                    graph.plot(filesave=filesave, figAx=figAx, pltClose=pltClose)
                    # prepare output summary files
                    out = out + graph.printShort()
                    outIllum = outIllum + graph.printShort(onlyIllum=True)
                    outDark = outDark + graph.printShort(onlyDark=True)
                    graphAllJV.append(graph.returnDataCurves())

        # graph with all JV curves area-corrected
        for c in graphAllJV:
            c.update({"color": ""})
        if len(cellDict[s].keys()) > 0:  # retrieve consistent xlabel, ylabel
            cds = cellDict[s]
            try:
                cdsc = cds[list(cds.keys())[0]]
                cdscd = cdsc[list(cdsc.keys())[0]]
                randomfile = cdscd[list(cdscd.keys())[0]]
                graphtmp = Graph(randomfile)
                graphAllJV.update(
                    {
                        "xlabel": graphtmp.attr("xlabel"),
                        "ylabel": graphtmp.attr("ylabel"),
                    }
                )
            except IndexError:
                pass
        filesave = os.path.join(folder, "export_" + s + "_summary_allJV")
        graphAllJV.plot(filesave, figAx=figAx)
        if pltClose and figAx is None:
            plt.close()

        # print sample summary
        filesave = "export_" + s + "_summary" + ".txt"
        filesave = os.path.join(folder, filesave)
        # print('End of JV curves processing, showing summary file...')
        # print(out)
        print("Summary saved in file", filesave, ".")
        with open(filesave, "w") as f:
            f.write(out)
        if groupCell:
            filesave = "export_" + s + "_summary_dark" + ".txt"
            filesave = os.path.join(folder, filesave)
            with open(filesave, "w") as f:
                f.write(outDark)
            processSampleCellsMap(filesave, figAx=figAx, pltClose=pltClose)
            filesave = "export_" + s + "_summary_illum" + ".txt"
            filesave = os.path.join(folder, filesave)
            with open(filesave, "w") as f:
                f.write(outIllum)
            processSampleCellsMap(filesave, figAx=figAx, pltClose=pltClose)
            writeFileAvgMax(filesave, filesave=True, ifPrint=True)
    print("Script processJV folder done.")
    # print('return graph', type(graph), graph)
    # Graph.plot(graph, os.path.join(folder, 'export_test'))
    return graphAllJV


def writeFileAvgMax(
    fileOrContent, filesave=False, withHeader=True, colSample=True, ifPrint=True
):
    colOfInterest = ["Voc", "Jsc", "FF", "Eff"]
    if isinstance(fileOrContent, Graph):
        content = fileOrContent
        filename = content.attr("sample").replace("\n", "")
        if filename == "":
            filename = content.attr("label")
    else:
        content = Graph(fileOrContent)
        filename = fileOrContent
    #    print(content)
    # identify columns of interest
    colLbl = content.attr("collabels")
    cols = []
    idxs = []
    for c in colOfInterest:
        cols.append([np.nan])
        idxs.append(np.nan)
        for i in range(len(colLbl)):
            if c in colLbl[i]:
                cols[-1] = content[i].y()
                idxs[-1] = i
    # start to compile output in variable out
    out = ""
    if withHeader:
        if not colSample:
            out += "filename\t" + filename + "\n"
            out += "Sample\t" + content.attr("sample") + "\n"
        if colSample:
            out += "\t"
        # column headers
        out += "Best cell (eff.)" + "\t" * len(colOfInterest)
        out += "Parameter average" + "\t" * len(colOfInterest)
        out += "Parameter median" + "\t" * len(colOfInterest)
        out += "\n"
        # column name
        if colSample:
            out += "Sample\t"
        for i in range(3):
            for c in idxs:
                out += (colLbl[c] if c is not np.isnan(c) else "") + "\t"
        out += "\n"
    # sample name
    if colSample:
        samplename = "DEFAULT"
        if "sample" in content.headers:
            samplename = content.headers["sample"]
        elif content[-1].attr("sample name", None) is not None:
            samplename = content[-1].attr("sample name")
        elif content[-1].attr("label", None) is not None:
            samplename = content[-1].attr("label")
        if isinstance(samplename, list):
            samplename = str(samplename[0])
        samplename = samplename.replace("\n", " ")
        out += samplename + "\t"
    # best cell
    eff = None
    for i in range(len(colLbl)):
        if "Eff" in colLbl[i]:
            eff = content[i].y()
    if eff is not None:
        idx = np.argmax(eff)
        for c in cols:
            out += str(c[idx]) + "\t"
    else:
        print("Could not find column Eff")
    # averages
    for c in cols:
        out += str(np.average(c)) + "\t"
    # averages
    for c in cols:
        out += str(np.median(c)) + "\t"
    # new line
    out += "\n"
    # maybe save result in a file
    if isinstance(filename, str) and filesave is True:
        fname = filename.replace(".txt", "_avgmax.txt")
        with open(fname, "w") as f:
            f.write(out)
    if ifPrint:
        print(out)
    return out


def processSampleCellsMap(
    file, colorscale=None, figAx=None, pltClose=True, newGraphKwargs={}
):
    newGraphKwargs = deepcopy(newGraphKwargs)
    newGraphKwargs.update({"silent": True})

    content = Graph(file, **newGraphKwargs)
    colToPlot = [
        "Voc",
        "Jsc",
        "FF",
        "Eff",
        "Rp",
        "Rs",
        "n",
        "J0",
        "Rp acquis. software",
        "Rs acquis. software",
        "",
        "Rsquare",
    ]
    inveScale = [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
    ]  # inverted color scale
    cols = content.attr("collabels")
    rows = content.attr("rowlabels")
    if not isinstance(cols, list):
        print("Error processSampleCellsMap: cols is not a list (value", cols, ")")
    if colorscale is None:
        if content.attr("colorscale", None) is not None:
            colorscale = content.attr("colorscale")
        if content.attr("cmap", None) is not None:
            colorscale = content.attr("cmap")
    filelist = []
    filesummary = []

    # curve blank
    curveblank = Curve_Subplot([[0], [0]], {"subplotfile": ""})
    curveblank.update(
        {"subplotupdate": {"arbitraryfunctions": [["set_axis_off", [], {}]]}}
    )
    # locate possible warnings for poor fits - in view of modifying markersize
    rsquare = 1
    for j in range(len(cols)):
        if cols[j].startswith("Rsquare"):
            rsquare = content[j].y()
            print("rsquare", j, rsquare)

    # combined plots
    graphs = []
    # axisheights = [[], []]
    figsize = [6, 4]
    sadjust = [0.1, 0.1, 0.9, 0.9]
    # main loop
    for i in range(len(colToPlot)):
        look = colToPlot[i]
        AB = 0 if i < 4 else 1
        while len(graphs) <= AB:
            graphs.append(Graph("", **newGraphKwargs))
        colFound = False

        if look == "":  # blank space is desired
            graphs[AB].append(curveblank)
            continue

        for j in range(len(cols)):
            c = cols[j]
            if c[: len(look)] == look and (
                len(c) <= len(look)
                or c[len(look)] in [" ", "_", "-", ".", "[", "]", "(", ")"]
            ):
                c = c.replace("_", " [").replace("(", "[")
                if "[" in c and "]" not in c:
                    c += "]"
                c = c.replace("[pc]", "[%]").replace("mApcm2", "mA/cm2")
                # sort somehow identical cells?
                vals = content[j].y()
                lookfname = look.replace(".", "").replace(" ", "").lower()
                filesave = ".".join(file.split(".")[:-1]) + "_" + lookfname
                filelist.append(filesave + ".txt")

                sizefactor = 1
                # reduce marker size if fit was poor: Rp, Rs, J0 , n, Rsquare
                if i in [4, 5, 6, 7, 11]:
                    if isinstance(rsquare, int):  # not a list, or np.array
                        rsquare = [rsquare] * len(vals)
                    sizefactor = [
                        1
                        if rsquare[k] > 0.995
                        else (0.25 if rsquare[k] > 0.95 else 1 / 9)
                        for k in range(len(vals))
                    ]
                # no marker if data nonsensical
                if i in [4, 8]:  # if Rp < 0
                    sf = np.array([1 if val > 0 else 0 for val in vals])
                    sizefactor = sf * sizefactor

                res = plotSampleCellsMap(
                    rows,
                    vals,
                    c,
                    sizefactor,
                    colorscale=colorscale,
                    filesave=filesave,
                    figAx=figAx,
                    inverseScale=inveScale[i],
                    pltClose=pltClose,
                )

                if isinstance(res, Graph) and len(res) > 0:
                    fname = res.attr("filesave")
                    attrs = {
                        "subplotfile": fname + ".txt",
                        "label": "Subplot " + str(look),
                    }
                    graphs[AB].append(Curve_Subplot([[0], [0]], attrs))
                    figsize = res.attr("figsize", [6, 4])  # will be same for all plots
                    sadjust = res.attr(
                        "subplots_adjust", [0.1, 0.1, 0.9, 0.9]
                    )  # will be same for all plots
                colFound = True
                break
        if not colFound:
            msg = "Warning processSampleCellsMap: column not found ({} or similar)"
            print(msg.format(look))
            graphs[AB].append(curveblank)

    # remove empty subplots at the end of each graph
    for graph in graphs:
        for i in range(len(graph) - 1, -1, -1):
            if graph[i] == curveblank:
                graph.deleteCurve(i)
            else:
                break
        if len(graph) == 1:  # at least 2 subplots, otherwise grapa may not show well
            graph.append(curveblank)

    # set correct graph size for compiled graphs
    panelsize = [
        figsize[0] * (sadjust[2] - sadjust[0]),
        figsize[1] * (sadjust[3] - sadjust[1]),
    ]
    if len(graphs[0]) > 0:
        margins = [0.7, 0.5, 0.7, 0.5, 1.0, 0.5]
        graphs[0].update({"subplotsheight_ratios": ""})
        graphs[0][0].update_spa_figsize_abs(
            panelsize, margins, ncols=1, nrows=len(graphs[0]), graph=graphs[0]
        )
        filesave = ".".join(file.split(".")[:-1]) + "_" + ["basic", "diode"][0]
        graphs[0].filename = filesave
        graphs[0].plot(filesave=filesave, figAx=figAx)
        if pltClose:
            plt.close()
        filesummary.append(filesave + ".txt")

    if len(graphs[1]) > 0:
        margins = [0.7, 0.5, 0.7, 0.5, 1.0, 0.5]
        ncols = 2 if len(graphs[1]) > 2 else 1
        nrows = np.ceil(len(graphs[1]) / ncols)
        graphs[1].update({"subplotstranspose": 1})
        graphs[1][0].update_spa_figsize_abs(
            panelsize, margins, ncols=ncols, nrows=nrows, graph=graphs[1]
        )
        filesave = ".".join(file.split(".")[:-1]) + "_" + ["basic", "diode"][1]
        graphs[1].filename = filesave
        graphs[1].plot(filesave=filesave, figAx=figAx)
        if pltClose:
            plt.close()
        filesummary.append(filesave + ".txt")

    return filelist, filesummary


def plotSampleCellsMap(
    cells,
    values,
    title,
    sizefactor=1,
    colorscale=None,
    filesave="",
    figAx=None,
    inverseScale=False,
    pltClose=True,
    newGraphKwargs={},
):
    # sizefactor: float, or np.array for each item in values. e.g. 0.25 to reduce markersize
    sizeCell = np.array([0.6, 0.6])
    margin = np.array([0.4, 0.4])

    newGraphKwargs = deepcopy(newGraphKwargs)
    newGraphKwargs.update({"silent": True})

    if len(values) == 0:
        return False

    # check cells are with correct form, i.e. 'a1'
    x, y, val = [], [], []
    split = [research(r"([a-zA-Z])([0-9]*)", c).groups() for c in cells]
    for i in range(len(split)):
        if len(split[i]) == 2:
            x.append(float(ord(split[i][0].lower()) - 96))
            y.append(float(split[i][1]))
            # prefer to work on a copy and not modifying the list values
            val.append(values[i])
    x, y, val = np.array(x), np.array(y), np.array(val)
    if title == "Voc [V]":
        title = "Voc [mV]"
        val *= 1000
    title = (
        title.replace("Voc_V", "Voc [V]")
        .replace("Jsc_mApcm2", "Jsc [mA/cm2]")
        .replace("FF_pc", "FF [%]")
        .replace("Eff_pc", "Eff. [%]")
        .replace("_Ohmcm2", " [Ohmcm2]")
    )
    title = (
        title.replace("A/cm2", "A cm$^{-2}$")
        .replace("J0 ", "J$_0$ ")
        .replace("Ohmcm2", "$\Omega$ cm$^2$")
    )

    valNorm = val if not (val == 0).all() and not len(val) == 1 else [0.5] * len(val)
    if isinstance(colorscale, Colorscale):
        colorscale = colorscale.getColorScale()
    if isinstance(
        colorscale, str
    ):  # 'autumn' is a string, but a color list might not have been recognized as a list
        colorscale = strToVar(colorscale)
    if colorscale is None:
        colorscale = [[1, 0, 0], [1, 1, 0.5], [0, 1, 0]]
    if inverseScale and isinstance(colorscale, list):
        colorscale = colorscale[::-1]

    xticks = np.arange(0, max(x) + 1, 1)  # if max(x) > 6 else np.arange(0,max(x)+1,1)
    yticks = np.arange(0, max(y) + 1, 1)
    axSize = np.array(
        [
            sizeCell[0] * (max(xticks) - min(xticks)),
            sizeCell[1] * (max(yticks) - min(yticks)),
        ]
    )
    figsize = axSize + 2 * margin[1]
    marg = margin / figsize
    txtCoords = np.transpose(
        [
            (x - 0.5 - min(xticks)) / (max(xticks) - min(xticks)),
            (y - 0.5 - min(yticks)) / (max(yticks) - min(yticks)),
        ]
    )
    toPrint = [roundSignificant(v, 3) for v in val]
    if np.average(val) > 1e2:
        toPrint = ["{:1.0f}".format(v) for v in toPrint]
    if np.average(val) < 1e-3:
        toPrint = ["{:.1E}".format(v).replace("E-0", "E-") for v in toPrint]
    if title.startswith("Rsquare"):  # more digits for Rsquare
        toPrint = [roundSignificant(v, 5) for v in val]
    markersize = (sizeCell[0] * 72) ** 2
    valSize = np.array([markersize for val in x])
    valSize *= sizefactor
    graph = Graph("", **newGraphKwargs)
    texttxt = []
    textarg = []
    for i in range(len(val)):
        texttxt.append(toPrint[i])
        textarg.append(
            {
                "xytext": list(txtCoords[i]),
                "xycoords": "axes fraction",
                "horizontalalignment": "center",
                "verticalalignment": "center",
            }
        )
    graph.append(
        Curve(
            [x - 0.5, y - 0.5],
            {
                "type": "scatter",
                "marker": "s",
                # 'markersize': (sizeCell[0]*72)**2,
                "markeredgewidth": 0,
                "cmap": colorscale,
            },
        )
    )
    graph.append(Curve([x - 0.5, valNorm], {"type": "scatter_c"}))
    graph.append(Curve([x - 0.5, valSize], {"type": "scatter_s"}))

    graph.update(
        {
            "subplots_adjust": [marg[0], marg[1], 1 - marg[0], 1 - marg[1]],
            "figsize": list(figsize),
            "text": texttxt,
            "textargs": textarg,
            "title": graph.formatAxisLabel(title),
            "xlim": [min(xticks), max(xticks)],
            "ylim": [min(yticks), max(yticks)],
        }
    )
    fct = []
    fct.append(["set_xticks", [list(xticks)], {}])
    fct.append(["set_yticks", [list(yticks)], {}])
    fct.append(["set_xticklabels", [[]], {}])
    fct.append(["set_yticklabels", [[]], {}])
    fct.append(["set_xticks", [list(xticks[1:] - 0.5)], {"minor": True}])
    fct.append(["set_yticks", [list(yticks[1:] - 0.5)], {"minor": True}])
    fct.append(
        [
            "set_xticklabels",
            [[chr(int(i) - 1 + ord("a")) for i in xticks[1:]]],
            {"minor": True},
        ]
    )
    fct.append(["set_yticklabels", [[int(i) for i in yticks[1:]]], {"minor": True}])
    fct.append(["tick_params", [], {"axis": "both", "which": "minor", "length": 0}])
    fct.append(["grid", [True], {}])
    graph.update({"arbitraryfunctions": fct})

    if filesave is not None:
        graph.headers.update({"filesave": os.path.basename(filesave)})
        # graph.plot(filesave, figAx=figAx)  # plot -> would need to plt.close() accordingly
        graph.export(filesave)  # export only txt file, and not the image
    else:
        graph.plot(figAx=figAx, ifSubPlot=True)
        if pltClose and figAx is None:
            plt.close()
    return graph


if __name__ == "__main__":
    # go through files, store files content in order to later select pairs
    folder = "./../examples/JV/SAMPLE_A/"
    # processJVfolder(folder, fitDiodeWeight=5, pltClose=True,  groupCell=True)
    processJVfolder(folder, groupCell=True, fitDiodeWeight=5, pltClose=False)

    # file = r'./../examples/JV\SAMPLE_B_3layerMo\export_sample_b_3layermo_summary_illum.txt'
    # processSampleCellsMap(file, pltClose=True)

    # writeFileAvgMax(file)

    plt.show()
