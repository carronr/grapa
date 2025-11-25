# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import sys
from copy import deepcopy
from re import search as research
import glob
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

path_ = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path_ not in sys.path:
    sys.path.append(path_)

from grapa.graph import Graph
from grapa.utils.parser_dispatcher import file_read_first3lines
from grapa.database import Database
from grapa.mathModule import is_number, roundSignificant
from grapa.utils.string_manipulations import strToVar
from grapa.colorscale import Colorscale
from grapa.curve import Curve
from grapa.curve_subplot import Curve_Subplot

from grapa.datatypes.graphJV import GraphJV
from grapa.datatypes.curveJV import CurveJV
from grapa.datatypes.graphJVDarkIllum import GraphJVDarkIllum


# prompt for folder
def prompt_for_folder():
    """Prompt user for a folder, return path as string."""
    # Start->Execute, cmd, type "pip install tkinter"
    from tkinter.filedialog import askdirectory

    path = askdirectory()
    return path


# auxiliary function
def dict_to_list_sorted(d):
    """Convert dictionary keys to a sorted list"""
    lst = []
    for key in d:
        lst.append(key)
    lst.sort()
    return lst


class AreaDBHandler:
    """Handler for area databases, to avoid opening the same database multiple times
    .get(folder, sample) returns an AreaDB object

    Retrospectvely: class seems useless, could be bypassed"""

    def __init__(self):
        self._dict = {}

    def get(self, folder, sample):
        """.get(folder, sample) returns an AreaDB object"""
        print("areaDBHandler", folder, sample)
        if folder not in self._dict:
            self._dict.update({folder: {}})
        if sample not in self._dict[folder]:
            self._dict[folder].update({sample: AreaDB(folder, sample)})
        return self._dict[folder][sample]


#  auxiliary class - add some useful tools to the database class
class AreaDB(Database):
    """Class to handle area database files"""

    def __init__(self, folder, sample):
        self.col_idx = 0
        self.folder = folder
        self.sample = sample
        # different possible names for area databases
        test = [
            [os.path.join(folder, sample + ".txt"), ""],
            [os.path.join(folder, sample + "_area.txt"), ""],
            [os.path.join(folder, sample + "_areas.txt"), ""],
            [os.path.join(folder, "area.txt"), ""],
            [os.path.join(folder, "area.xlsx"), sample],
            [os.path.join(folder, "areas.txt"), ""],
            [os.path.join(folder, "areas.xlsx"), sample],
        ]
        self.flag = False
        self.flag_cheat = False
        for t in test:
            print("Area database: look for file ", t[0])
            if not os.path.exists(t[0]):  # file not found
                continue

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
                            self.col_idx = col
                    self.flag = True
                    print("Database of cell area parsed and area identified.")
                except Exception:
                    print("-> database file data could not be interpreted correctly.")
                    continue
                break  # break if success
        if not self.flag:
            print("areaDB: cannot find area database file.")

    def get_area(self, cell):
        """get area for a given cell, return np.nan if not found"""
        if not self.flag:  # if database could not be opened
            if self.flag_cheat:
                if cell in self.data:
                    return self.data[cell]
            return np.nan
        out = self.value(self.col_idx, cell)
        if isinstance(out, list):
            print("Warning: please check that cell (picel) exists in database!", cell)
            return np.nan
        if np.isnan(out):
            out = self.value(self.col_idx, self.sample + " " + cell)
        if np.isnan(out):
            msg = "AreaDB get_area: row {} not found ({})"
            print(msg.format(cell, self.rowLabels))
        #        print ('area cell',cell,':', out, '(',self.sample,')')
        return out

    def set_area(self, cell, value):
        """set area for a given cell"""
        if self.flag:
            self.setValue(self.col_idx, cell, value, silent=True)
        else:
            self.flag_cheat = True
            if not hasattr(self, "data"):
                self.data = {}
            self.data.update({cell: value})


class IvDataLocator:

    @classmethod
    def parse_folder_or_file(
        cls, folder_or_file: str, newgraphkwargs: dict
    ) -> Tuple[dict, dict, str]:
        if os.path.isfile(folder_or_file):
            return cls._parse_file_unique(folder_or_file, newgraphkwargs)
        # is folder
        sampledict, areadict, header = cls._parse_folder_files_raw(
            folder_or_file, newgraphkwargs
        )
        if len(sampledict) == 0:
            sampledict, areadict, header = cls._parse_folder_files_consolidated(
                folder_or_file, newgraphkwargs
            )
        return sampledict, areadict, header

    @staticmethod
    def _add_to_sampledict(sampledict, areadict, curve: CurveJV, folder, file=None):
        try:
            sample = curve.sample()
            cell = curve.cell()
            dark = curve.darkOrIllum(ifText=True)
            measid = int(curve.measId())
        except Exception:
            print("failed step retrieve identifiers")
            return False

        if sample == "" or cell == "":
            msg = "WARNING: cannot identify sample ({}) or cell ({})."
            print(msg.format(sample, cell))
            return False

        if sample != "" and cell != "":
            if sample not in sampledict:
                sampledict.update({sample: {}})
            if cell not in sampledict[sample]:
                sampledict[sample].update({cell: {}})
            if dark not in sampledict[sample][cell]:
                sampledict[sample][cell].update({dark: {}})
            to_insert = file if file is not None else curve
            sampledict[sample][cell][dark].update({measid: to_insert})

            # retrieve area inormation from dedicated file, if possible
            if sample not in areadict:
                area_handler = AreaDBHandler()
                areadict.update({sample: area_handler.get(folder, sample)})

            areadb = areadict[sample].get_area(cell)
            areaattr = curve.area()
            areainit = curve.attr("Acquis soft Cell area")
            if np.isnan(areadb) and is_number(areaattr) and not np.isnan(areaattr):
                areadict[sample].set_area(cell, areaattr)
                areadb = areadict[sample].get_area(cell)

            msgsub = []
            if is_number(areadb) and is_number(areaattr) and areadb != areaattr:
                msgsub.append("parsed as {}".format(areaattr))
            if is_number(areaattr) and is_number(areainit) and areainit != areaattr:
                msgsub.append("initial acquisition {}".format(areainit))
            msg = "  {}, cell {}: area {} cm2 ({})"
            print(msg.format(sample, cell, areadb, ", ".join(msgsub)))

    @classmethod
    def _parse_folder_files_raw(cls, folder: str, newgraphkwargs: dict):
        sampledict = {}
        areadict = {}
        header = ""
        header_done = False

        path = os.path.join(folder, "*.txt")
        for file in glob.glob(path):
            filename, ext = os.path.splitext(file)
            ext = ext.lower()
            line1, line2, line3 = file_read_first3lines(filename=file)
            if GraphJV.isFileReadable(
                filename, ext, line1=line1, line2=line2, line3=line3
            ):
                try:
                    graph = Graph(file, **newgraphkwargs)
                except IndexError:
                    continue  # next file
            else:
                continue
            print("File:", os.path.basename(file))

            curve = graph[-1]
            cls._add_to_sampledict(sampledict, areadict, curve, folder, file=file)

            if not header_done:
                header += curve.printShort(header=True)
                header_done = True
        return sampledict, areadict, header

    @classmethod
    def _parse_folder_files_consolidated(cls, folder: str, newgraphkwargs: dict):
        sampledict = {}
        areadict = {}
        header = ""
        header_done = False

        path = os.path.join(folder, "*.txt")
        for file in glob.glob(path):
            if os.path.basename(file).startswith("export"):
                continue
            graph = Graph(file, **newgraphkwargs)
            if len(graph) < 2:
                continue
            onlyjv = True
            for curve in graph:
                if not isinstance(curve, CurveJV):
                    onlyjv = False
                    break
            if not onlyjv:
                continue

            msg = "Found file containing only CurveJV Curves, at least 2 of them: {}"
            print(msg.format(file))
            print("Do not consider other files, execution may last too long.")
            for curve in graph:
                cls._add_to_sampledict(sampledict, areadict, curve, folder)
                if not header_done:
                    header += curve.printShort(header=True)
                    header_done = True
        return sampledict, areadict, header

    @classmethod
    def _parse_file_unique(cls, file: str, newgraphkwargs: dict):
        sampledict = {}
        areadict = {}
        header = ""
        header_done = False

        folder = os.path.dirname(file)
        graph = Graph(file, **newgraphkwargs)
        for curve in graph:
            cls._add_to_sampledict(sampledict, areadict, curve, folder)
            if not header_done:
                header += curve.printShort(header=True)
                header_done = True
        return sampledict, areadict, header


def _plot_jvall(graph: Graph, s: str, folder: str, fig_ax: list, pltclose: bool):
    fsave = "export_{}_summary_allJV".format(s)
    for c in graph:
        c.update({"color": ""})
    graph.update({"xlabel": GraphJV.AXISLABELS[0], "ylabel": GraphJV.AXISLABELS[1]})
    graph.plot(os.path.join(folder, fsave), fig_ax=fig_ax)
    if pltclose and fig_ax is None:
        plt.close()


def _process_sample(s, celldict, areadict, header0, struct):
    encoding = "utf-8"
    folder, fig_ax, pltClose, group_cell, fitDiodeWeight, ylim, newgraphkwargs = struct
    kwargs_plot = {"fig_ax": fig_ax, "pltClose": pltClose}
    complement = {"ylim": ylim, "saveSilent": True, "_fitDiodeWeight": fitDiodeWeight}

    s_str = "Sample " + str(s) if is_number(s) else str(s)
    out_base = "Sample\t{}\nlabel\t{}\n{}"
    outall = out_base.format(s_str, s_str.replace("_", "\\n"), header0)
    outillum = out_base.format(s_str, s_str.replace("_", "\\n"), header0)
    outdark = out_base.format(s_str, s_str.replace("_", "\\n"), header0)

    graph_alljv = Graph("", **newgraphkwargs)
    cell_list = dict_to_list_sorted(celldict)
    for c in cell_list:
        area = areadict[s].get_area(c)
        kwargs_more = {"area": area, "complement": complement}
        kwargs_more.update(newgraphkwargs)
        darkillum_list = dict_to_list_sorted(celldict[c])
        # if want to process each emasureemnt independently
        if not group_cell:
            for d in darkillum_list:
                files_list = dict_to_list_sorted(celldict[c][d])
                for m in files_list:
                    graph = GraphJVDarkIllum(celldict[c][d][m], "", **kwargs_more)
                    graph_alljv.append(graph.get_datacurves())
                    outall += graph.printShort()
                    fsave_tmp = ("_" + d).replace("_" + CurveJV.ILLUM, "")
                    fsave = "export_{}_{}_{}{}".format(s, c, m, fsave_tmp)
                    graph.plot(os.path.join(folder, fsave), **kwargs_plot)
                    print("Graph saved as", fsave)
        else:
            # if want to restrict 1 msmt dark + 1 illum per cell
            if len(darkillum_list) > 2:
                print("test WARNING sorting JV files.")
            # only dark on only illum measurement
            if len(darkillum_list) == 1:
                d = darkillum_list[0]
                files_list = dict_to_list_sorted(celldict[c][d])
                m = files_list[0]
                file_dark = (
                    celldict[c][d][m] if darkillum_list[0] == CurveJV.DARK else ""
                )
                file_illum = (
                    celldict[c][d][m] if darkillum_list[0] == CurveJV.ILLUM else ""
                )
                # create Graph file
                graph = GraphJVDarkIllum(file_dark, file_illum, **kwargs_more)
                fsave_tmp = (
                    ("_" + d)
                    .replace("_" + CurveJV.ILLUM, "")
                    .replace("_" + CurveJV.DARK, "")
                )
                fsave = "export_{}_{}_{}{}".format(s, c, m, fsave_tmp)
                graph.plot(filesave=os.path.join(folder, fsave), **kwargs_plot)
                print("Graph saved as", fsave)
                graph_alljv.append(graph.get_datacurves())
                # prepare output summary files
                outall = outall + graph.printShort()
                if darkillum_list[0] == CurveJV.DARK:
                    outdark += graph.printShort()
                else:
                    outillum += graph.printShort()
                # if len(listGraph) > 1 :
                #    msg = '.'.join([cellDict[s][c][d][m2] for m2 in cellDict[s][c][d]][1:])
                #    print ('test WARNING: other files ignored (,',msg,')')

            # can identify pair of dark-illum files
            if len(darkillum_list) == 2:
                list_files0 = dict_to_list_sorted(celldict[c][darkillum_list[0]])
                file_dark = celldict[c][darkillum_list[0]][list_files0[0]]
                list_files1 = dict_to_list_sorted(celldict[c][darkillum_list[1]])
                file_illum = celldict[c][darkillum_list[1]][list_files1[0]]
                # create Graph file
                graph = GraphJVDarkIllum(file_dark, file_illum, **kwargs_more)
                fsave_tmp = "export_{}_{}_{}-{}"
                fsave = fsave_tmp.format(s, c, list_files0[0], list_files1[0])
                print("Graph saved as", fsave)
                graph.plot(filesave=os.path.join(folder, fsave), **kwargs_plot)
                graph_alljv.append(graph.get_datacurves())
                # prepare output summary files
                outall += graph.printShort()
                outillum += graph.printShort(onlyIllum=True)
                outdark += graph.printShort(onlyDark=True)

    # graph with all JV curves area-corrected
    _plot_jvall(graph_alljv, s, folder, fig_ax, pltClose)

    # print sample summary
    fsave = "export_{}_summary.txt".format(s)
    fsave = os.path.join(folder, fsave)
    print("Summary saved in file {}.".format(fsave))
    with open(fsave, "w", encoding=encoding) as f:
        f.write(outall)

    if group_cell:  # summaries dark and illum
        fsave = "export_{}_summary_dark.txt".format(s)
        fsave = os.path.join(folder, fsave)
        with open(fsave, "w", encoding=encoding) as f:
            f.write(outdark)
        processSampleCellsMap(fsave, figAx=fig_ax, pltClose=pltClose)

        fsave = "export_{}_summary_illum.txt".format(s)
        fsave = os.path.join(folder, fsave)
        with open(fsave, "w", encoding=encoding) as f:
            f.write(outillum)
        processSampleCellsMap(fsave, figAx=fig_ax, pltClose=pltClose)
        writeFileAvgMax(fsave, filesave=True, ifPrint=True)
    return graph_alljv


# main function
def processJVfolder(
    folder: str,
    ylim: list = [-50, 150],
    fitDiodeWeight: float = 0,
    groupCell: bool = True,
    figAx: list = None,
    pltClose: bool = True,
    newGraphKwargs: dict = {},
):
    """Process a folder containing JV files, identify pairs of dark-illum files,"""
    msg = "Script processJV folder initiated. Data processing can last a few seconds."
    print(msg)
    newgraphkwargs = deepcopy(newGraphKwargs)
    newgraphkwargs.update({"silent": True})
    if figAx is not None:
        pltClose = False

    # parse sample, cell, dark/illum, msmtid from files
    print("Parsing file or folder {}...".format(folder))
    sampledict, areadict, header = IvDataLocator.parse_folder_or_file(
        folder, newgraphkwargs
    )

    if len(sampledict) == 0:
        print("Unfortunately, no suitable file found...")
        return Graph("", **newgraphkwargs)

    # process each sample found
    struct = (folder, figAx, pltClose, groupCell, fitDiodeWeight, ylim, newgraphkwargs)
    for s, celldict in sampledict.items():
        graph_alljv = _process_sample(s, celldict, areadict, header, struct)
    print("Script processJV folder done.")
    return graph_alljv


def writeFileAvgMax(
    fileOrContent, filesave=False, withHeader=True, colSample=True, ifPrint=True
):
    """Process a file or Graph containing summary of solar cell parameters for different
    cells."""
    encoding = "utf-8"
    cols_of_interest = ["Voc", "Jsc", "FF", "Eff"]
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

    curves_of_interest = {}
    for key in cols_of_interest:
        curves_of_interest[key] = {
            "y": [],
            "average": "",
            "median": "",
            "argmax": None,
            "collabely": key,
        }
        found = False
        for curve in content:
            collabels = curve.attr("_collabels")
            if not isinstance(collabels, (list, tuple)) or len(collabels) < 2:
                continue
            if key in collabels[1]:
                found = True
                y = curve.y()
                curves_of_interest[key]["curve"] = curve
                curves_of_interest[key]["collabely"] = collabels[1]
                curves_of_interest[key]["y"] = y
                if len(y) > 0:
                    curves_of_interest[key]["average"] = np.average(y)
                    curves_of_interest[key]["median"] = np.average(y)
                    curves_of_interest[key]["argmax"] = np.argmax(y)
                break
        if not found:
            print("ScriptJV: loop curves could NOT find key:", key)

    # start to compile output in variable out
    out = ""
    if withHeader:
        if not colSample:
            out += "filename\t" + filename + "\n"
            out += "Sample\t" + content.attr("sample") + "\n"
        if colSample:
            out += "\t"
        # column headers
        out += "Best cell (eff.)" + "\t" * len(cols_of_interest)
        out += "Parameter average" + "\t" * len(cols_of_interest)
        out += "Parameter median" + "\t" * len(cols_of_interest)
        out += "\n"
        # column name
        if colSample:
            out += "Sample\t"
        for i in range(3):
            for _, item in curves_of_interest.items():
                out += item["collabely"] + "\t"
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
    idx_best = curves_of_interest["Eff"]["argmax"]
    for _, item in curves_of_interest.items():
        if idx_best is not None:
            val = item["y"][idx_best] if len(item["y"]) > idx_best else ""
        else:
            val = ""
        out += str(val) + "\t"
    if idx_best is None:
        print("Could not find column Eff")
    # averages
    for _, item in curves_of_interest.items():
        out += str(item["average"]) + "\t"
    # averages
    for _, item in curves_of_interest.items():
        out += str(item["median"]) + "\t"
    # new line
    out += "\n"
    # maybe save result in a file
    if isinstance(filename, str) and filesave is True:
        fname = filename.replace(".txt", "_avgmax.txt")
        with open(fname, "w", encoding=encoding) as f:
            f.write(out)
    if ifPrint:
        print(out)
    return out


def processSampleCellsMap(
    file, colorscale=None, figAx=None, pltClose=True, newGraphKwargs={}
):
    """Process a file containing summary of solar cell parameters for different cells,
    and plot maps of the different parameters."""
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
    for j, colsj in enumerate(cols):
        if colsj.startswith("Rsquare"):
            rsquare = content[j].y()
            # print("rsquare", j, rsquare)

    # combined plots
    graphs: List[Graph] = []
    # axisheights = [[], []]
    figsize = [6, 4]
    sadjust = [0.1, 0.1, 0.9, 0.9]
    # main loop
    for i, look in enumerate(colToPlot):
        AB = 0 if i < 4 else 1
        while len(graphs) <= AB:
            graphs.append(Graph("", **newGraphKwargs))
        col_found = False

        if look == "":  # blank space is desired
            graphs[AB].append(curveblank)
            continue

        for j, c in enumerate(cols):
            if not (
                c[: len(look)] == look
                and (
                    len(c) <= len(look)
                    or c[len(look)] in [" ", "_", "-", ".", "[", "]", "(", ")"]
                )
            ):
                continue

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
                    (
                        1
                        if rsquare[k] > 0.995
                        else (0.25 if rsquare[k] > 0.95 else 1 / 9)
                    )
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
            col_found = True
            break
        if not col_found:
            msg = "Warning processSampleCellsMap: column not found ({} or similar)"
            print(msg.format(look))
            graphs[AB].append(curveblank)

    # remove empty subplots at the end of each graph
    for graph in graphs:
        for i in range(len(graph) - 1, -1, -1):
            if graph[i] == curveblank:
                del graph[i]
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
        graphs[0].plot(filesave=filesave, fig_ax=figAx)
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
        graphs[1].plot(filesave=filesave, fig_ax=figAx)
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
    """Plot a map of solar cell parameters for different cells."""
    # sizefactor: float, or np.array for each item in values. e.g. 0.25 to reduce markersize
    size_cell = np.array([0.6, 0.6])
    margin = np.array([0.4, 0.4])

    newgraphkwargs = deepcopy(newGraphKwargs)
    newgraphkwargs.update({"silent": True})

    if len(values) == 0:
        return False

    # check cells are with correct form, i.e. 'a1'
    ticklabeldash = False
    x, y, val = [], [], []
    split = [research(r"([a-zA-Z#])([0-9]*)", c).groups() for c in cells]
    for i, spliti in enumerate(split):
        if len(spliti) == 2 and len(spliti[1]) > 0:
            # max(1, ...): to convert "#" into "a"
            x.append(max(1, float(ord(spliti[0].lower()) - 96)))
            y.append(float(spliti[1]))
            # prefer to work on a copy and not modifying the list values
            val.append(values[i])
            if spliti[0] == "#":
                ticklabeldash = True
        else:
            print("plotSampleCellsMap: cell name not legal:", split)
    if len(x) == 0:
        print("plotSampleCellsMap: empty list of cells.")
        return

    x, y, val = np.array(x), np.array(y), np.array(val)
    transpose = False
    if max(y) > 2 and max(x) == 1 and ticklabeldash:
        transpose = True
        x, y = y, x

    if title == "Voc [V]":
        title = "Voc [mV]"
        val *= 1000
    replacement = {
        "Voc_V": "Voc [V]",
        "Jsc_mApcm2": "Jsc [mA/cm2]",
        "FF_pc": "FF [%]",
        "Eff_pc": "Eff. [%]",
        "_Ohmcm2": " [Ohmcm2]",
        "A/cm2": "A cm$^{-2}$",
        "J0 ": "J$_0$ ",
        "Ohmcm2": "$\\Omega$ cm$^2$",
    }
    for old, new in replacement.items():
        title = title.replace(old, new)

    val_norm = val if not (val == 0).all() and not len(val) == 1 else [0.5] * len(val)
    if isinstance(colorscale, Colorscale):
        colorscale = colorscale.get_colorscale()
    if isinstance(colorscale, str):
        # 'autumn' is a string, but a color list might not have been recognized as list
        colorscale = strToVar(colorscale)
    if colorscale is None:
        colorscale = [[1, 0, 0], [1, 1, 0.5], [0, 1, 0]]
    if inverseScale and isinstance(colorscale, list):
        colorscale = colorscale[::-1]

    xticks = np.arange(0, max(x) + 1, 1)  # if max(x) > 6 else np.arange(0,max(x)+1,1)
    yticks = np.arange(0, max(y) + 1, 1)
    ax_size = np.array(
        [
            size_cell[0] * (max(xticks) - min(xticks)),
            size_cell[1] * (max(yticks) - min(yticks)),
        ]
    )
    figsize = ax_size + 2 * margin[1]
    marg = margin / figsize
    txt_coords = np.transpose(
        [
            (x - 0.5 - min(xticks)) / (max(xticks) - min(xticks)),
            (y - 0.5 - min(yticks)) / (max(yticks) - min(yticks)),
        ]
    )
    to_print = [roundSignificant(v, 3) for v in val]
    if np.average(val) > 1e2:
        to_print = ["{:1.0f}".format(v) for v in to_print]
    if np.average(val) < 1e-3:
        to_print = ["{:.1E}".format(v).replace("E-0", "E-") for v in to_print]
    if title.startswith("Rsquare"):  # more digits for Rsquare
        to_print = [roundSignificant(v, 5) for v in val]
    markersize = (size_cell[0] * 72) ** 2
    val_size = np.array([markersize for val in x])
    val_size *= sizefactor
    graph = Graph("", **newgraphkwargs)
    texttxt = []
    textarg = []
    for i in range(len(val)):
        texttxt.append(to_print[i])
        textarg.append(
            {
                "xytext": list(txt_coords[i]),
                "xycoords": "axes fraction",
                "horizontalalignment": "center",
                "verticalalignment": "center",
            }
        )

    curveattr = {
        "type": "scatter",
        "marker": "s",
        "markeredgewidth": 0,
        "cmap": colorscale,
        # 'markersize': (sizeCell[0]*72)**2,
    }
    graph.append(Curve([x - 0.5, y - 0.5], curveattr))
    graph.append(Curve([x - 0.5, val_norm], {"type": "scatter_c"}))
    graph.append(Curve([x - 0.5, val_size], {"type": "scatter_s"}))

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
    fct = [
        ["set_xticks", [list(xticks)], {}],
        ["set_yticks", [list(yticks)], {}],
        ["set_xticklabels", [[]], {}],
        ["set_yticklabels", [[]], {}],
        ["set_xticks", [list(xticks[1:] - 0.5)], {"minor": True}],
        ["set_yticks", [list(yticks[1:] - 0.5)], {"minor": True}],
    ]
    letterticks_base = yticks if transpose else xticks
    letterticks = [[chr(int(i) - 1 + ord("a")) for i in letterticks_base[1:]]]
    if ticklabeldash:
        letterticks[0] = "#"
    if transpose:
        fct.append(["set_yticklabels", letterticks, {"minor": True}])
        fct.append(["set_xticklabels", [[int(i) for i in xticks[1:]]], {"minor": True}])
    else:  # usual case
        fct.append(["set_xticklabels", letterticks, {"minor": True}])
        fct.append(["set_yticklabels", [[int(i) for i in yticks[1:]]], {"minor": True}])

    fct.append(["tick_params", [], {"axis": "both", "which": "minor", "length": 0}])
    fct.append(["grid", [True], {}])
    graph.update({"arbitraryfunctions": fct})

    if filesave is not None:
        graph.headers.update({"filesave": os.path.basename(filesave)})
        # graph.plot(filesave, figAx=figAx)  # plot -> would need to plt.close() accordingly
        graph.export(filesave)  # export only txt file, and not the image
    else:
        graph.plot(fig_ax=figAx, if_subplot=True)
        if pltClose and figAx is None:
            plt.close()
    return graph


def standalone():
    """Standalone function to test the script"""
    # go through files, store files content in order to later select pairs
    folder = "./../examples/JV/SAMPLE_A/"

    processJVfolder(folder, fitDiodeWeight=5, pltClose=True, groupCell=True)
    # processJVfolder(folder, groupCell=True, fitDiodeWeight=5, pltClose=False)

    # file = r'./../examples/JV\SAMPLE_B_3layerMo\export_sample_b_3layermo_summary_illum.txt'
    # processSampleCellsMap(file, pltClose=True)

    # writeFileAvgMax(file)

    plt.show()


if __name__ == "__main__":
    standalone()
