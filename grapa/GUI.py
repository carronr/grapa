# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

print("Loading numpy...")
import os
import numpy as np
import sys
import contextlib

print("Loading matplotlib...")
import matplotlib.pyplot as plt

print("Loading tkinter...")
import tkinter as tk
from tkinter import BOTH, X, Y
import tkinter.filedialog

print("Loading grapa...")
path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if path not in sys.path:
    sys.path.append(path)
from grapa import __version__
from grapa.graph import Graph
from grapa.graphIO import FILEIO_GRAPHTYPE_GRAPH
from grapa.curve import Curve
from grapa.gui.observable import Observable
from grapa.gui.GUIMainElements import (
    GUIFrameMenuMain,
    GUIFrameConsole,
    GUIFrameCentral,
    GUIFrameTemplateColorize,
    GUIFrameActionsGeneric,
    GUIFrameTree,
    GUIFramePropertyEditor,
    GUIFrameActionsCurves,
)


# WISHLIST:
# - config autotemplate for boxplot
# - B+W preview
# - Write workflow Cf
# - Write workflow CV
# - Write workflow JV
# - Write workflow Jsc Voc
# - open config file in OS default editor

# TODO
# - LONGTERM check import .graph instead of grapa. Look at standard python packages. Check from GUI, with spyder various versions, and when calling script from external python script (Note: gui Change curve type was failing in early test, when importing in different manner)
# - LONGTERM: get rid of graph.sampleInfo
# - progressively remove all uses of pyplot, does not mix well with tkagg and stuff
# - CurveIMage colorbar, make it as dropdown combobox
# - Stackplot: transparency Alpha ?
# - scatter, add text values for each point. [clear all annotations] [text formatting "{.2g}". Remove only] [kwargs: centered both]

# BUGS
# TODO export/copytoclipboard with alter (screen data), fails when several curves with same x coordinates. to check: if no x transform; if x transform all the same; if x transform makes different x values out of same x input. Possible solution: no compact mode when x transform.
# TODO: bug can prevent ERE to complete ??? SAMPLE???


# 0.6.4.0 ONGOING
# Additions
# - GUI: Added an option to modify the graph background color for visualization purpose. The output image remains with a transparent background, to be provided separately.
# - GUI: Added a button to reverse the curve order. Ignores curve selection.
# - Curve EQE: Bandgap derivative, added calculation of the bandgap PV by Rau (https://doi.org/10.1103/PhysRevApplied.7.044016), by averaging the energy weighted by EQE derivative value, over the derivative FWHM. Its value is slightly more sensitive to experimental noise than the derivative method. When the derivative peak is asymmetric, the value tends to be slightly higher than the derivative peak (up to 30meV ?).
# - Curve EQE: Bandgap derivative, also added a fit to the derivative suited to best estimation of sigma value, intended for independent estimation of DeltaVoc_rad.
# - Curve EQE: Added new function to calculate the short-circuit Voc loss, due to Jsc < Jsc_SQ. Calculation following Rau et al PRA 7 044016 (2017) DOI: https://doi.org/10.1103/PhysRevApplied.7.044016
# - Curve EQE: full revision of function ERE. Added the calculation Qe-LED by Rau (equivalent to ERE with geometrical factor fg=1), and the Voc bloss breackdown into DeltaVoc short-circuit, radiative, non-radiative. The center-of-mass of the PL peak (EQE*blackbody) is also provided for comparison purposes. The bandgap PV of Rau is used for calculations, or given by user. Note: changes in input bandgap is mostly accomodated in the DeltaVoc_rad value. Added auxiliary Curves to visualize data extraction fits. Added auxiliary Curve for parameter copy-paste.
# - Curve Image: Can now configure levels using arange, linspace, logspace, geomspace. The parameter extend can also be set.
# - Graph PLQY: implemented reading of output files of power dependency module.
# - Script CVfT: From a set of C-f data acquired at different voltages and temperatures, provides C-V-f maps for each temperatures indicating low values for phase, as well as C-f, C-V with T and C-V with Hz plots.
# - Script Cf: added Bode plot, |Z| and impedance angle versus log(frequency)
# - Script Boxplot: Added summary graph presenting all generated boxplots
# Modifications
# - General: Conversion nm-eV is now calculated from Plank and c constants (previously: 1239.5, now about 1239.8419)
# - General: graph colorize modifed behavior: if sameIfEmptyLabel, same colors are also applied in case label is hidden (labelhide)
# - Curve EQE: Revised parametrization of bandgap by derivative peak method. The fit is now parametrized in unit of eV.
# - Graph PLQY: when opening a file, added PLQY(time) as curve hidden by default
# - Graph TinyTusker: various improvements
# - Script Cf, image derivative: redesigned the image. The axes are now omega versus 1000/T (input data are in K, calulated on-the-fly with alter keyword). The fit curves of activation energies can be directly added onto the C-f derivative image.
# - Script JV: Rp, Rs from acquisition software are now reported in summary files and in graphical summary (diode).
# - Script JV: Rsquare fit quality restricted to the diode region is reported in the summary files and in graphical summary (diode). The marker size of the other fit parameters shrinks in case poor Rsquare values were obtained.
# - Script Correlation: Improved detection of input parameters varied in a logarithmic manner.
# - Script Correlation: Revised colorscale of plot "parseddata" for datasets with 2 input parameters
# Bug corrections
# - General: Solved a bug that prevented making figures with a unique subplot
# - General: The property xtickslabels and ytickslabels can now be used also in conjunction with the property alter.
# - General: Plot type fill_between and fill_betweenx now have more proper behavior.
# - GUI: Small adjustments against MacOS dark mode
# - GUI: Solved a bug that appeared when a tab was closed before the figure drawing was finished. Graphs drawn later on were not drawn correctly if contained several axes.
# Miscellaneous
# - General: Centralized physical constants in a unique file constants.py. Hopefully everything works as before.
# - Implementation: new text files to store content of (now renamed) variables Graph.dataInfoKeysGraphData, Graph.graphInfoKeysData, Graph.headersKeys
# - Implementation: tidy up the code at a number of places


# to be able to print in something else than default console
@contextlib.contextmanager
def stdout_redirect(where):
    cp = sys.stdout
    sys.stdout = where
    try:
        yield where
    finally:
        sys.stdout = cp  # sys.stdout = sys.__stdout__#does not work apparently


class Application(tk.Frame):
    DEFAULT_SCREENDPI = 72

    def __init__(self, master):
        self.master = master
        super().__init__(master)
        self.master.title("Grapa software v" + __version__)
        self.initiated = False
        self.newGraphKwargs = {"silent": True}
        try:  # possibly retrieve arguments from command line
            self.newGraphKwargs.update({"config": sys.argv[1]})
            # argv[1] is the config.txt file to be used in this session
        except Exception:
            pass
        # create observable before UI -> element can register at initialization
        self.observables = {}
        # handles some GUI changes when changing the selected curve
        self.observables["focusTree"] = Observable()
        # there is 1 other observable, buried in the tabs. See below.
        # start define GUI
        self.pack(fill=BOTH, expand=True)
        self.fonts = {}
        self.initFonts(self)
        self.createWidgets(self)
        self.getCanvas().draw()  # to resize canvas, for autoDPI to work
        self.master.update()
        # open default Graph
        filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "examples",
            "subplots_examples.txt",
        )
        self.openFile(filename)
        self.updateUI()
        # register updates of UI when user changes tabs
        self.frameCentral.frameGraph.tabs.register(self.callback_tabChanged)
        # .initiated is a flag to allow update of UI, set True after init of UI
        self.initiated = True
        # other keys are bound to master within menuLeft or elsewhere
        self.master.bind("<Control-r>", lambda e: self.updateUI())
        # TODO
        # print('GraphSIMS: check ratios are ok, also when not same number of traces above and below fraction')

    def initFonts(self, frame):
        import tkinter.font as font

        a = tk.Label(frame, text="")
        self.fonts["bold"] = font.Font(font=a["font"])
        self.fonts["bold"].configure(weight="bold")
        self.fonts["fg_default"] = a.cget("fg")

    def createWidgets(self, frame):
        # right frame
        fr = tk.Frame(frame)
        fr.pack(side="right", fill=Y, pady=2, expand=False)
        self.fillUIFrameRight(fr)
        # frame left/graph/console
        fr = tk.Frame(frame)
        fr.pack(side="left", anchor="n", fill=BOTH, expand=True)
        self.fillUIFrameMain(fr)

    def fillUIFrameMain(self, frame):
        # create console, indicators
        self.frameConsole = GUIFrameConsole(frame, self)
        self.frameConsole.frame.pack(side="bottom", fill=X)
        # create graph and left menu
        fr = tk.Frame(frame)
        fr.pack(side="top", fill=BOTH, expand=True)
        self.menuLeft = GUIFrameMenuMain(fr, self)
        self.menuLeft.frame.pack(side="left", anchor="n", fill=Y, pady=2)
        self.frameCentral = GUIFrameCentral(fr, self, relief="raised", borderwidth=2)
        self.frameCentral.frame.pack(
            side="left", anchor="n", fill=BOTH, pady=2, expand=True
        )

    def fillUIFrameRight(self, frame):
        pady = 5
        # template actions
        self.frameTplCol = GUIFrameTemplateColorize(frame, self)
        self.frameTplCol.frame.pack(side="top", fill=X, anchor="w", pady=pady)
        # properties
        self.frameTree = GUIFrameTree(frame, self)
        self.frameTree.frame.pack(side="top", fill=X, anchor="w", pady=pady)
        # NEW property
        self.frameProp = GUIFramePropertyEditor(frame, self)
        self.frameProp.frame.pack(side="top", fill=X, anchor="w", pady=pady)
        # actions on Curves
        self.frameActGen = GUIFrameActionsGeneric(frame, self)
        self.frameActGen.frame.pack(side="top", fill=X, anchor="w", pady=pady)
        # Actions on curves
        self.frameActCrv = GUIFrameActionsCurves(frame, self)
        self.frameActCrv.frame.pack(side="top", fill=X, anchor="w", pady=pady)

    # updateUI function
    def updateUI(self):
        # print('updateUI main')
        # import time
        sectionstoupdate = [
            [self.frameTplCol, "section Template & Colorize"],
            [self.frameTree, "Treeview property box"],
            [self.frameProp, "Property editor"],
            [self.frameActGen, "section Actions of Curves "],
            [self.frameActCrv, "section Actions specific"],
            [self.frameCentral, "central section (plot and editor)"],
            [self.frameConsole, "section Console"],
        ]
        for section in sectionstoupdate:
            # t0 = time.perf_counter()
            try:
                section[0].updateUI()
            except Exception as e:
                print("Exception during update of", section[1], ".")
                print("Exception", type(e), e)
                print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
            # t1 = time.perf_counter()
            # print('updateUI elapsed time:', t1-t0, section[1])

    # getters and setters
    def getAx(self):
        return self.frameCentral.frameGraph.ax

    def getFig(self):
        return self.frameCentral.frameGraph.fig

    def getCanvas(self):
        return self.frameCentral.frameGraph.canvas

    def graph(self, newgraph=None, **kwargs):
        """
        Get current Graph
        newgraph: to change current Graph. Does NOT refresh interface
        """
        tabs = self.frameCentral.frameGraph.tabs
        if newgraph is not None:
            if isinstance(newgraph, Graph):
                # by default, give same dpi for new plot as for previous
                if "dpi" not in kwargs:
                    kwargs["dpi"] = self.DEFAULT_SCREENDPI  # initial value
                tabs.appendNew(newgraph, **kwargs)
                tabs.select(-1)
                self.setAutoScreenDPI()
            else:
                print("WARNING: GUI.graph(), newgraph must be a Graph.")
        # return self.back_graph
        return tabs.getGraph()

    def getFolder(self, newvalue=None):
        """
        Get current active folder
        - newvalue: set newvalue for the current active folder
        """
        if newvalue is not None:
            self.frameCentral.frameGraph.tabs.updateCurrent(folder=newvalue)
        return self.frameCentral.frameGraph.tabs.getFolder()

    def getFile(self, newvalue=None):
        """
        Get current active file. Also updates folder and title
        - newvalue: set newvalue for the current active file
        """
        if newvalue is not None:
            self.frameCentral.frameGraph.tabs.updateCurrent(filename=newvalue)
            # also updates folder
        return self.frameCentral.frameGraph.tabs.getFilename()

    def getTabProperties(self, **kwargs):
        """
        Get properties of current active tab.
        - **kwargs: updates the properties of current active tab
        """
        if len(kwargs) > 0:
            self.frameCentral.frameGraph.tabs.updateCurrent(**kwargs)
            # also updates folder
        return self.frameCentral.frameGraph.tabs.getProperties()

    def getSelectedCurves(self, multiple=False):
        """Returns a list of unique, sorted indices [idx0, idx1, ...]"""
        curves, keys = self.frameTree.getTreeActiveCurve(multiple=multiple)
        return sorted(list(set(curves)))

    def getClipboard(self):
        """returns content of clipboard"""
        return self.master.clipboard_get()

    def setClipboard(self, data):
        """place new data in the clipboard"""
        self.master.clipboard_clear()
        self.master.clipboard_append(data)

    def setAutoScreenDPI(self):
        self.frameCentral.frameOptions.setAutoScreenDPI()

    def setScreenDPI(self, value):
        self.getTabProperties(dpi=value)
        self.frameCentral.frameOptions.checkValidScreenDPI()

    # callbacks
    def callback_tabChanged(self, *args, **kwargs):
        """react to change in tab selection"""
        # print('Tab changed triggers updateUI')
        if self.initiated:
            self.updateUI()

    def disableCanvasCallbacks(self):
        self.frameCentral.frameGraph.disableCanvasCallbacks()

    def enableCanvasCallbacks(self):
        self.frameCentral.frameGraph.enableCanvasCallbacks()

    # Other function to handle functionalities of applications
    def storeSelectedCurves(self):
        """
        Keeps in memory the selected curves. To be called in functions that
        will call updateUI(), before any action that may change the curves
        (esp. order, number, new Curves etc. Changes in attributes should be
        safe)
        """
        self.frameTree.storeSelectedCurves()

    @staticmethod
    def argsToStr(*args, **kwargs):
        """Format args and kwargs for printing commands in the console"""

        def toStr(a):
            if isinstance(a, str):
                return "'" + a.replace("\n", "\\n") + "'"
            elif isinstance(a, Graph):
                return "graph"
            elif isinstance(a, Curve):
                return "curve"
            return str(a)

        p = [toStr(a) for a in args]
        p += [(key + "=" + toStr(kwargs[key])) for key in kwargs]
        return ", ".join(p)

    def callGraphMethod(self, method, *args, **kwargs):
        """Execute 'method' on graph, with *args and **kwargs"""
        out = getattr(self.graph(), method)(*args, **kwargs)
        if self.ifPrintCommands():
            print("graph." + method + "(" + self.argsToStr(*args, **kwargs) + ")")
        return out

    def callCurveMethod(self, curve, method, *args, **kwargs):
        """Execute 'method' on graph[curve], with *args and **kwargs"""
        out = getattr(self.graph()[curve], method)(*args, **kwargs)
        if self.ifPrintCommands():
            msg = "graph[{}].{}({})"
            print(msg.format(str(curve), method, self.argsToStr(*args, **kwargs)))
        return out

    def promptFile(self, initialdir="", type="open", multiple=False, **kwargs):
        """Prompts for file open/save"""
        if initialdir == "":
            initialdir = self.getFolder()
        if type == "save":
            return tk.filedialog.asksaveasfilename(initialdir=initialdir, **kwargs)
        if multiple:
            out = list(tk.filedialog.askopenfilenames(initialdir=initialdir, **kwargs))
            if len(out) == 0:
                return None
            elif len(out) == 1:
                return out[0]
            return out
        # if not multiple
        return tk.filedialog.askopenfilename(initialdir=initialdir, **kwargs)

    def promptFolder(self, initialdir=""):
        """Prompts for folder open"""
        if initialdir == "":
            initialdir = self.getFolder()
        return tk.filedialog.askdirectory(initialdir=initialdir)

    def ifPrintCommands(self):
        return self.menuLeft.varPrintCommands.get()

    def openFile(self, file):
        """
        Open a file
        - file: a str, or a list of str to open multiple files, or a Graph
        """
        lbl = file
        if isinstance(file, Graph):
            # do not print anything
            graph = file
            lbl = ""
            if hasattr(graph, "fileexport"):
                lbl = graph.fileexport
            elif hasattr(graph, "filename"):
                lbl = graph.filename
        elif isinstance(file, list):
            print("Open multiple files (first:", lbl[0], ")")
            lbl = file[0]
            graph = Graph(file, **self.newGraphKwargs)
        else:
            print("Open file:", lbl.replace("/", "\\"))
            graph = Graph(file, **self.newGraphKwargs)
        if lbl == "":
            lbl = None
        self.graph(newgraph=graph, filename=lbl)
        if self.ifPrintCommands():
            print("graph = Graph('" + str(file) + "')")
        # updateUI triggerd by apparition (and selection) and new tab
        pass

    def mergeGraph(self, graph):
        """graph can be a str (filename), or a Graph"""
        file = ""
        if not isinstance(graph, Graph):
            file = graph
            print("Merge with file:", file)
            graph = Graph(graph)
        else:
            pass  # stays silent
            # print('Merge with graph', file)
        self.graph().merge(graph)
        # change default save folder only if was empty
        if self.getFolder() == "" and file != "":
            self.getFolder(file)
        if self.ifPrintCommands():
            print("graph.merge(Graph('" + file + "'))")
        self.updateUI()

    def saveGraph(self, filesave, fileext="", saveAltered=False, ifCompact=True):
        """
        Saves Graph into a file. Separated plot() and export() calls.
        filename: str, filename to save gr graph
        fileext: str, image data format (e.g. .png). If '', retrieve preference.
        saveAltered: as per Graph.export(). Default False.
        ifCompact: as per Graph.export(). Default True.
        """
        graph = self.graph()
        if graph.attr("meastype") == "":
            graph.update({"meastype": FILEIO_GRAPHTYPE_GRAPH})
        try:
            fig, ax = graph.plot(
                filesave=filesave,
                imgFormat=fileext,
                ifExport=False,
                figAx=[self.getFig(), None],
            )
            while isinstance(ax, (list, np.ndarray)) and len(ax) > 0:
                ax = ax[0]
        except Exception as e:
            print("ERROR: Exception during plotting of the Graph.")
            print(type(e), e)
        if fileext in [".xml"]:
            filesave += fileext
        graph.export(filesave=filesave, saveAltered=saveAltered, ifCompact=ifCompact)
        self.getFile(filesave)  # updates file, folder and tab title
        if self.ifPrintCommands():
            msg = "graph.plot(filesave='{}', imgFormat='{}', ifExport=False))"
            print(msg.format(filesave, fileext))
            msg = "graph.export(filesave='{}', saveAltered='{}', ifCompact='{}')"
            print(msg.format(filesave, str(saveAltered), str(ifCompact)))
        self.updateUI()

    def insertCurveToGraph(self, curve, updateUI=True):
        """Appends 1 Curve, or a list of Curves to the active graph"""
        if updateUI:
            self.storeSelectedCurves()
        if isinstance(curve, list):
            for c in curve:
                self.insertCurveToGraph(c, updateUI=False)
        elif isinstance(curve, Curve):
            kw = {}
            selected = self.getSelectedCurves(multiple=False)
            if len(selected) > 0 and selected[0] >= 0:
                kw.update({"idx": selected[0] + 1})
            self.graph().append(curve, **kw)
        else:
            msg = "ERROR GUI graphAppendCurve, could not handle class of curve {}."
            print(msg.format(type(curve)))
        if updateUI:
            self.updateUI()

    def blinkWidget(
        self, field, niter, delay=500, property_="background", values=["", "red"]
    ):
        if "" in values:  # if default value - retrieve current setting
            values = list(values)  # work on duplicate
            for i in range(len(values)):
                if values[i] == "":
                    values[i] = field.cget(property_)

        field.config(**{property_: values[niter % len(values)]})
        if niter > 0:
            kwargs = {"property_": property_, "values": values, "delay": delay}
            self.master.after(
                delay, lambda: self.blinkWidget(field, niter - 1, **kwargs)
            )

    def printLastRelease(self):
        file = "versionNotes.txt"
        file = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
        try:
            f = open(file, "r")
        except Exception:
            # print('Exception file open', e)
            return
        out = ""
        line = ""
        flag = True
        while flag:
            last = line
            line = f.readline()
            if "Release" in line:
                out = last + line
                flag = False
        if line == "":
            print('Not found string "Release"')
            return  # not found release
        date = line[len("Release") :].replace(" ", "")
        if date[0] == "d":
            date = date[1:]
        from dateutil import parser
        from datetime import datetime

        try:
            dayssincelast = datetime.now() - parser.parse(date)
        except ValueError as e:
            dayssincelast = ""
            print("Exception in PrintLastRelease, date", date, type(e), e)
        line = ""
        if dayssincelast.days < 5:
            flag = True
            while flag:
                last = line
                line = f.readline()
                if "Release" in line or "Version" in line:
                    flag = False
                if flag and last.strip() != "":
                    out += last
            print(out)
        f.close()
        return

    def quit(self):
        """Quits application"""
        plt.close(self.getFig())
        self.master.quit()
        self.master.destroy()


def buildUI():
    root = tk.Tk()
    try:
        root.iconbitmap("datareading.ico")
    except Exception:
        pass
    app = Application(master=root)

    # starts runnning programm
    with stdout_redirect(app.frameConsole.console):
        # retrieve content of last release
        try:
            app.printLastRelease()
        except Exception:
            pass
        # start execution
        app.mainloop()


if __name__ == "__main__":
    buildUI()
