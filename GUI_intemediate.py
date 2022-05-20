# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

print('Loading matplotlib...')
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as NavigationToolbar2Tk
except ImportError:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt
print('Loading tkinter...')
import tkinter as tk
from tkinter import ttk
from tkinter import BOTH, X, Y
import tkinter.filedialog

print('Loading os...')
import os
print('Loading numpy...')
import numpy as np
print('Loading copy')
import copy
print('Loading sys...')
import sys
import contextlib

print('Loading grapa...')
path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if path not in sys.path:
    sys.path.append(path)
from grapa.graph import Graph
from grapa.graphIO import FILEIO_GRAPHTYPE_GRAPH
from grapa.curve import Curve
from grapa.mathModule import is_number, stringToVariable, roundSignificant
from grapa.colorscale import Colorscale, PhotoImageColorscale
from grapa.observable import Observable, ObserverStringVarMethodOrKey
from grapa.gui.createToolTip import CreateToolTip
from grapa.gui.GUImisc import FrameTitleContentHide, FrameTitleContentHideHorizontal
from grapa.gui.GUImisc import TextWriteable
from grapa.gui.GUImisc import EntryVar, OptionMenuVar, CheckbuttonVar, ComboboxVar, LabelVar
from grapa.gui.GUImainelements import GUIFrameMenuMain, GUIFrameConsole

# WISHLIST:
# - config autotemplate for boxplot
# - B+W preview
# Stackplot: transparency Alpha ?

# BUG: fit JV, when unit in mV
# - Add button reverse curve order

# TODO: Write workflow Cf
# TODO: Write workflow CV
# TODO: Write workflow JV
# TODO: Write workflow Jsc Voc

# TODO LIST PRIORITY
# - colorbar: also shortcut for type image?


# TODO: GUI to handle multiple graphs2; graph handler, for multiple graph tabs
# TODO: folder auto when file is open / only after multigraph
# TODO BUG: fit JV: create JV curve not necessarily right after selected curve (because hidden/visible urves above?)
# TODO: CurveSpectrum: check order of actions
# TODO: check import .graph instead of grapa. Look at standard python packages. Check from GUI, with spyder various versions, and when calling script from external python script
# TODO: TRPL lifetime estimate (!not grapa)

# CURRENT VERSION
# Additions
# - CurveSpectrum: added currection function instrumental response 800nm
# - CurveEQE: added an additional reference EQE spectrum (Empa PI 20.8%). The data loading mechanisms is modified and new reference spectra are now easy to add - see file datatypes/EQE_referenceSpectra.
# - scriptCV: The warnings are reported at the end. Also, more robust processing of input files with slightly different formatting.
# - Popup Annotations: added a vertical scrollbar, changed the order of displayed elements
# - CurveXRF: added an annotation function to place labels for peak intensities according to the database https://xdb.lbl.gov/Section1/Table_1-3.pdf.



# Version 0.5.4.8
# Modifications
# - TRPL: the data type TRPL was  modified to properly load files generated using scripts.
# - Modified loading of preference file. Hopefully a bit faster when opening many files.
# - CurveJV: added data transform 'Log10 abs (raw)' (abs0), to plot the log of JV curve without Jsc subtraction
# - scriptJV: also exports a file with a few statistical quantities of JV data (average, mediam, max)
# - scriptJV: reduced amount of generated files. The individual summary images are not generated anymore as already plotted in combined format. Also, the different cells are exported as images (3x) and only once as text file.
# - prepare future update of the GUI to support several graphs simultaneously
# BUGS:
# - Mitigated bugs with Winpython version 3.9 (issues with matplotlib 3.3.9; hidden curves in grey due to changes in tk)



# Version 0.5.4.7
# Modifications
# - Adjusted width of GUI Curve action fields for CurveSIMS to prevent time-consuming redimensioning of the GUI interface.
# - The Colorize GUI function can now print its effect in the console
# Bugs
# - Solved an issue with twiny ylim values


"""
*** *** *** example of config.txt file *** *** ***
# only keywords (first word) matters: comment line are loaded but never refered
# to repeating keyword will overwrite content of first instance
# graph labels default unit presentation: [unit], (unit), / unit, or [], (), /
graph_labels_units	[]
# graph labels presence of symbols (ie: $C$ in 'Capacitance $C$ [nF]')
graph_labels_symbols	False
# path to inkscape executable, to export in .emf image format. Can be a string, or a list of strings
inkscape_path	["C:\Program Files\Inkscape\inkscape.exe", "G:\__CommonData\Software\_inkscape_standalone\32bits\App\Inkscape\inkscape.exe"]
# GUI default colorscales. Each colorscale is represented as a string (matplotlib colorscales), or a list of colors.
# Each color can be a [r,g,b] triplet, a [r,g,b,a] quadruplet, or a [h,l,s,'hls'] quadruplet. rgb, rgba, hls, hsv colorscape are supported.
GUI_colorscale00	[[1,0,0], [1,1,0.5], [0,1,0]]
GUI_colorscale01	[[1,0.3,0], [0.7,0,0.7], [0,0.3,1]]
GUI_colorscale02	[[1,0.43,0], [0,0,1]]
GUI_colorscale03	[[0.91,0.25,1], [1.09,0.75,1], 'hls']
GUI_colorscale04	[[0.70,0.25,1], [0.50,0.75,1], 'hls']
GUI_colorscale05	[[1,0,1], [1,0.5,1], [0.5,0.75,1], 'hls']
GUI_colorscale07	'inferno'
GUI_colorscale10	'gnuplot2'
GUI_colorscale11	'viridis'
GUI_colorscale12	'afmhot'
GUI_colorscale13	'YlGn'
# default saving image format
save_imgformat	.png
*** *** *** end of example *** *** ***
"""


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

    DEFAULT_SCREENDPI = 100

    def __init__(self, master=None):
        self.master = master
        tk.Frame.__init__(self, master)
        self.initiated = False
        self.newGraphKwargs = {'silent': True}
        try:  # possibly retrieve arguments from command line
            self.newGraphKwargs.update({'config': sys.argv[1]})
            # argv[1] is the config.txt file to be used in this session
        except Exception:
            pass
        # initialize Graph object to some value
        self.back_graph = Graph([[1, 2, 3, 4, 5, 6, 7, 8],
                                 [5, 6, 1, 3, 8, 9, 3, 5]],
                                '', **self.newGraphKwargs)
        # handles some GUI changes when changing the selected curve
        self.observableCurve = Observable()
        # handles some GUI changes when changing the active curve in the data picker
        self.observableDataPickerCurve = Observable()
        # start define GUI
        self.pack(fill=BOTH, expand=True)
        FrameMain = tk.Frame(self)
        FrameMain.pack(fill=BOTH, expand=True)
        self.initFonts(FrameMain)
        self.createWidgets(FrameMain)
        # open default Graph
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'examples', 'subplots_examples.txt')
        self.openFile(filename)
        # set initial screen resolution
        self.fieldScreenDPI.set(75)
        self.setScreenDpi()
        # update GUI
        self.updateUI()
        # self.updateUI()
        # self.varDataPickerCurve.set(0)  # most likely useless -> comment out
        # initiated only after creation of widgets and first update of GUI
        self.initiated = True
        # some keys already bound in self.menuLeft
        self.master.bind('<Control-r>', lambda e: self.updateUI())
        self.master.bind('<Control-Shift-C>', lambda e: self.curveDataToClipboard())
        self.master.bind('<Control-m>', lambda e: self.dataPickerSavePoint())
        self.master.bind('<Control-h>', lambda e: self.curveShowHideCurve())
        self.master.bind('<Control-Delete>', lambda e: self.curveDelete())

    def createFrame(self, frame, func, dictArgsInit, dictArgsPack, argsFunc=None, geometry='pack'):
        """
        All frames are generated according to this 3-line pattern
        1/ creation inside a know frame, 2/ pack, 3/ fill frame by executing a
        function with frame name as parameter
        Possible extension is to handle a different geometry manager
        """
        newFrame = tk.Frame(frame, **dictArgsInit)
        if geometry == 'grid':
            newFrame.grid(**dictArgsPack)
        else:
            newFrame.pack(**dictArgsPack)
        if func is not None:
            func(newFrame) if argsFunc is None else func(newFrame, *argsFunc)
        return newFrame

    def initFonts(self, frame):
        import tkinter.font as font
        a = tk.Label(frame, text='')
        self.fontBold = font.Font(font=a['font'])
        self.fontBold.configure(weight='bold')

    def createWidgets(self, frame):
        # right frame
        self.createFrame(frame, self.fillUIFrameRight, {'relief': 'raised', 'borderwidth': 2}, {'side': 'right', 'fill': Y, 'pady': 2, 'expand': False})
        # frame left/graph/console
        self.createFrame(frame, self.fillUIFrameMain, {}, {'side': 'left', 'anchor': 'n', 'fill': BOTH, 'expand': True})

    def fillUIFrameMain(self, frame):
        # create console, indicators
        self.panelConsole = GUIFrameConsole(frame, self)
        self.panelConsole.frame.pack(side='bottom', fill=X)
        # create graph and left menu
        self.createFrame(frame, self.fillUIFrameMainTop,    {}, {'side': 'top', 'fill': BOTH, 'expand': True})

    def fillUIFrameMainTop(self, frame):
        self.menuLeft = GUIFrameMenuMain(frame, self)
        self.menuLeft.frame.pack(side='left', anchor='n', fill=Y, pady=2)
        # left frame
        # self.createFrame(frame, self.fillUIFrameLeft,   {'relief': 'raised', 'borderwidth': 2}, {'side': 'left', 'anchor': 'n', 'fill': Y, 'pady': 2})
        # graph + options frame
        self.createFrame(frame, self.fillUIFrameCenter, {'relief': 'raised', 'borderwidth': 2}, {'side': 'left', 'anchor': 'n', 'fill': BOTH, 'pady': 2, 'expand': True})

    def fillUIFrameMainBottom(self, frame):
        self.varLabelFile = LabelVar(frame, value='')
        self.varLabelFile.pack(side='top', anchor='w')
        # current folder
        self.varLabelFolder = LabelVar(frame, value='')
        self.varLabelFolder.pack(side='top', anchor='w')
        # console
        self.Console = TextWriteable(frame, wrap='word', width=100, height=8)
        self.Console.pack(side='bottom', anchor='w')

    def fillUIFrameCenter(self, frame):
        # some container for graph
        self.createFrame(frame, self.fillUIFrameGraphBelow, {}, {'side': 'bottom', 'anchor': 'w', 'fill': X})
        self.createFrame(frame, self.fillUIFrameCenterTop, {}, {'side': 'bottom', 'anchor': 'w', 'fill': X})
        self.createFrame(frame, self.fillUIFrameGraphBox, {}, {'side': 'top', 'anchor': 'w', 'fill': BOTH, 'expand': True})
    def fillUIFrameGraphBox(self, frame):
        # canvas for graph
        figsize = Graph.FIGSIZE_DEFAULT
        figsize = [figsize[0], figsize[1]*1.15]
        self.Graph_fig = plt.figure(figsize=figsize, dpi=self.DEFAULT_SCREENDPI)
        self.canvas = FigureCanvasTkAgg(self.Graph_fig, master=frame)
        self.canvas.get_tk_widget().configure(background='white',
                                              width =1.03*Graph.FIGSIZE_DEFAULT[0]*self.DEFAULT_SCREENDPI,
                                              height=1.03*Graph.FIGSIZE_DEFAULT[1]*self.DEFAULT_SCREENDPI)
        self.canvas.get_tk_widget().pack(side='top', anchor='w', fill=tk.BOTH, expand=True)
        self.callback_eventCanvasPressed = False
        self.canvas.mpl_connect('resize_event', self.updateUponResizeWindow)
        self.canvasEvents = []
        # toolbar
        self.canvas._tkcanvas.pack(side='top', anchor='w', expand=True, fill=BOTH)  # fill=BOTH, expand=True
        self.createFrame(frame, self.fillUIFrameGraphBoxToolbar, {}, {'side': 'top', 'anchor': 'w', 'fill': X})
        try:
            self.canvas.show()
        except AttributeError:
            # handles FigureCanvasTkAgg has no attribute show in later versions
            # of matplotlib
            pass
        self.Graph_ax = self.Graph_fig.add_subplot(111)

    def enableCanvasCallbacks(self):
        def callback_pressCanvas(event):
            self.callback_eventCanvasPressed = True
            callback_notifyCanvas(event)

        def callback_releaseCanvas(event):
            self.callback_eventCanvasPressed = False

        def callback_notifyCanvas(event):
            if not self.callback_eventCanvasPressed:
                return
            # print('Clicked at', event.xdata, event.ydata, 'curve', self.varDataPickerRestrict.get(), self.varDataPickerCurve.get())
            xdata, ydata = event.xdata, event.ydata
            self.varDataPickerIdx = np.nan
            if (event.xdata is not None and self.varDataPickerRestrict.get()
                    and self.varDataPickerCurve.get() > -1):
                curve = self.graph()[self.varDataPickerCurve.get()]
                if curve is not None:
                    xdata, ydata, idx = curve.getPointClosestToXY(xdata, ydata, alter=self.graph().attr('alter'))
                    self.varDataPickerIdx = idx
            if xdata is not None:
                self.varDataPickerX.set(xdata)
            if ydata is not None:
                self.varDataPickerY.set(ydata)
            self.callback_updateCrosshair()
        self.disableCanvasCallbacks_()
        self.canvasEvents.append(self.canvas.mpl_connect('button_press_event', callback_pressCanvas))
        self.canvasEvents.append(self.canvas.mpl_connect('button_release_event', callback_releaseCanvas))
        self.canvasEvents.append(self.canvas.mpl_connect('motion_notify_event', callback_notifyCanvas))

    def disableCanvasCallbacks_(self):
        for cid in self.canvasEvents:
            self.canvas.mpl_disconnect(cid)

    def fillUIFrameGraphBoxToolbar(self, frame):
        u = tk.Button(frame, text='Refresh GUI', command=self.updateUI)
        u.pack(side='left', anchor='center', padx=5, pady=2)
        CreateToolTip(u, 'Ctrl+R')
        tk.Button(frame, text='Save zoom/subplots', command=self.setLimitsSubplotsToCurrent).pack(side='left', anchor='center', padx=5, pady=2)
        tk.Label(frame, text="   ").pack(side='left')
        self.toolbar = NavigationToolbar2Tk(self.canvas, frame)
        self.toolbar.update()

    def fillUIFrameGraphBelow(self, frame):
        self.createFrame(frame, self.fillUIFrameGraphQuickMods,  {}, {'side': 'top', 'anchor': 'w', 'fill': X})
        FrameTitleContentHideHorizontal(frame, self.fillUIFrameGraphDataPickerTitle, self.fillUIFrameGraphDataPicker,
                                        default='hide', showHideTitle=True).pack(side='top', fill=X, anchor='w')

    def fillUIFrameGraphDataPickerTitle(self, frame):
        tk.Label(frame, text='Data picker').pack(side='top')
    def fillUIFrameGraphDataPicker(self, frame):
        self.createFrame(frame, self.fillUIFrameGraphDataPickerUp, {}, {'side': 'top', 'anchor': 'w', 'fill': X})
        self.createFrame(frame, self.fillUIFrameGraphDataPickerDwn, {}, {'side': 'top', 'anchor': 'w', 'fill': X})
    def fillUIFrameGraphDataPickerUp(self, frame):
        tk.Label(frame, text='Click on graph').pack(side='left', anchor='c')
        self.varDataPickerX = tk.DoubleVar()
        self.varDataPickerY = tk.DoubleVar()
        self.varDataPickerRestrict = tk.BooleanVar()
        self.varDataPickerCurve = tk.IntVar()
        self.varDataPickerCurve.set(0)
        self.varDataPickerCurve.trace("w", self.callback_dataPickerCurve)
        self.varDataPickerCrosshair = tk.BooleanVar()
        self.varDataPickerIdx = np.nan
        tk.Label(frame, text='x').pack(side='left', anchor='c')
        tk.Entry(frame, text=self.varDataPickerX, width=10).pack(side='left', anchor='c')
        tk.Label(frame, text='y').pack(side='left', anchor='c')
        tk.Entry(frame, text=self.varDataPickerY, width=10).pack(side='left', anchor='c')
        tk.Checkbutton(frame, text='Restrict to data', variable=self.varDataPickerRestrict).pack(side='left', anchor='c')
        tk.Label(frame, text='curve').pack(side='left', anchor='c')
        self.optionmenuDataPickerCurve = tk.OptionMenu(frame, self.varDataPickerCurve, '')
        self.optionmenuDataPickerCurve.pack(side='left', anchor='c')
        tk.Checkbutton(frame, text='Crosshair', variable=self.varDataPickerCrosshair, command=self.callback_updateCrosshair).pack(side='left', anchor='c')

    def fillUIFrameGraphDataPickerDwn(self, frame):
        tk.Button(frame, text='Create text with coordinates', command=self.dataPickerToTextbox).pack(side='left', anchor='c')
        tk.Label(frame, text=' or ').pack(side='left')
        u = tk.Button(frame, text='Save point', command=self.dataPickerSavePoint)
        u.pack(side='left', anchor='c')
        CreateToolTip(u, 'Ctrl+M')
        self.varDataPickerSaveTransform = tk.BooleanVar()
        self.varDataPickerSaveTransform.set(True)
        tk.Checkbutton(frame, text='screen data', variable=self.varDataPickerSaveTransform).pack(side='left', anchor='c')
        self.varDataPickerSaveCurveSpec = tk.BooleanVar()
        self.varDataPickerSaveCurveSpec.set(False)
        tk.Checkbutton(frame, text='Curve specific', variable=self.varDataPickerSaveCurveSpec).pack(side='left', anchor='c')
        # explanatory text for checkbox
        ObsDataPickerSpecific = ObserverStringVarMethodOrKey(tk.StringVar(), '', 'getDataCustomPickerXY', methodArgs=[0], methodKwargs={'strDescription': True})
        self.observableDataPickerCurve.register(ObsDataPickerSpecific)
        tk.Label(frame, textvariable=ObsDataPickerSpecific.var()).pack(side='left')

    def callback_dataPickerCurve(self, *args):
        # callback get some arguments which we are not using
        # the observer pattern is an overkill here, but it is slightly cleaner
        curve = self.varDataPickerCurve.get()
        self.observableDataPickerCurve.update_observers(self.graph()[curve] if curve > -1 else None)


    def fillUIFrameGraphQuickMods(self, frame):
        # tk.Label (frame, text='Quick modifs').pack(side='left', anchor='c')
        self.varQuickModsXlabel = EntryVar(frame, '', width=15)
        self.varQuickModsYlabel = EntryVar(frame, '', width=15)
        self.varQuickModsXlim0 = EntryVar(frame, '', width=5)
        self.varQuickModsXlim1 = EntryVar(frame, '', width=5)
        self.varQuickModsYlim0 = EntryVar(frame, '', width=5)
        self.varQuickModsYlim1 = EntryVar(frame, '', width=5)
        tk.Label(frame, text='xlabel:').pack(side='left', anchor='c')
        self.varQuickModsXlabel.pack(side='left', anchor='c')
        tk.Label(frame, text='   ylabel:').pack(side='left', anchor='c')
        self.varQuickModsYlabel.pack(side='left', anchor='c')
        tk.Label(frame, text='   xlim:').pack(side='left', anchor='c')
        self.varQuickModsXlim0.pack(side='left', anchor='c')
        tk.Label(frame, text='to').pack(side='left', anchor='c')
        self.varQuickModsXlim1.pack(side='left', anchor='c')
        tk.Label(frame, text='   ylim:').pack(side='left', anchor='c')
        self.varQuickModsYlim0.pack(side='left', anchor='c')
        tk.Label(frame, text='to').pack(side='left', anchor='c')
        self.varQuickModsYlim1.pack(side='left', anchor='c')
        tk.Label(frame, text='   ').pack(side='left')
        tk.Button(frame, text='Save', command=self.quickMods).pack(side='left', anchor='c')
        tk.Label(frame, text='   ').pack(side='left')
        self.varQuickModsXlabel.bind('<Return>', lambda event: self.quickMods())
        self.varQuickModsYlabel.bind('<Return>', lambda event: self.quickMods())
        self.varQuickModsXlim0.bind('<Return>', lambda event: self.quickMods())
        self.varQuickModsXlim1.bind('<Return>', lambda event: self.quickMods())
        self.varQuickModsYlim0.bind('<Return>', lambda event: self.quickMods())
        self.varQuickModsYlim1.bind('<Return>', lambda event: self.quickMods())
        tk.Button(frame, text='Data editor', command=self.popupDataEditor).pack(side='right', anchor='c')

    def fillUIFrameCenterTop(self, frame):
        tk.Label(frame, text='Data transform').pack(side='left', anchor='n', pady=7)
        self.alterListGUI = self.graph().alterListGUI()
        [idx, display, alter, typePlot] = self.plotChangeView_identify()
        self.OMAlter = OptionMenuVar(frame, [i[0] for i in self.alterListGUI],
                                     default=display, func=self.plotChangeView2)
        self.OMAlter.pack(side='left', anchor='center')
        # button change graph type
        tk.Label(frame, text='   Plot type').pack(side='left', anchor='n', pady=7)
        plotTypeList = ['', 'plot', 'plot norm.', 'semilogy', 'semilogy norm.',
                        'semilogx', 'loglog']
        self.OMTypePlot = OptionMenuVar(frame, plotTypeList, default='',
                                        func=self.plotTypePlot)
        self.OMTypePlot.pack(side='left')
        # popup to handle text annotations
        tk.Label(frame, text='     ').pack(side='left')
        tk.Button(frame, text='Annotations, legend and titles', command=self.popupAnnotations).pack(side='left', anchor='n', pady=5)
        # screen dpi
        self.createFrame(frame, self.fillUIFrameCenterTopDPI, {}, {'side': 'right', 'anchor': 'n'})


    def fillUIFrameCenterTopDPI(self, frame):
        tk.Label(frame, text='Screen dpi').pack(side='left', anchor='n', pady=7)
        self.fieldScreenDPI = EntryVar(frame, value=self.DEFAULT_SCREENDPI, width=5)
        self.fieldScreenDPI.pack(side='left', anchor='n', pady=8, padx=5)
        self.fieldScreenDPI.bind('<Return>', lambda event: self.setScreenDpi())
        tk.Button(frame, text='Save', command=self.setScreenDpi).pack(side='left', anchor='n', pady=5)


    def fillUIFrameRight(self, frame):
        pady = 5
        # template actions
        FrameTitleContentHide(frame, self.fillUIFrameRightTemplateColTitle, self.fillUIFrameRightTemplateColorize, default='hide', contentkwargs={'padx': 10}).pack(side='top', fill=X, anchor='w', pady=pady)
        # properties
        self.previousSelectedCurve = [-1]
        FrameTitleContentHide(frame, self.fillUIFrameRightTreeTitle, self.fillUIFrameRightTreeContent).pack(side='top', fill=X, anchor='w', pady=pady)
        # NEW property
        FrameTitleContentHide(frame, self.fillUIFrameRightNewTitle, self.fillUIFrameRightNew, contentkwargs={'padx': 10}).pack(side='top', fill=X, anchor='w', pady=pady)
        # actions on Curves
        FrameTitleContentHide(frame, self.fillUIFrameRightActionGenericTitle, self.fillUIFrameRightActionGeneric).pack(side='top', fill=X, anchor='w', pady=pady)
        # Actions on curves
        self.fillUIFrameRightActionTitle(frame)
        self.FrameRActionList = []
        self.FrameRAction = self.createFrame(frame, self.fillUIFrameRightAction, {}, {'side': 'top', 'fill': X, 'padx': '10', 'anchor': 'w', 'pady': pady}, argsFunc=[-1])
        # list of curves - Listbox choice to user
        self.ListBoxCurves = tk.Listbox(frame, width=30, height=4)  # dont want to delete the object as corresponding code can be useful at some point
        # self.ListBoxCurves.pack(side='top', anchor='w')


    def fillUIFrameRightTreeTitle(self, frame):
        tk.Label(frame, text='List of properties', font=self.fontBold).pack(side='top', anchor='w')
    def fillUIFrameRightTreeContent(self, frame):
        self.createFrame(frame, self.fillUIFrameRightTree, {}, {'side': 'top', 'fill': X, 'padx': 5})
        # EDIT property
        fr = FrameTitleContentHide(frame, self.fillUIFrameRightEditTitle, self.fillUIFrameRightEdit, contentkwargs={'padx': 10}, default='hide')
        # fr.pack(side='top', fill=X, anchor='w')

    def fillUIFrameRightTree(self, frame):
        # START OF FIX: because of change in tk. Baseically added the following
        def fixed_map(option):
            # Returns the style map for 'option' with any styles starting with
            # ("!disabled", "!selected", ...) filtered out
            # style.map() returns an empty list for missing options, so this
            # should be future-safe
            return [elm for elm in style.map("Treeview", query_opt=option)
                    if elm[:2] != ("!disabled", "!selected")]
        style = ttk.Style()
        style.map("Treeview", foreground=fixed_map("foreground"),
                  background=fixed_map("background"))
        # END OF FIX
        self.Tree = ttk.Treeview(frame, columns=('#1'))
        self.Tree.pack(side='left', anchor='n')
        self.Treeysb = ttk.Scrollbar(frame, orient='vertical', command=self.Tree.yview)
        self.Treeysb.pack(side='right', anchor='n', fill=Y)
        self.Tree.configure(yscroll=self.Treeysb.set)
        self.Tree.column('#0', width=170)
        self.Tree.heading('#0', text='Name')
        self.Tree.column('#1', width=290)
        self.Tree.heading('#1', text='Value')
        self.Tree.bind('<<TreeviewSelect>>', self.treeSelectItem)

    def fillUIFrameRightActionGenericTitle(self, frame):
        tk.Label(frame, text='Actions on Curves', font=self.fontBold).pack(side='left', anchor='n')
    def fillUIFrameRightActionGeneric(self, frame):
        self.createFrame(frame, self.fillUIFrameRightAGLeft,  {}, {'side': 'left', 'anchor': 'n', 'padx': '10'})
        self.createFrame(frame, self.fillUIFrameRightAGRight, {}, {'side': 'right', 'anchor': 'n'})

    def fillUIFrameRightAGLeft(self, frame):
        self.createFrame(frame, self.fillUIFrameRightAGShift,     {}, {'side': 'top', 'anchor': 'w'})
        self.createFrame(frame, self.fillUIFrameRightAGDelete,    {}, {'side': 'top', 'anchor': 'w'})
        self.createFrame(frame, self.fillUIFrameRightAGDuplicate, {}, {'side': 'top', 'anchor': 'w'})
        self.createFrame(frame, self.fillUIFrameRightAGShowHide,  {}, {'side': 'top', 'anchor': 'w'})

    def fillUIFrameRightAGRight(self, frame):
        self.createFrame(frame, self.fillUIFrameRightAGClipboard, {}, {'side': 'top', 'anchor': 'w'})
        self.createFrame(frame, self.fillUIFrameRightAGCast,      {}, {'side': 'top', 'anchor': 'w'})
        self.createFrame(frame, self.fillUIFrameRightAGQuickAttr, {}, {'side': 'top', 'anchor': 'w'})
        self.createFrame(frame, self.fillUIFrameRightAGLabelReplace, {}, {'side': 'top', 'anchor': 'w'})

    def fillUIFrameRightAGDelete(self, frame):
        tk.Label(frame, text='Delete Curve').pack(side='left')
        de = tk.Button(frame, text='Curve', command=self.curveDelete)
        de.pack(side='left', padx=3)
        tk.Button(frame, text='All hidden', command=self.curveDeleteAllHidden).pack(side='left')
        CreateToolTip(de, "Ctrl+Delete")


    def fillUIFrameRightAGDuplicate(self, frame):
        tk.Label(frame, text='Duplicate Curve').pack(side='left')
        tk.Button(frame, text='Duplicate', command=self.curveDuplicate).pack(side='left')

    def fillUIFrameRightAGClipboard(self, frame):
        tk.Label(frame, text='Copy to clipboard').pack(side='left')
        tk.Button(frame, text='Curve', command=self.curveDataToClipboard).pack(side='left', padx='5')
        tk.Button(frame, text='Graph', command=self.graphDataToClipboard).pack(side='left')
        vals = ['raw', 'with properties', 'screen data', 'screen data, prop.']
        self.OptMenuToClipboardAttr = OptionMenuVar(frame, vals, default='options', width=6)
        self.OptMenuToClipboardAttr.pack(side='left')
#        self.ChkBtnToClipboardAttr = CheckbuttonVar(frame, 'props', False)
#        self.ChkBtnToClipboardAttr.pack(side='left')
#        self.ChkBtnToClipboardAlter = CheckbuttonVar(frame, 'screen data', False)
#        self.ChkBtnToClipboardAlter.pack(side='left')

    def fillUIFrameRightAGShowHide(self, frame):
        ObsShowHideCurve = ObserverStringVarMethodOrKey(tk.StringVar(), 'Select Curve', 'isHidden', valuesDict={True: 'Show Curve', False: 'Hide Curve'})
        self.observableCurve.register(ObsShowHideCurve)
        sh = tk.Button(frame, textvariable=ObsShowHideCurve.var(), command=self.curveShowHideCurve)
        sh.pack(side='left')
        tk.Button(frame, text='All',    command=self.curveShowHideAll).pack(side='left', padx='5')
        tk.Button(frame, text='Invert', command=self.curveShowHideInvert).pack(side='left')
        CreateToolTip(sh, "Ctrl+H")


    def fillUIFrameRightAGShift(self, frame):
        tk.Label(frame, text='Reorder').pack(side='left')
        tk.Button(frame, text=u"\u21E7", command=self.curveShiftTop).pack(side='left', padx=1)
        tk.Button(frame, text=u"\u21D1 Up", command=self.curveShiftUp).pack(side='left', padx=1)
        tk.Button(frame, text=u"\u21D3 Down", command=self.curveShiftDown).pack(side='left', padx=1)
        tk.Button(frame, text=u"\u21E9", command=self.curveShiftBott).pack(side='left', padx=1)

    def fillUIFrameRightAGCast(self, frame):
        tk.Label(frame, text='Change Curve type').pack(side='left')
        self.castCurveList = []
        self.varCastCurve = tk.StringVar()
        self.varCastCurve.set('')
        self.MenuCastCurve = tk.OptionMenu(frame, self.varCastCurve, '')
        self.MenuCastCurve.pack(side='left', padx='2')
        tk.Button(frame, text='Save', command=self.castCurve).pack(side='left')

    def chooseColor(self):
        from tkinter.colorchooser import askcolor
        curve = self.getActiveCurve(multiple=False)
        if curve != -1:
            from matplotlib.colors import hex2color, rgb2hex
            try:
                colorcurrent = rgb2hex(stringToVariable(self.VarRAGQAColor.get()))
            except Exception:
                colorcurrent = None
            ans = askcolor(color=colorcurrent)
            if ans[0] is not None:
                self.ObsRAGQAColor.set(self.listToString([np.round(a, 3) for a in hex2color(ans[1])]))  # [np.round(val/256,3) for val in ans[0]]
    def fillUIFrameRightAGQuickAttr(self, frame):
        VarRAGQALabel = tk.StringVar()
        self.VarRAGQAColor = tk.StringVar()
        self.ObsRAGQALabel = ObserverStringVarMethodOrKey(VarRAGQALabel, '', 'label')
        self.ObsRAGQAColor = ObserverStringVarMethodOrKey(self.VarRAGQAColor, '', 'color')
        self.observableCurve.register(self.ObsRAGQALabel)
        self.observableCurve.register(self.ObsRAGQAColor)
        tk.Label(frame, text='Label').pack(side='left')
        a = tk.Entry(frame, text=VarRAGQALabel, width=15)
        a.pack(side='left')
        tk.Label(frame, text='Color').pack(side='left')
        tk.Button(frame, text='Pick', command=self.chooseColor).pack(side='left')
        b = tk.Entry(frame, text=self.VarRAGQAColor, width=8)
        b.pack(side='left')
        tk.Button(frame, text='Save', command=self.AGQuickAttrSet).pack(side='left')
        a.bind('<Return>', lambda event: self.AGQuickAttrSet())
        b.bind('<Return>', lambda event: self.AGQuickAttrSet())
    def AGQuickAttrSet(self):
        curves = self.getActiveCurve()
        if len(curves) > 0:
            self.updateProperty(curves, 'label', self.ObsRAGQALabel.get(), ifUpdate=False, varType=str)
            self.updateProperty(curves, 'color', self.ObsRAGQAColor.get(), ifUpdate=False)
            self.updateUI()
    def fillUIFrameRightAGLabelReplace(self, frame):
        self.varReplaceLabelsOld = tk.StringVar()
        self.varReplaceLabelsOld.set('old string')
        self.varReplaceLabelsNew = tk.StringVar()
        self.varReplaceLabelsNew.set('new string')
        tk.Label(frame, text='Replace in labels').pack(side='left')
        a = tk.Entry(frame, textvariable=self.varReplaceLabelsOld, width=10)
        a.pack(side='left')
        b = tk.Entry(frame, textvariable=self.varReplaceLabelsNew, width=10)
        b.pack(side='left')
        tk.Button(frame, text='Replace',  command=self.graphReplaceLabels).pack(side='left')
        a.bind('<Return>', lambda event: self.graphReplaceLabels())
        b.bind('<Return>', lambda event: self.graphReplaceLabels())

    def fillUIFrameRightTemplateColTitle(self, frame):
        tk.Label(frame, text='Template & Colorize', font=self.fontBold).pack(side='left')

    def fillUIFrameRightTemplateColorize(self, frame):
        self.createFrame(frame, self.fillUIFrameRightTemplate,         {}, {'side': 'top', 'anchor': 'w', 'fill': X})
        self.createFrame(frame, self.fillUIFrameRightAGColorizeTop,    {}, {'side': 'top', 'anchor': 'w'})
        self.createFrame(frame, self.fillUIFrameRightAGColorizeMiddle, {}, {'side': 'top', 'anchor': 'w'})
        self.createFrame(frame, self.fillUIFrameRightAGColorizeBottom, {}, {'side': 'top', 'anchor': 'w'})
    def fillUIFrameRightAGColorizeTop(self, frame):
        self.varRColEmpty = tk.IntVar()
        self.varRColEmpty.set(0)
        self.varRColInvert = tk.IntVar()
        self.varRColInvert.set(0)
        self.varRColAvoidWhite = tk.IntVar()
        self.varRColAvoidWhite.set(1)
        self.varRColCurvesList = tk.IntVar()
        self.varRColCurvesList.set(0)
        tk.Button(frame, text='Colorize', command=self.colorizeGraph).pack(side='left')
        tk.Checkbutton(frame, text='repeat if no label  ', variable=self.varRColEmpty).pack(side='left')
        tk.Checkbutton(frame, text='invert', variable=self.varRColInvert).pack(side='left')
        tk.Checkbutton(frame, text='avoid white', variable=self.varRColAvoidWhite).pack(side='left')
        tk.Checkbutton(frame, text='curve selection', variable=self.varRColCurvesList).pack(side='left')
    def fillUIFrameRightAGColorizeMiddle(self, frame):
        self.varRColChoice = tk.StringVar()
        self.varRColChoice.set('')
        tk.Entry(frame, text=self.varRColChoice, width=50).pack(side='left')
    def fillUIFrameRightAGColorizeBottom(self, frame):
        nbPerLine = 12
        self.colList = Colorscale.GUIdefaults(**self.newGraphKwargs)
        RColColorscale = [None] * len(self.colList)
        self.RColPhoto = [None] * len(self.colList)  # self. to keep a reference to the images after the method ends its execution
        width, height = 30, 15
        for i in range(len(self.colList)):
            self.RColPhoto[i] = PhotoImageColorscale(width=width, height=height)
            try:
                self.RColPhoto[i].fillColorscale(self.colList[i])
            except ValueError:
                # if python does not recognize values (e.g. inferno, viridis)
                self.RColPhoto[i] = None  # back to uninitialized state
            if self.RColPhoto[i] is not None:
                RColColorscale[i] = tk.Button(frame, image=self.RColPhoto[i], command=lambda i_=i: self.colorizeGraphSetScale(i_))
                RColColorscale[i].grid(column=int(i % nbPerLine), row=int(np.floor(i/nbPerLine)))

    def fillUIFrameRightTemplate(self, frame):
        self.varRTplCurves = tk.IntVar()
        self.varRTplCurves.set(1)
        tk.Button(frame, text='Load & apply template', command=self.loadTemplate).pack(side='left',  anchor='n', pady='5')
        tk.Checkbutton(frame, text='also Curves properties', variable=self.varRTplCurves).pack(side='left')
        tk.Button(frame, text='Save template',         command=self.saveTemplate).pack(side='right', anchor='n', pady='5')

    def fillUIFrameRightEditTitle(self, frame):
        tk.Label(frame, text='Edit property', font=self.fontBold).pack(side='top', anchor='w')
    def fillUIFrameRightEdit(self, frame):
        self.createFrame(frame, self.fillUIFrameRightEditH, {}, {'side': 'top', 'anchor': 'w', 'fill': X})
        self.EditPropExample = tk.StringVar()
        self.EditPropExample.set('')
        tk.Label(frame, text='\n').pack(side='left', anchor='n', padx=0)
        tk.Label(frame, textvariable=self.EditPropExample, justify='left').pack(side='left', anchor='n')
    def fillUIFrameRightEditH(self, frame):
        self.varEditPropProp = tk.StringVar()
        self.varEditPropProp.set('Property')
        self.varEditPropVal = tk.StringVar()
        self.varEditPropVal.set('')
        tk.Label(frame, textvariable=self.varEditPropProp).pack(side='left')
        entry = tk.Entry(frame, text=self.varEditPropVal, width=35)
        entry.pack(side='left')
        tk.Button(frame, text='Save', command=self.editProperty).pack(side='right')
        entry.bind('<Return>', lambda event: self.editProperty())

    def fillUIFrameRightNewTitle(self, frame):
        tk.Label(frame, text='Property editor', font=self.fontBold).pack(side='top', anchor='w')
    def fillUIFrameRightNew(self, frame):
        self.createFrame(frame, self.fillUIFrameRightNewH, {}, {'side': 'top', 'anchor': 'w', 'fill': X})
        self.NewPropExample = tk.StringVar()
        self.NewPropExample.set('')
        tk.Label(frame, text='\n').pack(side='left', anchor='n', padx=0)
        tk.Label(frame, textvariable=self.NewPropExample, justify='left').pack(side='left', anchor='n')
    def fillUIFrameRightNewH(self, frame):
        self.varNewPropProp = tk.StringVar()
        self.varNewPropProp.set('')
        self.varNewPropProp.trace('w', self.newPropertySelect)
        self.varNewPropVal = tk.StringVar()
        self.varNewPropVal.set('')
        self.varNewPropValPrevious = self.varNewPropVal.get()
        tk.Label(frame, text='Property:').pack(side='left')
        self.OptionmenuNewProp = tk.OptionMenu(frame, self.varNewPropProp, '')
        self.OptionmenuNewProp.pack(side='left')
        # self.EntryNewProp = tk.Entry (frame, text=self.varNewPropVal, width=30)
        self.EntryNewProp = ttk.Combobox(frame, textvariable=self.varNewPropVal,
                                         values=[], width=30)
        self.EntryNewProp.pack(side='left')
        self.EntryNewProp.bind('<Return>', lambda event: self.newPropertySet())
        tk.Button(frame, text='Save', command=self.newPropertySet).pack(side='right')

    def fillUIFrameRightActionTitle(self, frame):
        self.createFrame(frame, self.fillUIFrameRightActionTitle2, {}, {'side': 'top', 'anchor': 'w', 'fill': X})

    def fillUIFrameRightActionTitle2(self, frame):
        tk.Label(frame, text='Actions specific to selected Curve', font=self.fontBold).pack(side='left', anchor='center')
        hline = FrameTitleContentHide.frameHline(frame)
        hline.pack(side='left', anchor='center', fill=X, expand=1, padx=5)

    def fillUIFrameRightAction(self, frame, curve):
        for action in self.FrameRActionList:
            for field in action:
                if isinstance(field, list):
                    for var in field:
                        del var
                else:
                    field.destroy()
        self.FrameRActionList = []
        if curve != -1:
            # print('funcListGUI', self.graph().curve(curve).funcListGUI())
            tmp = []
            funcList = self.graph()[curve].funcListGUI(graph=self.graph(), graph_i=curve)
            for j in range(len(funcList)):
                act = funcList[j]
                # clean input
                if len(act) == 3:
                    act.append([''] * len(act[2]))
                try:
                    len(act[3])
                except TypeError:  # as e:
                    act[3] = [''] * len(act[2])
                for i in range(len(act[3])):  # default values
                    if isinstance(act[3][i], (list, np.ndarray)):
                        act[3][i] = self.listToString(act[3][i])
                    # act[3][i] = '"'+str(act[3][i])+'"'
                tmp.append([])
                tmp[-1].append([])  # index 0 is list
                tmp[-1][0].append(act[0])  # Validation button
                for i in range(len(act[2])):  # for field in act[2]:
                    tmp[-1][0].append(tk.StringVar())
                    tmp[-1][0][-1].set(act[3][i])
                if len(act) >= 5:         # hidden variables
                    tmp[-1][0].append(act[4])
                tmp[-1].append(tk.Frame(frame))
                tmp[-1][-1].pack(side='top', anchor='w', fill=X)
                if funcList[j][0] is None:
                    tmp[-1].append(tk.Label(tmp[-1][1], text=act[1]))
                else:
                    tmp[-1].append(tk.Button(tmp[-1][1], text=act[1], command=lambda j_=j: self.curveAction(j_)))
                    # previously: self.curveAjction(curve, funcList[j][0], tmp[j][0], j))
                tmp[-1][-1].pack(side='left', anchor='w')
                for i in range(len(act[2])):  # list of Entries
                    fieldprop = copy.deepcopy(act[5][i]) if len(act) > 5 else {}
                    fieldtype = tk.Entry
                    if 'field' in fieldprop:
                        if fieldprop['field'] == 'Combobox':
                            fieldtype = ttk.Combobox
                            if 'width' not in fieldprop:
                                fieldprop.update({'width': int(1.1*max([len(t) for t in fieldprop['values']]))})
                        else:
                            fieldtype = getattr(tk, fieldprop['field'])
                        del fieldprop['field']
                    bind = None
                    if 'bind' in fieldprop:
                        bind = fieldprop['bind']
                        del fieldprop['bind']
                    tmp[-1].append(tk.Label(tmp[-1][1], text=act[2][i]))
                    tmp[-1][-1].pack(side='left', anchor='w')
                    if fieldtype == tk.Entry and 'width' not in fieldprop:
                        width = int(max(8, (40 - len(act[2][i])/3 - len(act[1])/2)/len(act[2])))
                        if len(tmp[-1][0][i+1].get()) < 2:
                            width = int(0.3 * width + 0.7)
                        fieldprop.update({'width': width})
                    if fieldtype == tk.Checkbutton:
                        fieldprop.update({'variable': tmp[-1][0][i+1]})
                    else:
                        fieldprop.update({'textvariable': tmp[-1][0][i+1]})
                    field = fieldtype(tmp[-1][1], **fieldprop)
                    if bind is not None:
                        if bind == 'beforespace':
                            bind = lambda event: event.widget.set(event.widget.get().split(' ')[0])
                        field.bind('<<ComboboxSelected>>', bind)
                    tmp[-1].append(field)  # textvariable=tmp[-1][0][i+1]
                    tmp[-1][-1].pack(side='left', anchor='w')
                    tmp[-1][-1].bind('<Return>', lambda event, j_=j: self.curveAction(j_))
            self.FrameRActionList = tmp
        if self.FrameRActionList == []:
            self.FrameRActionList.append([tk.Label(frame, text='No possible action.')])
            self.FrameRActionList[0][-1].pack(side='top', anchor='w')

    def plotChangeView_identify(self):
        # retrieve current alter
        alter = self.graph().attr('alter')
        if alter == '':
            alter = self.alterListGUI[0][1]
        # find index in list of allowed alterations
        for i in range(len(self.alterListGUI)):
            if alter == self.alterListGUI[i][1]:
                return [i] + self.alterListGUI[i]
        typePlot = self.graph().attr('typeplot')
        return [np.nan, 'File-defined', alter, typePlot]

    def plotChangeView2(self, new):
        [idx, display, alter, typePlot] = self.plotChangeView_identify()
        if display == new:
            return  # no change
        for a in self.alterListGUI:
            if new == a[0]:
                self.OMAlter.set(new)
                self.OMTypePlot.set(a[2])
                self.graph().update({'alter': a[1], 'typeplot': a[2]})
                new = None
                break
        if new is None:
            self.updateUI()

    def plotTypePlot(self, new):
        self.OMTypePlot.set(new)
        self.graph().update({'typeplot': new})
        self.updateUI()


    def setScreenDpi(self):
        new = stringToVariable(self.fieldScreenDPI.get())
        if new == '':
            new = self.DEFAULT_SCREENDPI
        if not is_number(new):
            return False
        figSize = np.array(self.Graph_fig.get_size_inches())
        minMaxPx = np.array([[10, 10], [1000, 1000]])
        newMinMax = np.array([max(minMaxPx[0]/figSize),
                              min(2*minMaxPx[1]/figSize)])
        new = min(max(new, newMinMax[0]), newMinMax[1])
        new = roundSignificant(new, 2)
        self.fieldScreenDPI.set(new)
        print('Set screen DPI to '+str(new)+'.')
        self.Graph_fig.set_dpi(new)
        if self.initiated:
            self.updateUI()


    def curveAction(self, j):
        curve = self.getActiveCurve(multiple=False)
        args = self.FrameRActionList[j][0]
        # retrieve function to call
        func = args[0]
        # hidden arguments
        hidden = {}
        if isinstance(args[-1], dict):
            hidden = args[-1]
            del args[-1]
        # user-adjustable parameters
        args = args[1:]
        args = [stringToVariable(a.get()) for a in args]
        # check if func is method of Graph object and not of the Curve
        if not hasattr(func, '__self__'):
            args = [self.graph()] + args

        def executeFunc(curve, func, args, kwargs):
            # execute curve action
            res = func(*args, **hidden)
            if self.ifPrintCommands():
                try:
                    subject = func.__module__
                    if 'graph' in subject:
                        subject = 'graph'
                    elif 'curve' in subject:
                        subject = 'graph.curve('+str(curve)+')'
                    else:
                        print('WARNING curveAction print commands: subject not',
                              'determined (', subject, func.__name__, j, ')')
                    p = [("'" + a + "'" if isinstance(a, str) else str(a)) for a in args]
                    p +=[(a + "='" + hidden[a] + "'" if isinstance(hidden[a], str) else a + "=" + str(hidden[a])) for a in hidden]
                    print('res = ' + subject + '.' + func.__name__ + '(' + ', '.join(p) + ')')
                except Exception:
                    pass  # error while doing useless output does not really matter
            # where to place new Curves
            idx = curve + 1
            while idx < len(self.graph()):
                type_ = self.graph()[idx].attr('type')
                if not type_.startswith('scatter') and not type_.startswith('errorbar'):
                    break
                idx += 1
            if isinstance(res, Curve):
                self.graph().append(res, idx=idx)
                if self.ifPrintCommands():
                    print("graph.append(res" + ('' if idx == len(self.graph()) else ', idx=' + str(idx)) + ")")
            elif isinstance(res, list) and np.array([isinstance(c, Curve) for c in res]).all():
                self.graph().append(res, idx=idx)
                if self.ifPrintCommands():
                    print("graph.append(res" + ('' if idx == len(self.graph()) else ', idx=' + str(idx)) + ")")
            elif res:
                if res == True:
                    pass
                else:
                    # print ('Curve action output:')
                    print(res)
            else:
                print('Curve action output:')
                print(res)

        # handling actions on multiple Curves
        toExecute = {curve: func}
        curves = self.getActiveCurve(multiple=True)
        if len(curves) > 1:
            funcListRef = self.graph()[curve].funcListGUI(graph=self.graph(), graph_i=curve)
            for c in curves:
                if c != curve:
                    funcListOth = self.graph()[c].funcListGUI(graph=self.graph(), graph_i=c)
                    if len(funcListOth) > j:
                        if (funcListOth[j][0].__name__ == funcListRef[j][0].__name__
                                and funcListOth[j][1] == funcListRef[j][1]
                                and funcListOth[j][2] == funcListRef[j][2]):
                            toExecute.update({c: funcListOth[j][0]})
        keys = list(toExecute.keys())
        keys.sort(reverse=True)
        for c in keys:
            if len(keys) > 1:
                lbl = self.graph()[c].attr('label', '')
                print('Action on Curve', c,
                      (('(' + lbl + ')') if len(lbl) > 0 else ''))
            executeFunc(c, toExecute[c], args, hidden)
        # after execution
        self.updateUI()

    def castCurve(self):
        curves = self.getActiveCurve()
        newType = self.varCastCurve.get()
        for curve in curves:
            if curve > -1 and curve < len(self.graph()):
                test = self.executeGraphMethod('castCurve', newType, curve)
                # test = self.graph().castCurve(newType, curve)
                if not test:
                    print('castCurve impossible.')
            else:
                print('castCurve impossible (', newType, curve, ')')
        self.updateUI()

    def curveShift(self, upDown, relative=True):
        curves_ = self.getActiveCurve()
        curves = list(curves_)
        curves.sort(reverse=(True if (upDown > 0) else False))
        selected = []
        for curve in curves:
            idx2 = upDown
            if curve == 0:
                idx2 = max(idx2, 0)
            if curve == len(self.graph())-1 and relative:
                idx2 = min(idx2, 0)
            if relative:
                self.executeGraphMethod('swapCurves', curve, idx2, relative=True)
                selected.append(curve+idx2)
            else:
                self.executeGraphMethod('moveCurveToIndex', curve, idx2)
                selected.append(idx2)
                # print('moveCurve', curve, idx2, upDown)
                if idx2 < curve or (idx2 == curve and curve == 0):
                    upDown += 1
                elif idx2 > curve or (idx2 == curve and curve >= len(self.graph())-1):
                    upDown -= 1
            # self.graph().swapCurves(curve, upDown, relative=True)
        for i in range(len(selected)):
            selected[i] = max(0, min(len(self.graph())-1, selected[i]))
        self.previousSelectedCurve = selected
        self.updateUI()

    def curveShiftDown(self):
        self.curveShift(1)

    def curveShiftUp(self):
        self.curveShift(-1)

    def curveShiftTop(self):
        self.curveShift(0, relative=False)

    def curveShiftBott(self):
        self.curveShift(len(self.graph())-1, relative=False)

    def curveShowHideCurve(self):
        curves = self.getActiveCurve()
        for curve in curves:
            if curve > -1 and curve < len(self.graph()):
                self.executeGraphCurveMethod(curve, 'swapShowHide')
                # self.graph().curve(curve).swapShowHide()
        self.updateUI()

    def curveShowHideAll(self):
        graph = self.graph()
        if len(graph) > 0:
            new = '' if graph[0].isHidden() else 'none'
            for curve in graph:
                curve.update({'linestyle': new})
            if self.ifPrintCommands():
                print("for curve in graph:")
                print("    curve.update({'linestyle': '"+new+"'})")
            self.updateUI()

    def curveShowHideInvert(self):
        for curve in self.graph():
            curve.swapShowHide()
        if self.ifPrintCommands():
            print("for curve in graph:")
            print("    curve.swapShowHide()")
        self.updateUI()

    def colorizeGraph(self):
        col = stringToVariable(self.varRColChoice.get())
        if len(col) == 0:
            col = self.colList[0].getColorScale()
            self.colorizeGraphSetScale(0)
        invert = self.varRColInvert.get()
        kwargs = {'avoidWhite': self.varRColAvoidWhite.get(),
                  'sameIfEmptyLabel':  self.varRColEmpty.get()}
        if self.varRColCurvesList.get():
            curves = self.getActiveCurve()
            if len(curves) > 0 and curves[0] >= 0:
                # if no curve is selected, colorize all curves
                kwargs.update({'curvesselection': curves})
        try:
            self.graph().colorize(Colorscale(col, invert=invert), **kwargs)
        except ValueError as e:
            # error to be printed in GUI console, and not hidden in terminal
            print('ValueError:', e)
        if self.ifPrintCommands():
            print('graph.colorize(Colorscale(' + str(col) + ', invert='
                  + str(invert) + '), '
                  + ', '.join(['{}={!r}'.format(k, v) for k, v in kwargs.items()])
                  + ')')
        self.updateUI()

    def colorizeGraphSetScale(self, i):
        # reads variable (Colorscale object), extract colors (np.array),
        # converts into string
        if isinstance(self.colList[i].getColorScale(), str):
            scale = self.colList[i].getColorScale()
        else:
            scale = '[' + ', '.join([('['+ ','.join(str(nb if not nb.is_integer() else int(nb)) for nb in elem)+']') if not isinstance(elem, str) else ("'"+elem+"'") for elem in self.colList[i].getColorScale()]) + ']'
        self.varRColChoice.set(scale)

    def dataPickerGetPoint(self):
        # default is data in datapicker textbox
        x = self.varDataPickerX.get()
        y = self.varDataPickerY.get()
        attrUpd = {}
        if self.varDataPickerRestrict.get():
            # if datapicker was restricted to existing data point
            curve = self.varDataPickerCurve.get()
            if curve >= 0:
                graph = self.graph()
                idx = self.varDataPickerIdx
                alter = graph.attr('alter')
                # if user want data transform instead if raw data
                if self.varDataPickerSaveTransform.get():
                    # raw data is displayed in checkbox, need transform
                    if isinstance(alter, str):
                        alter = ['', alter]
                    x = graph[curve].x_offsets(index=idx, alter=alter[0])[0]
                    y = graph[curve].y_offsets(index=idx, alter=alter[1])[0]
                # if user want curve-specific data picker
                if self.varDataPickerSaveCurveSpec.get():
                    # default will be transformed & offset modified data
                    # maybe the Curve object overrides the default method?
                    # case for CurveCf at least
                    x, y, attrUpd = graph[curve].getDataCustomPickerXY(idx, alter=alter)
        return x, y, attrUpd

    def dataPickerSavePoint(self):
        attr = {'linespec': 'x', 'color': 'k', '_dataPicker': True}
        x, y, attrUpd = self.dataPickerGetPoint()
        attr.update(attrUpd)
        # implement in curve results
        graph = self.graph()
        c = graph.curves('_dataPicker', True)
        if len(c) == 0:
            # must create datapicker Curve
            curve = Curve(np.array([[x], [y]]), attr)
            if curve.attr('Curve', None) is not None:
                casted = curve.castCurve(curve.attr('Curve'))
                if casted is not False:
                    curve = casted
            graph.append(curve)
        else:
            # datapicker Curve exists already
            c[0].appendPoints([x], [y])
        self.updateUI()

    def dataPickerToTextbox(self):
        x, y, attrUpd = self.dataPickerGetPoint()
        text = 'x: ' + str(roundSignificant(x, 5)) + '\ny: ' + str(roundSignificant(y, 5))
        textxy = ''
        textargs = {'textcoords': 'data', 'xytext': [x, y], 'fontsize': 8}
        self.executeGraphMethod('addText', text, textxy, textargs=textargs)
        if not self.ifPrintCommands():
            print('New text annotation:', text.replace('\n', '\\n'))
        self.updateUI()


    # manager for text annotations
    def popupAnnotations(self):
        # opens manager
        from gui.GUIpopup import GuiManagerAnnotations
        self.windowAnnotate = tk.Toplevel(self.master)
        self.test = GuiManagerAnnotations(self.windowAnnotate, self.graph(),
                                          self.popupAnnotationsCatch)

    def popupAnnotationsCatch(self, dictupdate):
        self.graph().update(dictupdate)
        self.updateUI()

    # manager for data edition
    def popupDataEditor(self):
        # opens manager
        from gui.GUIdataEditor import GuiDataEditor
        self.windowDataEditor = tk.Toplevel(self.master)
        self.test = GuiDataEditor(self.windowDataEditor, self.graph(),
                                  self.popupDataEditorCatch)

    def popupDataEditorCatch(self):
        # modification of the Curve are performed within popup. Little to do
        self.updateUI()

    def updateUI(self):
        # self.updateUI_graph()
        try:
            self.updateUI_graph()
        except Exception as e:
            print('Exception during update of the GUI Graph panel.')
            print('Exception', type(e), e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
        # self.updateUI_plot()
        try:
            self.updateUI_plot()
        except Exception as e:
            print('Exception during update of the GUI plot.')
            print('Exception', type(e), e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno),
                  type(e).__name__, e)

    def updateUI_plot(self):
        try:
            fig, ax = Graph.plot(self.graph(), figAx=[self.Graph_fig, None])
            while isinstance(ax, (list, np.ndarray)) and len(ax) > 0:
                ax = ax[-1]
            self.Graph_fig, self.Graph_ax = fig, ax
        except ValueError as e:
            print('Exception ValueError during GUI plot update.')
            print('Exception', e)
            pass
        self.callback_updateCrosshair(draw=False)
        try:
            self.canvas.show()
        except AttributeError:
            # handles FigureCanvasTkAgg has no attribute show in later
            # versions of matplotlib
            pass
        self.canvas.draw()


    def updateUI_graph_addTreeBranch(self, idx, attr, tree):
        keyList = []
        for key in attr:
            keyList.append(key)
        keyList.sort()
        for key in keyList:
            val = attr[key]
            if isinstance(val, list) or isinstance(val, np.ndarray):
                val = '"' + self.listToString(val).replace('"', '\\"') + '"'
            elif isinstance(val, str):
                val = '"' + val.replace('\\n', '\\\\n').replace('\n', '\\\\n').replace('"', '\\"') + '"'
            else:
                val = '"' + str(val) + '"'
            val = val.replace('\n', '\\\\n').replace('\\', '\\\\')
            try:
                tree.insert(idx, 'end', text=key, values=val, tag=key)  # idxLast =
            except Exception as e:
                print('Exception AddTreeBranch key', key, ', val', val, type(e), e)

    def updateUI_graph(self):
        graph = self.graph()
        self.alterListGUI = graph.alterListGUI()
        # clear displayed content
        self.Tree.delete(*self.Tree.get_children())
        self.ListBoxCurves.delete(0, tk.END)
        # --> TO DELETE
        self.optionmenuDataPickerCurve['menu'].delete(0, 'end')
        # tree: update graphinfo
        attr = graph.graphInfo
        idx0 = self.Tree.insert('', 'end', text='Graph', tag='-1', values=(''))
        self.updateUI_graph_addTreeBranch(idx0, attr, self.Tree)
        # tree: update headers & sampleinfo
        attr = copy.deepcopy(graph.headers)
        attr.update(graph.sampleInfo)
        if len(attr) > 0:
            idx = self.Tree.insert('', 'end', text='Misc.', tag='-1', values=(''))
            self.updateUI_graph_addTreeBranch(idx, attr, self.Tree)
        # tree & list of curves
        toSelect = [idx0]
        toSelectDefault = True

        for i in range(len(graph)):
            # listBox
            orderLbl = ['label', 'sample', 'filename']
            lbl = '(no label)'
            for test in orderLbl:
                tmp = graph[i].attr(test)
                if tmp != '':
                    lbl = str(tmp)
                    break
            self.ListBoxCurves.insert(i, str(i) + ' ' + lbl)
            # DATAPICKER  --> TO DELETE
            self.optionmenuDataPickerCurve['menu'].add_command(label=str(i)+' '+lbl.replace("'", '\''), command=tk._setit(self.varDataPickerCurve, i))
            # tree
            attr = graph[i].getAttributes()
            idx = self.Tree.insert('', 'end', text='Curve '+str(i)+' '+lbl.replace('\n', '\\n'), tag=str(i), values=(''))
            self.Tree.tag_configure(str(i), foreground=('grey' if graph[i].isHidden() else 'black'))
            self.updateUI_graph_addTreeBranch(idx, attr, self.Tree)
            if i in self.previousSelectedCurve:
                if toSelectDefault:
                    toSelect = [copy.deepcopy(idx)]
                    toSelectDefault = False
                else:
                    toSelect.append(copy.deepcopy(idx))
        self.Tree.focus(toSelect[0])
        self.Tree.selection_set(tuple(toSelect))

        # check data picker Curve selection --> TO DELETE
        if self.varDataPickerCurve.get() >= len(graph):
            self.varDataPickerCurve.set(len(graph)-1)
        # quick modifs --> TO DELETE
        xlim = graph.attr('xlim', ['', ''])
        ylim = graph.attr('ylim', ['', ''])
        xlim = [(x if not isinstance(x, str) and not np.isnan(x) else '') for x in xlim]
        ylim = [(y if not isinstance(y, str) and not np.isnan(y) else '') for y in ylim]
        self.varQuickModsXlabel.set(str(graph.attr('xlabel')).replace('\n', '\\n'))
        self.varQuickModsYlabel.set(str(graph.attr('ylabel')).replace('\n', '\\n'))
        self.varQuickModsXlim0.set(str(xlim[0]))
        self.varQuickModsXlim1.set(str(xlim[1]))
        self.varQuickModsYlim0.set(str(ylim[0]))
        self.varQuickModsYlim1.set(str(ylim[1]))
        # OptionMenu Curve transform (alter) --> TO DELETE
        [idx, display, alter, typePlot] = self.plotChangeView_identify()
        self.OMAlter.resetValues([i[0] for i in self.alterListGUI],
                                 default=display, func=self.plotChangeView2)

        # checks
        sel = [e for e in self.previousSelectedCurve if (e >= 0 and e < len(graph))]
        if len(sel) == 0:
            sel = [-1]
        self.previousSelectedCurve = sel


    def updateUponResizeWindow(self, *args):
        # check if GUI is created for the first time, to avoid updates when
        # windows is not ready
        if hasattr(self, 'initiated') and self.initiated:
            # self.canvas.resize(*args) if event fired from canvas configure event
            self.updateUI_plot()

    def callback_updateCrosshair(self, draw=True):
        # normally draw the canvas, except if called from updateUI which
        # handles draw() by itself
        if hasattr(self, 'crosshairx'):
            try:  # first delete existing crosshair
                self.crosshairx.remove()
                del self.crosshairx
            except ValueError:
                pass
        if hasattr(self, 'crosshairy'):
            try:  # first delete existing crosshair
                self.crosshairy.remove()
                del self.crosshairy
            except ValueError:
                pass
        if self.varDataPickerCrosshair.get():
            self.enableCanvasCallbacks()
            xdata = self.varDataPickerX.get()
            ydata = self.varDataPickerY.get()
            idx = self.varDataPickerIdx
            curve = self.varDataPickerCurve.get()
            alter = self.graph().attr('alter')
            if curve >= 0 and curve >= len(self.graph()):
                curve = len(self.graph()) - 1
                self.varDataPickerCurve.set(curve)
            if isinstance(alter, str):
                alter = ['', alter]
            restrict = self.varDataPickerRestrict.get()
            posx, posy = xdata, ydata
            if restrict and curve >= 0 and not np.isnan(idx):
                posx = self.graph()[curve].x_offsets(index=idx, alter=alter[0])
                posy = self.graph()[curve].y_offsets(index=idx, alter=alter[1])
                # print('crosshair', xdata, ydata, posx, posy)
            self.crosshairx = self.Graph_ax.axvline(posx, 0, 1, color=[0.5, 0.5, 0.5])
            self.crosshairy = self.Graph_ax.axhline(posy, 0, 1, color=[0.5, 0.5, 0.5])
        else:
            self.disableCanvasCallbacks_()
        if draw:
            self.canvas.draw()

    def disableCanvasCallbacks(self):
        self.frameCentral.frameGraph.disableCanvasCallbacks()

    def enableCanvasCallbacks(self):
        self.frameCentral.frameGraph.enableCanvasCallbacks()

    def get_ax(self):
        return self.frameCentral.frameGraph.ax()

    def get_canvas(self):
        return self.frameCentral.frameGraph.canvas()



    def treeActiveCurve(self, curItem):
        # handle several selected items
        if isinstance(curItem, tuple):
            out = [self.treeActiveCurve(item) for item in curItem]
            out = tuple(set(out))  # remove duplicates
            out = tuple(sorted(list(out)))
            return out
        # handle single element
        itemP = self.Tree.parent(curItem)
        itemList = self.Tree.item(curItem)
        tags = itemList['tags']
        if itemP != '':
            it = self.Tree.item(itemP)
            tags = it['tags']
        if isinstance(tags, list):
            tags = tags[0]
        return tags

    def treeSelectItem(self, a):
        curItem = self.Tree.focus()
        curve = self.treeActiveCurve(curItem)
        self.previousSelectedCurve = [curve]
        if curve == -1:
            keyList = Graph.graphInfoKeys
            valList = Graph.graphInfoKeysExample
        else:
            keyList = Graph.dataInfoKeysGraph
            valList = Graph.dataInfoKeysGraphExample
        # update GUI Edit property
        itemList = self.Tree.item(curItem)
        if self.Tree.parent(curItem) == '':
            self.varEditPropProp.set('Property')
            self.varEditPropVal.set('')
            self.EditPropExample.set('')
        else:
            prop = itemList['text']
            value = itemList['values'][0]
            self.varEditPropProp.set(prop)
            self.varEditPropVal.set(value)
            try:
                i = keyList.index(prop)
                self.EditPropExample.set(valList[i])
            except ValueError:
                self.EditPropExample.set('')
        # update GUI New property
        # do not want legend in the list
        new_choices = tuple(Graph.graphInfoKeys) if curve == -1 else tuple(Graph.dataInfoKeysGraph)
        new_choices = tuple(x for x in new_choices if x not in ['legend', ''])
        self.OptionmenuNewProp['menu'].delete(0, 'end')
        for choice in new_choices:
            self.OptionmenuNewProp['menu'].add_command(label=choice, command=tk._setit(self.varNewPropProp, choice))
        if len(new_choices) > 0:
            # change selection if specific keyword was selected in tree
            if self.Tree.parent(curItem) != '':
                self.varNewPropProp.set(itemList['text'])
                self.newPropertySelect()
            else:  # former case: selecting keyword in tree does not change new Property
                if self.varNewPropProp.get() not in new_choices:
                    self.varNewPropProp.set(new_choices[0])
                self.newPropertySelect()
        # do not want to erase the value field
        # update GUI curve actions
        self.fillUIFrameRightAction(self.FrameRAction, curve)
        # update observable related to selected curve
        graph = self.graph()
        self.observableCurve.update_observers(graph[curve] if curve > -1 and curve < len(graph) else curve)
        # update GUI cast action
        self.MenuCastCurve['menu'].delete(0, 'end')
        self.varCastCurve.set('')
        if curve < 0:
            self.castCurveList = []
        else:
            self.castCurveList = graph[curve].castCurveListGUI(onlyDifferent=False)
            for cast in self.castCurveList:
                self.MenuCastCurve['menu'].add_command(label=cast[0], command=tk._setit(self.varCastCurve, cast[0]))
            self.varCastCurve.set(graph[curve].classNameGUI())


    def newPropertySelect(self, *args):
        curve = self.getActiveCurve(multiple=False)
        prop = self.varNewPropProp.get()
        if curve == -1:
            keyList = Graph.graphInfoKeys
            valList = Graph.graphInfoKeysExample
            exaList = Graph.graphInfoKeysExalist
        else:
            keyList = Graph.dataInfoKeysGraph
            valList = Graph.dataInfoKeysGraphExample
            exaList = Graph.dataInfoKeysGraphExalist
        # replace content of Entry/Combobox field if a) is empty, or
        # b) match the previous automatically set value
        if self.varNewPropVal.get() == '' or (self.varNewPropVal.get() == self.varNewPropValPrevious):
            existingVal = self.graph().attr(prop) if curve == -1 else self.graph()[curve].attr(prop)
            self.varNewPropVal.set(str(existingVal))
            self.varNewPropValPrevious = self.varNewPropVal.get()
        try:
            # set example, and populate Combobox values field
            i = keyList.index(prop)
            self.NewPropExample.set(str(valList[i]))
            self.EntryNewProp['values'] = exaList[i]
        except ValueError:
            self.NewPropExample.set('')
            self.EntryNewProp['values'] = []
        # Entry field grey if unvalid choice
        if prop.startswith('=='):
            self.EntryNewProp.configure(state='disabled')
        else:
            self.EntryNewProp.configure(state='normal')


    def getActiveCurve(self, multiple=True):
        if not multiple:
            out = self.treeActiveCurve(self.Tree.focus())
            self.previousSelectedCurve = [out]
        else:
            out = self.treeActiveCurve(self.Tree.selection())
            self.previousSelectedCurve = out
        return out


    def updateProperty(self, curve, key, val, ifUpdate=True, varType='auto'):
        """
        perform curve(curve).update({key: stringToVariable(val)})
        curve -1: update() on self.graph()
        ifUpdate: by default update the GUI- if False, does not. Assumes the
            update will be performed only once, later
        varType: changes type of val into varType. Otherwise try best guess.
        """
        if key == 'Property' or key.startswith('--'):
            return
        # print ('Property edition: curve',curve,', key', key,', value', val, '(',type(val),')')
        # possibly force variable type
        cases = [str, int, float, list, dict]
        if varType in cases:
            val = varType(val)
        else:
            val = stringToVariable(val)
        # handling of newlines
        if isinstance(val, list):
            for i in range(len(val)):
                if isinstance(val[i], str):
                    val[i] = val[i].replace('\\n', '\n')
        elif isinstance(val, str):
            val = val.replace('\\n', '\n')
        # curve identification
        if not isinstance(curve, tuple):
            curve = tuple([curve])
        for c_ in curve:
            try:
                c = int(c_)
            except Exception:
                print('Cannot edit property: curve', c_, ', key', key,
                      ', value', val)
                return
            if c < 0:
                self.graph().update({key: val})
                if self.ifPrintCommands():
                    valstr = str(val) if not isinstance(val, str) else "'"+val+"'"
                    print('graph.update({\''+key+'\': '+valstr+'})')
            else:
                self.graph()[c].update({key: val})
                if self.ifPrintCommands():
                    valstr = str(val) if not isinstance(val, str) else "'"+val+"'"
                    valstr = valstr.replace('\n', '\\n')
                    print('graph.curve('+str(c)+').update({\''+key+'\': '+valstr+'})')
        if ifUpdate:
            self.updateUI()

    def executeGraphMethod(self, method, *args, **kwargs):
        out = getattr(self.graph(), method)(*args, **kwargs)

        def toStr(a):
            if isinstance(a, str):
                return "'" + a.replace('\n', '\\n') + "'"
            elif isinstance(a, Graph):
                return 'graph'
            elif isinstance(a, Curve):
                return 'curve'
            return str(a)

        if self.ifPrintCommands():
            p = [toStr(a) for a in args]
            p += [(a + "='" + toStr(kwargs[a])) for a in kwargs]
            print('graph.' + method + '(' + ', '.join(p) + ')')
        return out
    def executeGraphCurveMethod(self, curve, method, *args, **kwargs):
        out = getattr(self.graph()[curve], method)(*args, **kwargs)
        if self.ifPrintCommands():
            p = [("'"+a.replace('\n', '\\n')+"'" if isinstance(a, str) else str(a)) for a in args]
            p +=[(a+"='"+kwargs[a].replace('\n', '\\n')+"'" if isinstance(kwargs[a], str) else a+"="+str(kwargs[a])) for a in kwargs]
            print('graph.curve('+str(curve)+').'+method+'('+', '.join(p)+')')
        return out

    def quickMods(self):
        """ update the quick modifs, located below the graph """
        xlim = [stringToVariable(self.varQuickModsXlim0.get()),
                stringToVariable(self.varQuickModsXlim1.get())]
        ylim = [stringToVariable(self.varQuickModsYlim0.get()),
                stringToVariable(self.varQuickModsYlim1.get())]
        self.updateProperty(-1, 'xlabel', self.varQuickModsXlabel.get(), ifUpdate=False)
        self.updateProperty(-1, 'ylabel', self.varQuickModsYlabel.get(), ifUpdate=False)
        self.updateProperty(-1, 'xlim',   xlim, ifUpdate=False)
        self.updateProperty(-1, 'ylim',   ylim, ifUpdate=False)
        self.updateUI()


    def editProperty(self):
        """
        Edit current curve property: catch values, send to dedicated function.
        """
        curves = self.getActiveCurve()
        key = self.varEditPropProp.get()
        val = self.varEditPropVal.get()
        self.updateProperty(curves, key, val)

    def newPropertySet(self):
        """
        New property on current curve: catch values, send to dedicated function.
        """
        curves = self.getActiveCurve()
        key = self.varNewPropProp.get()
        val = self.varNewPropVal.get()
        if key == "['key', value]":
            val = stringToVariable(val)
            if not isinstance(val, (list, tuple)) or len(val) < 2:
                print('ERROR GUI.newPropertySet. Input must be a list or a',
                      'tuple with 2 elements (', val, type(val), ')')
                return
            key, val = val[0], val[1]
        self.updateProperty(curves, key, val)
        self.varNewPropVal.set('')
        self.varNewPropProp.set(key)

    def curveDelete(self):
        """ Delete the currently selected curve. """
        curves = list(self.getActiveCurve())
        curves.sort(reverse=True)
        for curve in curves:
            if not is_number(curve):
                break
                # can happen if someone presses the delete button twice in arow
            elif curve > -1:
                self.executeGraphMethod('deleteCurve', curve)
                # self.graph().deleteCurve(curve)
        self.previousSelectedCurve = [self.previousSelectedCurve[0]]
        self.updateUI()

    def curveDeleteAllHidden(self):
        """ Delete all the hidden curves. """
        toDel = []
        graph = self.graph()
        for c in range(len(graph)):
            if graph[c].isHidden():
                toDel.append(c)
        toDel.sort(reverse=True)
        for c in toDel:
            self.executeGraphMethod('deleteCurve', c)
            # self.graph().deleteCurve(c)
        self.updateUI()

    def curveDuplicate(self):
        """ Duplicate the currently selected curve. """
        curves = list(self.getActiveCurve())
        curves.sort(reverse=True)
        selected = []
        for curve in curves:
            if not is_number(curve):
                # can happen if someone presses the delete button twice in arow
                break
            if curve > -1:
                self.executeGraphMethod('duplicateCurve', curve)
                # self.graph().duplicateCurve(curve)
                selected = [s+1 for s in selected]
                selected.append(curve)
        self.previousSelectedCurve = selected
        self.updateUI()

    def graphReplaceLabels(self):
        old = self.varReplaceLabelsOld.get()
        new = self.varReplaceLabelsNew.get()
        # self.graph().replaceLabels(old, new)
        self.executeGraphMethod('replaceLabels', old, new)
        self.varReplaceLabelsOld.set('')
        self.varReplaceLabelsNew.set('')
        self.updateUI()


    def sendToClipboard(self, data):
        self.master.clipboard_clear()
        self.master.clipboard_append(data)

    def curveDataToClipboard(self):
        graph = self.graph()
        curves = self.getActiveCurve()
        content = ''
        opts = self.OptMenuToClipboardAttr.get()
        ifAttrs = 'prop' in opts
        ifTrans = 'screen' in opts
        if not ifAttrs:
            labels = [graph[c].attr('label').replace('\n', '\\n') for c in curves]
            content += ('\t' + '\t\t\t'.join(labels) + '\n')
        else:
            keys = []
            for c in curves:
                for key in graph[c].getAttributes():
                    if key not in keys:
                        keys.append(key)
            keys.sort()
            for key in keys:
                labels = [str(graph[c].attr(key)).replace('\n', '\\n') for c in curves]
                content += (key+'\t' + '\t\t\t'.join(labels) + '\n')
        data = [graph.getCurveData(c, ifAltered=ifTrans) for c in curves]
        length = max([d.shape[1] for d in data])
        for le in range(length):
            tmp = []
            for d in data:
                if le < d.shape[1]:
                    tmp.append('\t'.join([str(e) for e in d[:, le]]))
                else:
                    tmp.append('\t'.join([''] * d.shape[0]))
            content += '\t\t'.join(tmp) + '\n'
        self.sendToClipboard(content)

    def graphDataToClipboard(self):
        opts = self.OptMenuToClipboardAttr.get()
        ifAttrs = 'prop' in opts
        ifTrans = 'screen' in opts
        data = self.graph().export(ifClipboardExport=True, ifOnlyLabels=(not ifAttrs), saveAltered=ifTrans)
        self.sendToClipboard(data)

    def setAutoScreenDPI(self):
        wh = [self.canvas.get_tk_widget().winfo_width(),
              self.canvas.get_tk_widget().winfo_height()]
        figsize = self.graph().attr('figsize', Graph.FIGSIZE_DEFAULT)
        dpimax = min([wh[i] / figsize[i] for i in range(2)])
        dpi = float(self.fieldScreenDPI.get())
        new = None
        if dpi > dpimax * 1.02:  # shall reduce screen dpi
            new = np.max([10, np.round(2*(dpimax-3), -1)/2])
        elif dpi < dpimax * 0.8 and dpi < 100:  # maybe can zoom in
            new = np.min([100, np.round(2*(dpimax-3), -1)/2])
        if new is not None and new != dpi:
            self.fieldScreenDPI.set(new)
            self.setScreenDpi()
            self.blinkWidget(self.fieldScreenDPI, 5)

    def setLimitsSubplotsToCurrent(self):
        xlim = self.Graph_ax.get_xlim()
        ylim = self.Graph_ax.get_ylim()
        a = ['left', 'bottom', 'right', 'top', 'wspace', 'hspace']
        subplots = [getattr(self.Graph_fig.subplotpars, key) for key in a]
        self.executeGraphMethod('update', {'xlim': xlim, 'ylim': ylim,
                                           'subplots_adjust': subplots})
        self.updateUI()

    def graph(self, newgraph=None):
        """
        Get current Graph
        newgraph: to change current Graph. Does NOT refresh interface
        """
        if newgraph is not None:
            if isinstance(newgraph, Graph):
                self.back_graph = newgraph
            else:
                print('WARNING: GUI.graph(), newgraph must be a Graph.')
        return self.back_graph

    def currentFolder(self, newvalue=None):
        """
        Get current active folder
        - newvalue: set newvalue for the current active folder
        """
        if newvalue is not None:
            self.panelConsole.folder(newvalue)
        folder = self.panelConsole.folder()
        if folder == '':
            file = self.panelConsole.file()
            folder = os.path.dirname(os.path.abspath(file))
        if folder == '':
            folder = os.getcwd()
        return folder

    def currentFile(self, newvalue=None, newvaluefolder=True):
        """
        Get current active file
        - newvalue: set newvalue for the current active file
        - newvaluefolder: bool, set new value for the current folder based on
          newvalue
        """
        if newvalue is not None:
            self.panelConsole.file(newvalue)
            if newvaluefolder:
                folder = os.path.dirname(os.path.abspath(newvalue))
                self.currentFolder(folder)
        return self.panelConsole.file()

    def quit(self):
        plt.close(self.Graph_fig)
        self.master.quit()
        self.master.destroy()

    def ifPrintCommands(self):
        return self.menuLeft.varPrintCommands.get()

    def saveTemplate(self):
        """ save template file from current Graph """
        folder = self.currentFolder()
        f = tk.filedialog.asksaveasfilename(defaultextension='.txt', initialdir=folder)
        if f is not None and f != '':
            f, fileext = os.path.splitext(f)
            fileext = fileext.replace('py', '')
            self.graph().export(filesave=f, ifTemplate=True)

    def loadTemplate(self):
        """ load and apply template on current Graph """
        folder = self.currentFolder()
        file = tk.filedialog.askopenfilename(initialdir=folder)
        if file != '' and file is not None:
            print('Open template file:', file)
            template = Graph(file, complement={'label': ''})
            alsoCurves = self.varRTplCurves.get()
            self.graph().applyTemplate(template, alsoCurves=alsoCurves)
            self.updateUI()

    def openFile(self, file):
        """
        Open a file
        - file: a str, or a list of str to open multiple files, or a Graph
        """
        lbl = file
        if isinstance(file, Graph):
            # do not print anything
            graph = file
            lbl = ''
            if hasattr(graph, 'fileexport'):
                lbl = graph.fileexport
            elif hasattr(graph, 'filename'):
                lbl = graph.filename
        elif isinstance(file, list):
            print('Open multiple files (first:', lbl[0], ')')
            lbl = file[0]
            graph = Graph(file, **self.newGraphKwargs)
        else:
            print('Open file:', lbl.replace('/', '\\'))
            graph = Graph(file, **self.newGraphKwargs)
        self.graph(newgraph=graph)
        if lbl != '':
            self.currentFile(lbl)
            # folder = os.path.dirname(lbl)
            # self.currentFolder(folder)  # done by currentFile
        self.setAutoScreenDPI()
        self.alterListGUI = self.graph().alterListGUI()
        [idx, display, alter, typePlot] = self.plotChangeView_identify()
        self.OMAlter.set(idx)
        self.callback_dataPickerCurve()  # update help datapicker Curve specifi
        if self.ifPrintCommands():
            print("graph = Graph('"+file+"')")
        if self.initiated:
            self.updateUI()

    def mergeGraph(self, graph):
        """ graph can be a str (filename), or a Graph """
        file = ''
        if not isinstance(graph, Graph):
            file = graph
            print('Merge with file:', file)
            graph = Graph(graph)
        else:
            pass  # stays silent
            # print('Merge with graph', file)
        self.graph().merge(graph)
        # change default save folder only if was empty
        if self.currentFolder() == '' and file != '':
            self.currentFolder(file)
        if self.ifPrintCommands():
            print("graph.merge(Graph('"+file+"'))")
        self.updateUI()

    def saveGraph(self, filesave, fileext='', saveAltered=False, ifCompact=True):
        """
        Saves Graph into a file. Separated plot() and export() calls.
        filename: str, filename to save gr graph
        fileext: str, image data format (eg. .png). If '', retrieve preference.
        saveAltered: as per Graph.export(). Default False.
        ifCompact: as per Graph.export(). Default True.
        """
        graph = self.graph()
        if graph.attr('meastype') == '':
            graph.update({'meastype': FILEIO_GRAPHTYPE_GRAPH})
        try:
            fig, ax = graph.plot(filesave=filesave, imgFormat=fileext,
                                 ifExport=False, figAx=[self.Graph_fig, None])
            while isinstance(ax, (list, np.ndarray)) and len(ax) > 0:
                ax = ax[0]
            self.Graph_fig, self.Graph_ax = fig, ax
        except Exception as e:
            print('ERROR: Exception during plotting of the Graph.')
            print(type(e), e)
        self.currentFile(filesave)
        if fileext in ['.xml']:
            filesave += fileext
        graph.export(filesave=filesave, saveAltered=saveAltered,
                     ifCompact=ifCompact)
        if self.ifPrintCommands():
            print("graph.plot(filesave='" + filesave + "', imgFormat='"
                  + fileext + "', ifExport=False))")
            print("graph.export(filesave='" + filesave + "', saveAltered='"
                  + str(saveAltered) + "', ifCompact='" + str(ifCompact) + "')")
        self.updateUI()

    def appendCurveToGraph(self, curve, updateUI=True):
        """ Appends 1 Crve, or a list of Curves to the active graph """
        if isinstance(curve, list):
            for c in curve:
                self.graphAppendCurve(c, updateUI=False)
        elif isinstance(curve, Curve):
            self.graph().append(curve)
        else:
            print('ERROR GUI graphAppendCurve, could not handle class of',
                  '"curve"', type(curve), '.')
        if updateUI:
            self.updateUI()

    def blinkWidget(self, field, niter, delay=500, property_='background', values=['white', 'red']):
        field.config(**{property_: values[niter % len(values)]})
        if niter > 0:
            kwargs = {'property_': property_, 'values': values, 'delay': delay}
            func = lambda: self.blinkWidget(field, niter-1, **kwargs)
            self.master.after(delay, func)

    def listToString(self, val):
        return '[' + ', '.join([str(element) if not isinstance(element, str) else '\''+element+'\'' for element in val]) + ']'



def printLastRelease():
    file = 'versionNotes.txt'
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
    try:
        f = open(file, 'r')
    except Exception:
        # print('Exception file open', e)
        return
    out = ''
    line = ''
    flag = True
    while flag:
        last = line
        line = f.readline()
        if "Release" in line:
            out = last + line
            flag = False
    if line == '':
        print('Not found string "Release"')
        return  # not found release
    date = line[len('Release'):].replace(' ', '')
    if date[0] == 'd':
        date = date[1:]
    from dateutil import parser
    from datetime import datetime
    try:
        daysSinceLast = datetime.now() - parser.parse(date)
    except ValueError as e:
        print('Exception in PrintLastRelease, date', date, type(e), e)
    line = ''
    if daysSinceLast.days < 5:
        flag = True
        while flag:
            last = line
            line = f.readline()
            if "Release" in line or 'Version' in line:
                flag = False
            if flag and last.strip() != '':
                out += last
        print(out)
    f.close()
    return


def buildUI():
    root = tk.Tk()
    try:
        root.iconbitmap('datareading.ico')
    except Exception:
        pass
    app = Application(master=root)
    from grapa import __version__
    app.master.title('Grapa software v'+__version__)
    # starts runnning programm
    with stdout_redirect(app.panelConsole.console):
        # retrieve content of last release
        try:
            printLastRelease()
        except Exception:
            pass
        # start execution
        app.mainloop()


if __name__ == "__main__":
    buildUI()
