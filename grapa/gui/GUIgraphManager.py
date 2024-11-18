# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:30:42 2021

@author: car
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path not in sys.path:
    sys.path.append(path)
from grapa.graph import Graph
from grapa.gui.observable import Observable


class CustomNotebook(ttk.Notebook):
    """
    A ttk Notebook with close buttons on each tab
    https://stackoverflow.com/questions/39458337/is-there-a-way-to-add-close-buttons-to-tabs-in-tkinter-ttk-notebook
    """

    __initialized = False

    def __init__(self, *args, **kwargs):
        if not self.__initialized:
            self.__initialize_custom_style()
            CustomNotebook.__initialized = True
        kwargs["style"] = "CustomNotebook"
        ttk.Notebook.__init__(self, *args, **kwargs)
        self._active = None
        self.bind("<ButtonPress-1>", self.on_close_press, True)
        self.bind("<ButtonRelease-1>", self.on_close_release)

    def on_close_press(self, event):
        """Called when the button is pressed over the close button"""
        element = self.identify(event.x, event.y)
        if "close" in element:
            index = self.index("@%d,%d" % (event.x, event.y))
            self.state(["pressed"])
            self._active = index

    def on_close_release(self, event):
        """Called when the button is released over the close button"""
        if not self.instate(["pressed"]):
            return
        element = self.identify(event.x, event.y)
        try:
            index = self.index("@%d,%d" % (event.x, event.y))
        except Exception:  # mouse was released outside of the Notebook area
            index = None
        if "close" in element and self._active == index:
            self.forget(index)
            self.event_generate("<<NotebookTabClosed>>")
        self.state(["!pressed"])
        self._active = None

    def __initialize_custom_style(self):
        style = ttk.Style()
        self.images = (
            tk.PhotoImage(
                "img_close",
                data="""
                R0lGODlhCAAIAMIBAAAAADs7O4+Pj9nZ2Ts7Ozs7Ozs7Ozs7OyH+EUNyZWF0ZWQg
                d2l0aCBHSU1QACH5BAEKAAQALAAAAAAIAAgAAAMVGDBEA0qNJyGw7AmxmuaZhWEU
                5kEJADs=
                """,
            ),
            tk.PhotoImage(
                "img_closeactive",
                data="""
                R0lGODlhCAAIAMIEAAAAAP/SAP/bNNnZ2cbGxsbGxsbGxsbGxiH5BAEKAAQALAAA
                AAAIAAgAAAMVGDBEA0qNJyGw7AmxmuaZhWEU5kEJADs=
                """,
            ),
            tk.PhotoImage(
                "img_closepressed",
                data="""
                R0lGODlhCAAIAMIEAAAAAOUqKv9mZtnZ2Ts7Ozs7Ozs7Ozs7OyH+EUNyZWF0ZWQg
                d2l0aCBHSU1QACH5BAEKAAQALAAAAAAIAAgAAAMVGDBEA0qNJyGw7AmxmuaZhWEU
                5kEJADs=
            """,
            ),
        )
        style.element_create(
            "close",
            "image",
            "img_close",
            ("active", "pressed", "!disabled", "img_closepressed"),
            ("active", "!disabled", "img_closeactive"),
            border=8,
            sticky="",
        )
        style.layout("CustomNotebook", [("CustomNotebook.client", {"sticky": "nswe"})])
        style.layout(
            "CustomNotebook.Tab",
            [
                (
                    "CustomNotebook.tab",
                    {
                        "sticky": "nswe",
                        "children": [
                            (
                                "CustomNotebook.padding",
                                {
                                    "side": "top",
                                    "sticky": "nswe",
                                    "children": [
                                        (
                                            "CustomNotebook.focus",
                                            {
                                                "side": "top",
                                                "sticky": "nswe",
                                                "children": [
                                                    (
                                                        "CustomNotebook.label",
                                                        {"side": "left", "sticky": ""},
                                                    ),
                                                    (
                                                        "CustomNotebook.close",
                                                        {"side": "left", "sticky": ""},
                                                    ),
                                                ],
                                            },
                                        )
                                    ],
                                },
                            )
                        ],
                    },
                )
            ],
        )


class GraphHandler:
    """
    A container to store a graph as well as "satellite" data.
    Contains:
    _title: title of the tab
    _child: eg ref to tk.Frame
    _propdict: a dict, including graph, filename, folder, etc.
    Input:
    strOrGraph: a str, or a Graph object. Graph has precedence over filename.
        In such case, filename is interpreted as title
    title: the title, as a "satellite" information
    """

    DEFAULT_PROPDICT = {}

    def __init__(self, title, child, **kwargs):
        self.reset(title, child, **kwargs)

    def reset(self, title, child, **kwargs):
        self._title = str(title)
        self._child = child  # stores info for container, eg ref to tk.Frame
        self._propdict = {}  # other properties
        self._propdict.update(GraphHandler.DEFAULT_PROPDICT)
        self.update(**kwargs)

    def update(self, **kwargs):
        for key in kwargs:
            if kwargs[key] is not None:
                # do not want to overwrite something by accident
                self._propdict.update({key: kwargs[key]})

    def title(self, title=None):
        if title is not None:
            self._title = str(title)
            if len(self._title) > 80:
                self._title = self._title[:30] + " ... " + self._title[-30:]
        return self._title

    def child(self, child=None):
        if child is not None:
            self._child = child
        return self._child

    def properties(self):
        return self._propdict

    def property(self, key):
        return self._propdict[key]


class GraphsTabManager:
    """
    Combines a ttk.Notebook with a list of items, 1 item per tab
    Ensures that the content of Notebook and list of items are synchronized
    ._item and ._notebook should not be modified separately
    """

    def __init__(self, *args, defdict=None, **kwargs):
        """
        Create the GraphsTabHandler the same way you would create a
        ttk.Notebook
        defdict: default values to be provided to ._itemtype.DEFAULT_PROPDICT
        """
        self.observableTabChanged = Observable()
        # hidden properties
        self._notebook = CustomNotebook(*args, **kwargs)
        self._items = []
        self._itemtype = GraphHandler
        # configure default parameters
        defaultdict = {"filename": "", "folder": "", "graph": None, "dpi": 72}
        if isinstance(defdict, dict):
            defaultdict.update(defdict)
        self._itemtype.DEFAULT_PROPDICT = defaultdict
        # bind events
        self._notebook.bind("<<NotebookTabClosed>>", self._popEvent)
        self._notebook.bind("<<NotebookTabChanged>>", self._reactTabChanged)

    # to forward tkinter placement methods
    def pack(self, *args, **kwargs):
        self._notebook.pack(*args, **kwargs)

    def grid(self, *args, **kwargs):
        self._notebook.grid(*args, **kwargs)

    # handling of Observable
    def register(self, func):
        """Register callbacks for event <<NotebookTabChanged>>"""
        self.observableTabChanged.register(func)

    def _reactTabChanged(self, event):
        self._checkConsistency()
        self.observableTabChanged.update_observers(event)

    # To know current selected tab
    def index(self):
        return self._notebook.index(self._notebook.select())

    def item(self):
        return self._items[self.index()]

    # creation or removal of tabs
    def appendNew(
        self, strOrGraph, title=None, child=None, filename=None, folder=None, **kwargs
    ):
        """
        strOrGraph: can be a str or a Graph from which an item can be
            instanciated
        if child is None, empty Frame will be automatically created
        if title is None, its value will be guessed
        """
        if child is None:
            child = tk.Frame(self._notebook)
        title, kwargs = self._process(
            strOrGraph=strOrGraph,
            title=title,
            filename=filename,
            folder=folder,
            **kwargs
        )
        # print('appendNew title, kwargs', title, kwargs)
        if title is None:
            title = "no name"
        item = self._itemtype(title, child)
        item.update(**kwargs)
        if "folder" not in item.properties():
            item.update(folder=os.getcwd())
        # add new items to the list and to notebook
        self._items.append(item)
        # print('   ', item.title())
        self._notebook.add(item.child(), text=item.title())
        self._checkConsistency()

    # insert: not yet implemented

    # Handling of dynamic behavior and callbacks
    def select(self, idx):
        """
        Select a given tab in the Notebook. idx between 0 and len-1
        """
        # idx = max(0, min(idx, len(self._notebook.tabs())-1))  # no need
        if idx < 0:
            idx += len(self._notebook.tabs())
        return self._notebook.select(idx)
        # no need for callbackTabChanged,event NotebookTabChanged would follow

    def pop(self, idx=None):
        """
        Removes a element from both tab and item list. If None, the selected
        tab is deleted.
        """
        if idx is None:
            idx = self.index()
        item = self._items.pop(idx)
        self._notebook.forget(idx)
        self._checkConsistency()
        self._checkAtLeast1Element()
        return item
        # no need for callbackTabChanged,event NotebookTabChanged would follow

    # Technicalities
    def _popEvent(self, event):
        """
        Triggered by mouse-induced tab closure.
        Notebook.forget already done.
        """
        self._items.pop(self._notebook._active)
        self._checkConsistency()
        self._checkAtLeast1Element()
        # no need for callbackTabChanged,event NotebookTabChanged would follow

    def _process(
        self, strOrGraph=None, title=None, filename=None, folder=None, **kwargs
    ):
        """Return title, kw{'graph', 'filename', 'folder'}"""
        if "graph" in kwargs and not isinstance(kwargs["graph"], Graph):
            print("GraphsTabManager must provide a Graph for keyword graph")
            del kwargs["graph"]
        kw = {}
        # graph
        if isinstance(strOrGraph, Graph):
            kw["graph"] = strOrGraph
        elif strOrGraph is not None:
            kw["graph"] = Graph(strOrGraph)
            kw["filename"] = strOrGraph
        elif "graph" in kwargs:
            kw["graph"] = kwargs["graph"]
        # else graph will not be in kw
        # filename
        if filename is not None:
            kw["filename"] = filename
        elif "filename" not in kw:
            if "graph" in kw:
                if hasattr(kw["graph"], "fileexport"):
                    kw["filename"] = strOrGraph.fileexport
                elif hasattr(kw["graph"], "filename"):
                    kw["filename"] = strOrGraph.filename
            # else filename not in kw
        # else filename is already in kw
        # folder
        if folder is not None:
            kw["folder"] = folder
        else:
            if "filename" in kw:
                kw["folder"] = str(os.path.dirname(kw["filename"]))
                # print('folder from filename', kw['folder'])
        # kwargs
        kw.update(kwargs)  # add any other keywords provided
        # title
        if title is not None:
            pass  # title = title
        else:
            tit = ""
            if "filename" in kw:
                tit = os.path.basename(kw["filename"])
                split = os.path.splitext(tit)
                if len(split[0]) > 0:
                    tit = split[0]
            if tit == "" and "graph" in kw:
                tit = kw["graph"].attr("title")
                if isinstance(tit, list):
                    tit = tit[0]  # clear formatting instructions
                if tit == "":
                    tit = kw["graph"].attr("legendtitle")
                if isinstance(tit, list):
                    tit = tit[0]  # clear formatting instructions
            if tit != "":
                title = tit
        # title may be returned None - for updates. to check when reset and new
        return title, kw

    def updateCurrent(self, title=None, filename=None, folder=None, dpi=None, **kwargs):
        """
        Update selected tab and corresponding information
        kwargs: etc.
        """
        idx = self.index()
        title, kw = self._process(
            title=title, filename=filename, folder=folder, dpi=dpi, **kwargs
        )
        self._items[idx].title(title)
        self._items[idx].update(**kw)
        self._notebook.tab(idx, text=self._items[idx].title())

    def resetCurrent(
        self, strOrGraph, title=None, filename=None, folder=None, **kwargs
    ):
        """
        Resets information conencted to the selected tab
        - strOrGraph: can be a str (filename) or a Graph
        - title: overrides the filename or automatic title detection
        - kwargs: filename, folder, dpi, etc.
        """
        idx = self.index()
        title, kw = self._process(
            strOrGraph=strOrGraph,
            title=title,
            filename=filename,
            folder=folder,
            **kwargs
        )
        if title is None:
            title = "no name"
        self.item[idx].reset(title, tk.Frame(self._notebook), **kw)
        if "folder" not in self.item[idx].properties():
            self.item[idx].update(folder=os.getcwd())
        self._notebook.tab(idx, text=self._items[idx].title())

    def _checkAtLeast1Element(self):
        # always at least 1 tab
        if len(self._items) == 0:
            self.appendNew(Graph())  # empty element

    def _checkConsistency(self):
        """Checks consistency of tabs and items: identical text/titles"""
        # checks
        tabs = [self._notebook.tab(tab)["text"] for tab in self._notebook.tabs()]
        titles = [item.title() for item in self._items]
        if len(tabs) != len(titles):
            print(
                "ERROR GraphsTabManager consistency not same length, tab",
                "desynchronization",
            )
            print("   tabs", len(tabs), tabs)
            print("   titles", len(titles), titles)
            # TODO: automatic repair synchronization of lists
            return False
        for i in range(len(titles)):
            if titles[i] != tabs[i]:
                print("ERROR GraphsTabManager consistency 2, tab", "desynchronization")
                print(i, tabs[i], titles[i])
                return False
        # print('   checkConsistency ok')
        return True

    # Return properties of active item. Shortcut notation specific to item type
    def getTitle(self):
        """Returns current selected title"""
        return self.item().title()

    def getChild(self):
        """Returns current selected child (Frame)"""
        return self.item().child()

    def getGraph(self):
        """Returns current selected Graph"""
        return self.item().property("graph")

    def getFilename(self):
        """Returns current selected filename"""
        return self.item().property("filename")

    def getFolder(self):
        """Returns current selected folder"""
        return self.item().property("folder")

    def getProperties(self):
        """Returns current selected propdict dict of properties"""
        return self.item().properties()


def testAll():
    import random

    root = tk.Tk()
    h = GraphsTabManager(root, width=200, height=50)
    h.pack(side="top", fill="x", expand=True)
    for color in ("red", "orange", "green", "blue", "violet"):
        h.appendNew("", title=color)

    def new1():
        colors = ["black", "magenta", "cyan", "yellow"]
        random.shuffle(colors)
        h.appendNew(
            "", title=colors[0], child=tk.Frame(h._notebook, background=colors[0])
        )

    def new2():
        obj = [
            "./../examples/fancyAnnotations.txt",
            Graph("./../examples/subplots_examples.txt"),
        ]
        random.shuffle(obj)
        h.appendNew(obj[0])

    def new3():
        h.appendNew(Graph("./../examples/fancyAnnotations.txt"), title="my title")

    def delCurrent():
        h.pop(h.index())

    def delFirst():
        h.pop(0)

    def changeCurrentText():
        h.updateCurrent(title=h.getTitle() + " *")

    def selectNext():
        h.select(h.index() + 1)

    def print_():
        print(h.getProperties()["graph"])

    tk.Button(root, text="New colored", command=new1).pack(side="top", anchor="w")
    tk.Button(root, text="New 2", command=new2).pack(side="top", anchor="w")
    tk.Button(root, text="New with title", command=new3).pack(side="top", anchor="w")
    # tk.Button(root, text='Get active?', command=getActive).pack(side="top", anchor='w')
    tk.Button(root, text="Change text current", command=changeCurrentText).pack(
        side="top", anchor="w"
    )
    tk.Button(root, text="Del current", command=delCurrent).pack(side="top", anchor="w")
    tk.Button(root, text="Del first", command=delFirst).pack(side="top", anchor="w")
    tk.Button(root, text="Select next", command=selectNext).pack(side="top", anchor="w")
    tk.Button(root, text="print", command=print_).pack(side="top", anchor="w")
    root.mainloop()


if __name__ == "__main__":
    testAll()
