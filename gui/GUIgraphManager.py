# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:30:42 2021

@author: car
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if path not in sys.path:
    sys.path.append(path)
from grapa.graph import Graph
from grapa.observable import Observable


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
            self.state(['pressed'])
            self._active = index

    def on_close_release(self, event):
        """Called when the button is released over the close button"""
        if not self.instate(['pressed']):
            return
        element = self.identify(event.x, event.y)
        index = self.index("@%d,%d" % (event.x, event.y))
        if "close" in element and self._active == index:
            self.forget(index)
            self.event_generate("<<NotebookTabClosed>>")
        self.state(["!pressed"])
        self._active = None

    def __initialize_custom_style(self):
        style = ttk.Style()
        self.images = (
            tk.PhotoImage("img_close", data='''
                R0lGODlhCAAIAMIBAAAAADs7O4+Pj9nZ2Ts7Ozs7Ozs7Ozs7OyH+EUNyZWF0ZWQg
                d2l0aCBHSU1QACH5BAEKAAQALAAAAAAIAAgAAAMVGDBEA0qNJyGw7AmxmuaZhWEU
                5kEJADs=
                '''),
            tk.PhotoImage("img_closeactive", data='''
                R0lGODlhCAAIAMIEAAAAAP/SAP/bNNnZ2cbGxsbGxsbGxsbGxiH5BAEKAAQALAAA
                AAAIAAgAAAMVGDBEA0qNJyGw7AmxmuaZhWEU5kEJADs=
                '''),
            tk.PhotoImage("img_closepressed", data='''
                R0lGODlhCAAIAMIEAAAAAOUqKv9mZtnZ2Ts7Ozs7Ozs7Ozs7OyH+EUNyZWF0ZWQg
                d2l0aCBHSU1QACH5BAEKAAQALAAAAAAIAAgAAAMVGDBEA0qNJyGw7AmxmuaZhWEU
                5kEJADs=
            ''')
        )
        style.element_create("close", "image", "img_close",
                            ("active", "pressed", "!disabled", "img_closepressed"),
                            ("active", "!disabled", "img_closeactive"),
                            border=8, sticky='')
        style.layout("CustomNotebook", [("CustomNotebook.client", {"sticky": "nswe"})])
        style.layout("CustomNotebook.Tab", [
            ("CustomNotebook.tab", {
                "sticky": "nswe",
                "children": [
                    ("CustomNotebook.padding", {
                        "side": "top",
                        "sticky": "nswe",
                        "children": [
                            ("CustomNotebook.focus", {
                                "side": "top",
                                "sticky": "nswe",
                                "children": [
                                    ("CustomNotebook.label", {"side": "left", "sticky": ''}),
                                    ("CustomNotebook.close", {"side": "left", "sticky": ''}),
                                ]
                            })
                        ]
                    })
                ]
            })
        ])


class GraphHandler:
    """
    A container to store a graph as well as "satellite" data.
    Contains:
    _graph: a Graph object
    _filename: str
    _folder: str
    _title:
    _child: tobe used by a container
    Input:
    strOrGraph: a str, or a Graph object. Graph has precedence over filename.
        In such case, filename is interpreted as title
    title: the title, as a "satellite" information
    """

    def __init__(self, strOrGraph, title=None):
        """
        graph has precedence over filename. In such case, filename is
        interpreted as title
        """
        self._filename = ''
        self._folder = ''
        self._title = ''
        self._graph = None
        self._child = None  # stores info for container, eg ref to tk.Frame
        self.reset(strOrGraph, title=title)
        pass

    def reset(self, strOrGraph, title=None):
        if isinstance(strOrGraph, Graph):
            self._graph = strOrGraph
            self.update(title=title)
        else:
            self._graph = Graph(strOrGraph)
            self.update(filename=strOrGraph, title=title)
        # in principle do not want empty title
        if self.title() == '':
            test = self._graph.attr('title')
            if test != '':
                self.update(title=test)
        if self.title() == '':
            test = self._graph.attr('legendtitle')
            if test != '':
                self.update(title=test)
        if self.title() == '':
            self.update(title='no name')

    def update(self, filename=None, folder=None, title=None):
        """
        Update "side" properties: filename, folder, title.
        filename also updates the folder and title (changes are overriden by
        specific folder and title inputs)
        No effect if None values are provided.
        """
        if filename is not None:
            self._filename = filename
            self._folder = os.path.dirname(filename)
            self._title = os.path.basename(filename)
            while self._title.endswith('.txt'):
                self._title = self._title[:-4]
        if folder is not None:
            self._folder = folder
        if title is not None:
            self._title = title

    def graph(self):
        return self._graph

    def filename(self):
        return self._filename

    def folder(self):
        return self._folder

    def title(self):
        return self._title

    def child(self, child):
        if child is not None:
            self._child = child
        return self._child


class GraphsTabManager:
    """
    Combines a ttk.Notebook with a list of items, 1 item per tab
    Ensures that the content of Notebook and list of items are synchronized
    ._item and ._notebook should not be modified separately
    """

    def __init__(self, *args, **kwargs):
        """
        Create the GraphsTabHandler the same way you would create a
        ttk.Notebook
        """
        self.observableTabChanged = Observable()
        # hidden properties
        self._notebook = CustomNotebook(*args, **kwargs)
        self._items = []
        self._itemtype = GraphHandler
        self._notebook.bind('<<NotebookTabClosed>>', self._popEvent)
        self._notebook.bind('<<NotebookTabChanged>>', self._reactTabChanged)

    # to forward tkinter placement methods
    def pack(self, *args, **kwargs):
        self._notebook.pack(*args, **kwargs)

    def grid(self, *args, **kwargs):
        self._notebook.grid(*args, **kwargs)

    # To know current selected tab
    def index(self):
        return self._notebook.index(self._notebook.select())

    def item(self):
        return self._items[self.index()]

    # creation or removal of tabs
    def append(self, itemOrStrOrGraph, title=None, child=None):
        """
        itemOrStrOrGraph: can be a item (GraphHandler), or a str or a Graph
            from which an item can be instanciated
            if itemOrStrOrGraph is GraphHandler, title is ignored
        if child None, empty Frame will be automatically created
        """
        item = self._checkItemType(itemOrStrOrGraph, title=title)
        if child is None:
            child = tk.Frame(self._notebook)  # , background=item.title())
        item.child(child)
        self._items.append(item)
        self._notebook.add(child, text=item.title())
        self._checkConsistency()
    # insert: not yet implemented

    def pop(self, idx=None):
        """
        Removes a element from both tab and item list. If None, the selected
        tab is deleted.
        """
        print('pop')
        if idx is None:
            idx = self.index()
        item = self._items.pop(idx)
        self._notebook.forget(idx)
        self._checkConsistency()
        return item
        # no need for callbackTabChanged,event NotebookTabChanged would follow

    # Handling of dynamic behavior and callbacks
    def select(self, idx):
        """
        Select a given tab in the Notebook. idx between 0 and len-1
        """
        idx = max(0, min(idx, len(self._notebook.tabs())-1))
        return self._notebook.select(idx)
        # no need for callbackTabChanged,event NotebookTabChanged would follow

    # Technicalities
    def _popEvent(self, event):
        """ Triggered by mouse-induced tab closure. .forget already done """
        print('popevent')
        self._items.pop(self._notebook._active)
        self._checkConsistency()
        # no need for callbackTabChanged,event NotebookTabChanged would follow

    def _reactTabChanged(self, event):
        print('reactTabChanged')
        self._checkConsistency()
        self.observableTabChanged.update_observers()

    def _checkItemType(self, item, title=None):
        if not isinstance(item, self._itemtype):
            item = self._itemtype(item, title=title)
        return item

    def _checkConsistency(self):
        """ Checks consistency of tabs and items: identical text/titles """
        tabs = [self._notebook.tab(tab)['text'] for tab in self._notebook.tabs()]
        titles = [item.title() for item in self._items]
        if len(tabs) != len(titles):
            print('ERROR GraphsTabManager consistency not same length, tab desynchronization')
            print('   tabs', len(tabs), tabs)
            print('   titles', len(titles), titles)
            # TODO: automatic repair synchronization of lists
            return False
        for i in range(len(titles)):
            if titles[i] != tabs[i]:
                print('ERROR GraphsTabManager consistency 2, tab desynchronization')
                print(i, tabs[i], titles[i])
                return False
        print('   checkConsistency ok')
        return True

    # Return properties of active item. Shortcut notation specific to item type
    def graph(self):
        """ Returns current selected Graph """
        return self.item().graph()

    def title(self):
        """ Returns current selected title """
        return self.item().title()

    def filename(self):
        """ Returns current selected filename """
        return self.item().filename()

    def folder(self):
        """ Returns current selected folder """
        return self.item().folder()

    def child(self):
        """ Returns current selected child (Frame) """
        return self.item().child()

    def update(self, filename=None, folder=None, title=None):
        """ Update selected tab and corresponding information """
        idx = self.index()
        self._items[idx].update(filename=filename, folder=folder, title=title)
        self._notebook.tab(idx, text=self._items[idx].title())

    def reset(self, strOrGraph, title=None):
        """
        Resets information conencted to the selected tab
        - strOrGraph: can be a str (filename) or a Graph
        - title: overrides the filename or automatic title detection
        """
        idx = self.index()
        self._items[idx].reset(strOrGraph, title=title)
        self._notebook.tab(idx, text=self._items[idx].title())


def testAll():
    import random
    root = tk.Tk()
    h = GraphsTabManager(root, width=200, height=50)
    h.pack(side="top", fill="x", expand=True)
    for color in ("red", "orange", "green", "blue", "violet"):
        handler = GraphHandler(Graph(''), title=color)
        h.append(handler)

    def new1():
        colors = ['black', 'magenta', 'cyan', 'yellow']
        random.shuffle(colors)
        h.append('', title=colors[0], child=tk.Frame(h._notebook, background=colors[0]))

    def new2():
        obj = ['./../examples/fancyAnnotations.txt',
               Graph('./../examples/subplots_examples.txt')]
        random.shuffle(obj)
        h.append(obj[0])

    def new3():
        h.append(Graph('./../examples/fancyAnnotations.txt'), title='my title')

    def delCurrent():
        h.pop(h.index())

    def delFirst():
        h.pop(0)

    def changeCurrentText():
        h.update(title=h.title()+' *')

    def selectNext():
        h.select(h.index() + 1)

    def print_():
        print(h.graph())
    tk.Button(root, text='New colored', command=new1).pack(side="top", anchor='w')
    tk.Button(root, text='New 2', command=new2).pack(side="top", anchor='w')
    tk.Button(root, text='New with title', command=new3).pack(side="top", anchor='w')
    # tk.Button(root, text='Get active?', command=getActive).pack(side="top", anchor='w')
    tk.Button(root, text='Change text current', command=changeCurrentText).pack(side="top", anchor='w')
    tk.Button(root, text='Del current', command=delCurrent).pack(side="top", anchor='w')
    tk.Button(root, text='Del first', command=delFirst).pack(side="top", anchor='w')
    tk.Button(root, text='Select next', command=selectNext).pack(side="top", anchor='w')
    tk.Button(root, text='print', command=print_).pack(side="top", anchor='w')
    root.mainloop()


"""
def testCustomNotebook():
    # test CustomNotebook
    root = tk.Tk()
    notebook = CustomNotebook(root, width=200, height=50)
    notebook.pack(side="top", fill="x", expand=True)
    for color in ("red", "orange", "green", "blue", "violet"):
        frame = tk.Frame(notebook, background=color)
        notebook.add(frame, text=color)
    def getActive():
        idx = notebook.index(notebook.select())
        print(idx, notebook.tab(idx)['text'])
        return idx
    def new():
        import random
        colors = ['black', 'magenta', 'cyan', 'yellow']
        random.shuffle(colors)
        frame = tk.Frame(notebook, background=colors[0])
        notebook.add(frame, text=colors[0])
    def changeText():
        idx = getActive()
        new = str(notebook.tab(idx)['text'])+' *'
        print(new)
        notebook.tab(idx, text=new)
    tk.Button(root, text='New', command=new).pack(side="top", anchor='w')
    tk.Button(root, text='Get active?', command=getActive).pack(side="top", anchor='w')
    tk.Button(root, text='change text', command=changeText).pack(side="top", anchor='w')
    root.mainloop()
"""

if __name__ == "__main__":
    testAll()
