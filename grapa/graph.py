# -*- coding: utf-8 -*-
"""Graph class, most important in grapa together with Curve.
A Graph stores a list of Curves, as well as metadata (plotting information, other)

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import os
from copy import deepcopy
from re import findall as refindall
import importlib
import logging

import numpy as np
import matplotlib as mpl

from grapa import KEYWORDS_GRAPH, KEYWORDS_CURVE, KEYWORDS_HEADERS
from grapa.mathModule import is_number
from grapa.colorscale import colorize_graph
from grapa.curve import Curve
from grapa.utils.string_manipulations import strToVar, TextHandler
from grapa.utils.funcgui import AlterListItem
from grapa.utils.metadata import MetadataContainer
from grapa.utils.graphIO import GraphIO, export

logger = logging.getLogger(__name__)


class Graph:
    """A Graph can be though as containing a list of Curves, and is a convenient way to
    read, store and manipulate data, as well as export plot as images and text files.

    :param filename: str filename to open, or list[str] (filenames), or
           list of data series [xseries, yseries [, zseries...]].
           Default '' (graph empty).
    :param complement: dict, the elements will be stored as graph attributes.
           Example: {'xlim': [1, '']}. Special values:

           - 'isfilecontent' = True: content of variable filename is interpreted as data

           - 'readas' in ['database', 'generic']. 'database' can be useful. You may
             consider using pandas instead

    :param silent: to get more details on what is going on. Default True
    :param config: configuration datafile. Default 'config.txt'
    """

    # Class constants
    FIGSIZE_DEFAULT = (6.0, 4.0)
    FILEIO_DPI = 80
    DEFAULT = {"subplots_adjust": 0.15, "fontsize": mpl.rcParams["font.size"]}
    # Default axis labels. Normally not used, but can be overridden and used in
    # subclasses. Possible data formats are:
    # ['x-axis label [unit]', 'y-axis label [unit]']
    # [['Quantity x', 'Q', 'unit'], ['y', 'Q', 'unit']]. Q will be placed in between $ $
    AXISLABELS = ["", ""]

    CONFIG_FILENAME_DEFAULT = "config.txt"
    CONFIG_FILES = {}  # to store config files, to avoid multiple loadings

    # list of child classes of Graph
    _list_subclasses_parse = None

    # constructor
    def __init__(
        self,
        filename="",  # str, list[str], or list[xseries, yseries]
        complement: dict = "",  # OR EBTTER A DICT???
        silent: bool = True,
        config: str = CONFIG_FILENAME_DEFAULT,  # or None
    ):
        """default constructor. Calls method reset."""
        self.filename = None
        self.silent = None
        self.data = []
        self.headers = MetadataContainer()
        self.graphinfo = MetadataContainer()
        self._config = None
        self._init_config(config)
        # actually load the file
        self.reset(filename, complement=complement, silent=silent)

    def reset(self, filename, complement: dict = "", silent: bool = True):
        """Reset a Graph: empty its list of Curves, and metadata."""
        # complement: special keywords: 'readas', 'isfilecontent'
        # default values
        self.filename = filename
        self.silent = silent
        self.data = []
        self.headers.clear()
        self.graphinfo.clear()
        # try to identify and parse the datafile
        if isinstance(filename, str) and filename == "":
            if not self.silent:
                print("Empty Graph object created")
            return

        if isinstance(filename, str):
            # if single file was provided - a string, or if filename is the
            # content if the file - complement['isfilecontent'] must be true
            GraphIO.readDataFile(self, complement=complement)
        elif (
            len(filename) == 2
            and len(filename[0]) == len(filename[1])
            and is_number(filename[0][0])
        ):
            # filename is actually the data, as [xseries, yseries]
            if not self.silent:
                print('Class Graph: interpret "filename" as data content')
            self.filename = ""
            GraphIO.append_curve_from_datainput(self, filename, attributes=complement)
        else:
            # if a list was provided - open first file, then merge the
            # others one by one
            if not isinstance(complement, list):
                if complement != "":
                    msg = (
                        "Graph: complement must be a list if filename is a list."
                        " Filename {} elements, complement: {}."
                    )
                    logger.warning(msg.format(len(filename), complement))
                complement = [complement] * len(filename)
            self.filename = filename[0]
            GraphIO.readDataFile(self, complement[0])
            if len(filename) > 1:
                for i in range(1, len(filename)):
                    if len(complement) > 1:
                        self.merge(Graph(filename[i], complement=complement[i]))
                    else:
                        self.merge(Graph(filename[i]))
        # last: want to have abspath and not relative
        if hasattr(self, "filename"):
            self.filename = os.path.abspath(self.filename)

    def _init_config(self, config: str) -> None:
        """Initialize the configuration file. Tries to recycle if possible"""
        if config is not None:
            if config == self.CONFIG_FILENAME_DEFAULT:
                folder = os.path.dirname(os.path.realpath(__file__))
                config = os.path.join(folder, config)
            if config not in Graph.CONFIG_FILES:
                graph = Graph(config, complement={"readas": "database"}, config=None)
                if len(graph) > 0:
                    Graph.CONFIG_FILES[config] = graph
                else:
                    msg = (
                        "Graph.init_config not found, or not properly"
                        "formatted (2-columns key-values). Cannot use it."
                    )
                    logger.warning(msg)
            if config in Graph.CONFIG_FILES:
                self._config = Graph.CONFIG_FILES[config]

    # RELATED TO GUI
    def alterListGUI(self) -> list:
        """Compiles a list of possible data transforms, retrieved from all Curves"""
        out = []
        for curve in self:
            for item in curve.alterListGUI():
                if item not in out:
                    out.append(item)
        neutral = AlterListItem.item_neutral()
        if neutral not in out:
            out.insert(0, neutral)
        out = [(AlterListItem(*o) if isinstance(o, list) else o) for o in out]
        return out

    # "USUAL" CLASS METHODS
    def __str__(self) -> str:
        """Returns some information about the class instance."""
        out = "Content of Graph stored in file {}\n".format(self.filename)
        out += "Content of headers:\n"
        out += str(self.headers)
        out += "\nNumber of Curves: " + str(len(self))
        return out

    # interactions with other Graph objects
    def merge(self, graph):
        """Add in a Graph object the content of another Graph object"""
        # keep previous self.filename
        # copy data
        for x in graph.data:
            self.data.append(x)
        # copy headers, unless already exists (is so, info is lost)
        for key, value in graph.headers.items():
            if key not in self.headers:
                self.headers.update({key: value})
        # copy graphInfo, unless already exists (is so, info is lost)
        for key, value in graph.graphinfo.items():
            if key not in self.graphinfo:
                self.graphinfo.update({key: value})

    # methods handling the bunch of curves
    def __len__(self):
        """Returns the number of Curves."""
        return len(self.data)

    def __getitem__(self, key) -> Curve:
        """Returns a Curve object at index key."""
        return self.data[key]

    def __iter__(self):
        return self.data.__iter__()

    def __delitem__(self, key):
        """Deletes the Curve at index key."""
        self.curve_delete(key)

    def curve(self, index):
        """Returns the Curve object at index i in the list. Can also use graph[index]"""
        if index >= len(self) or len(self) == 0:
            msg = "Graph.curve: cannot find Curve (index {}, len(self) {})."
            logger.error(msg.format(index, len(self)))  # raise ?
            return None
        return self.data[index]

    def curves(
        self, key: str, value, str_lower: bool = False, str_startswith: bool = False
    ) -> list:
        """Returns a list of Curves for which curve.attr(key) value (type and value
        equality)

        :param key: the attribute to check. By default, 'label'.
        :param value: the value to test
        :param str_lower: if True, performs .lower() onto value and curve.attr(key)
               before comparison, in case each are str.
        :param str_startswith: is both value and the attribute are str, only look at the
               first characters
        :return: a list of Curves
        """
        out = []
        for curve in self:
            val = curve.attr(key)
            if isinstance(val, type(value)):
                if str_lower:
                    if isinstance(val, str):
                        val = val.lower()
                    if isinstance(value, str):
                        value = value.lower()
                if val == value:
                    out.append(curve)
                elif isinstance(val, str) and str_startswith and val.startswith(value):
                    out.append(curve)
        return out

    def append(self, curve, idx=None):
        """
        Add into the object list a Curve, a list or Curves, or every Curve in a
        Graph object.
        """
        insert = False if idx is None else True
        if isinstance(curve, Graph):
            for curv_ in curve:  # "curve" is a Graph, can iterate
                if insert:
                    self.data.insert(idx, curv_)
                    idx += 1
                else:
                    self.data.append(curv_)  # curv_ is a Curve, we can directly append
        elif isinstance(curve, list):
            for curv_ in curve:
                if insert:
                    self.append(curv_, idx)
                    idx += 1
                else:
                    self.append(curv_)  # call itself, to ensure curv_ is a Curve
        elif isinstance(curve, Curve):
            if insert:
                self.data.insert(idx, curve)
            else:
                self.data.append(curve)
        else:
            logger.error("Graph.append: failed (type: {})".format(type(curve)))

    def curve_delete(self, i):
        """Delete a Curve at index i from the Graph object."""
        if isinstance(i, (list, tuple)):
            i = list(np.sort(i)[::-1])
            for k in i:
                self.curve_delete(k)
        else:
            le0 = len(self.data)
            if is_number(i) and i < le0:
                # delete data
                del self.data[i]
                # delete in headers
                if "collabels" in self.headers and i < len(self.headers["collabels"]):
                    if len(self.headers["collabels"]) == 0:
                        del self.headers["collabels"]
                    else:
                        del self.headers["collabels"][i]
                if "collabelsdetail" in self.headers:
                    # certainly only useful for "databases"
                    for j in range(len(self.headers["collabelsdetail"])):
                        if i < len(self.headers["collabelsdetail"][j]):
                            del self.headers["collabelsdetail"][j][i]
                # nothing to delete in graphInfo

    def curve_replace(self, new_curve, idx):
        """Replaces a Curve with another."""
        if isinstance(new_curve, Curve):
            try:
                self.data[idx] = new_curve
                return True
            except Exception:
                msg = "Graph.curve_replace: cannot add Curve at index {}."
                logger.warning(msg.format(idx))  # , exc_info=True
        else:
            msg = "Graph.curve_replace: new_curve is not a Curve (type {})"
            logger.warning(msg.format(type(new_curve)))
        return False

    def curve_move_to_index(self, idxsource, idxtarget):
        """Change the position of a Curve in the list"""
        tmp = self.data.pop(idxsource)
        self.data.insert(idxtarget, tmp)
        return True

    def curve_duplicate(self, idx1):
        """
        Duplicate (clone) an existing curve and append it in the curves list.
        """
        if idx1 < -len(self) or idx1 >= len(self):
            msg = "Graph.curve_duplicate: idx1 not valid ({})."
            logger.error(msg.format(idx1))
            return False
        curve = deepcopy(self[idx1])
        self.data.insert(idx1 + 1, curve)
        return True

    def curves_swap(self, idx1, idx2, relative=False):
        """
        Exchange the Curves at index idx1 and idx2.
        For example useful to modify the plot order (and order in the legend)
        """
        if idx1 < -len(self) or idx1 >= len(self):
            msg = "Graph.curves_swap: idx1 not valid ({})."
            logger.error(msg.format(idx1))
            return False
        if relative:
            idx2 = idx1 + idx2
        if idx2 == idx1:  # swap with itself
            return True
        if idx2 < -len(self) or idx2 >= len(self):
            msg = "Graph.curves_swap: idx2 not valid ({})."
            logger.error(msg.format(idx2))
            return False
        swap = deepcopy(self[idx1])
        self.data[idx1] = self[idx2]
        self.data[idx2] = swap
        return True

    def curves_reverse(self):
        """Reverse the order of the Curves."""
        self.data.reverse()
        return True

    # methods handling content of curves
    def getCurveData(self, idx, ifAltered=True):
        if ifAltered:
            alter = self.get_alter()
            x = self[idx].x_offsets(alter=alter[0])
            y = self[idx].y_offsets(alter=alter[1])
        else:
            x = self[idx].x(alter="")
            y = self[idx].y(alter="")
        return np.array([x, y])

    def attr(self, key, default=MetadataContainer.VALUE_DEFAULT):
        """Retrieve an attribute value in header, graphInfo or in curve[0]"""
        if key in self.headers:
            return self.headers[key]
        if key in self.graphinfo:
            return self.graphinfo[key]
        if len(self) > 0:
            return self[0].attr(key, default=default)
        return default

    def get_attributes(self, keys_list: list = None) -> dict:
        """Returns a dict containing the content of both headers and graphInfo"""
        # TODO: test Graph.get_attributes
        out = {}
        # need to merge several dicts. Can erase meaningless values, we should not erase
        # meaningful ones. The data input methods should prevent risk of duplicated keys
        if keys_list is None:
            for container in [self.headers, self.graphinfo]:
                out.update(container.values())  # only contain meaningful values
        else:
            keys_list = [MetadataContainer.format(key) for key in keys_list]
            for key in keys_list:
                for container in [self.headers, self.graphinfo]:
                    if container.has_attr(key):  # only add if meaningful content
                        out[key] = container.get(key)
                if key not in out:
                    out[key] = MetadataContainer.VALUE_DEFAULT
        return out

    def has_attr(self, key):
        """Check if graph has attribute key"""
        if self.graphinfo.has_attr(key):
            return True
        return self.headers.has_attr(key)

    def update(self, attributes: dict, if_all=False) -> None:
        """Update the properties of the Graph object.
        The properties are stored in dict objects, so update works similarly as
        the dict.update() method.
        If a property is not known to belong to the Graph object, it is
        attributed to the 1st Curve object (unless if_all=True)
        """
        for key, value in attributes.items():
            k = MetadataContainer.format(key)
            if k in KEYWORDS_HEADERS["keys"]:
                self.headers.update({k: value})
            elif k in KEYWORDS_GRAPH["keys"] or k.startswith("subplots"):
                self.graphinfo.update({k: value})
            # put into one or several Curves
            elif if_all:
                for curve in self:
                    curve.update({k: value})
            else:
                if len(self) > 0:
                    self[-1].update({k: value})
                else:
                    msg = "Graph.update: cannot process ({}, {})."
                    logger.error(msg.format(key, value))

    def updateValuesDictkeys(self, *args, **kwargs):
        """Update the graph using a specific data input, for GUI purposes.
        Falls back onto ({key1: value1, key2: value2, ...}).
        Mandatory: len(args) == len(kwargs["keys"]).

        :param args: value1, value2, ...
        :param kwargs: key=['key1', 'key2', ...]
        """
        if "keys" not in kwargs:
            msg = (
                "Graph.updateValuesDictkeys: 'keys' key must be provided, must be a "
                "list of keys corresponding to the values provided in *args."
            )
            logger.error(msg)
            return False
        if len(kwargs["keys"]) != len(args):
            msg = (
                "WARNING Graph updateValuesDictkeys: len of list 'keys' argument "
                "must match the number of provided values ({} args provided, {} keys)"
            )
            logger.error(msg.format(len(args), len(kwargs["keys"])))
        lenmax = min(len(kwargs["keys"]), len(args))
        for i in range(lenmax):
            self.update({kwargs["keys"][i]: args[i]})
        return True

    def attr_pop(self, key: str) -> dict:
        """
        Delete an attribute of the Graph.
        Return the deleted attribute as a dict
        """
        k = MetadataContainer.format(key)
        out = {k: self.attr(k)}
        self.update({k: ""})
        return out

    @classmethod
    def _get_alter_to_format(cls, alter):
        """Return a formatted alter instruction"""
        if alter == "":
            alter = ["", ""]
        if isinstance(alter, str):  # nothing to do if it is dict
            alter = ["", alter]
        return alter

    def get_alter(self):
        """returns the formatted alter instruction of self"""
        return self._get_alter_to_format(self.attr("alter"))

    def castCurve(self, newtype, idx, silentSuccess=False):
        """
        Replace a Curve with another type of Curve with identical data and properties.
        """
        if -len(self) <= idx < len(self):
            newCurve = self[idx].castCurve(newtype)
            if isinstance(newCurve, Curve):
                flag = self.curve_replace(newCurve, idx)
                if flag:
                    if not silentSuccess:
                        msg = "Graph.castCurve: new Curve type: {} {}."
                        print(msg.format(self[idx].classNameGUI(), type(self[idx])))
                else:
                    print("Graph.castCurve")
                return flag
        else:
            msg = "Graph.castCurve: idx not in suitable range ({}, max {})."
            logger.error(msg.format(idx, len(self)))
        return False

    def colorize(
        self, colorscale, sameIfEmptyLabel=False, avoidWhite=False, curvesselection=None
    ):
        """
        Colorize a graph, by coloring each Curve along a colorscale gradient.
        """
        return colorize_graph(
            self,
            colorscale,
            same_if_empty_label=sameIfEmptyLabel,
            avoid_white=avoidWhite,
            curvesselection=curvesselection,
        )

    def apply_template(self, graph, also_curves=True):
        """
        Apply a template to the Graph object. The template is a Graph object which
        contains:

        - self.get_attributes listed in KEYWORDS_GRAPH["keys"]

        - self[i].get_attributes listed in KEYWORDS_CURVE["keys"]

        :param also_curves: True also apply Curves properties, False only apply Graph
               properties
        """
        # strip default attributes
        for key in Graph.DEFAULT:
            if graph.attr(key) == Graph.DEFAULT[key]:
                graph.update({key: ""})
        for key in KEYWORDS_GRAPH["keys"]:
            val = graph.attr(key)
            if not MetadataContainer.is_attr_value_default(val):
                self.update({key: val})
        if also_curves:
            for c in range(len(self)):
                if c >= len(graph):
                    break
                for key in KEYWORDS_CURVE["keys"]:
                    val = graph[c].attr(key)
                    if not MetadataContainer.is_attr_value_default(val):
                        self[c].update({key: val})

    def replace_labels(self, old, new):
        """
        Modify all labels of the Graph, by replacing 'old' by 'new'
        """
        for curve in self:
            curve.update({"label": curve.attr("label").replace(old, new)})

    def text_check_valid(self):
        """Call TextHandler.checkValidText, to make sure the text annotation
        attributes are consistent"""
        return TextHandler.check_valid(self)

    def text_add(self, text, textxy, textargs=None):
        """Adds a text to be annotated in the plot, handling the not-so-nice internal
        implementation.

        :param text: as single element, or as list (1 item per annotation)
        :param textxy: as single element, or as list (1 item per annotation)
        :param textargs: as single element, or as list (1 item per annotation)
        :return: initial values of self.attr("text"), "textxy", "textargs"
        """
        return TextHandler.add(self, text, textxy, textargs=textargs)

    def text_remove(self, by_id=-1):
        """By default, removes the last text annotation in the list (pop)

        :param by_id: index of the annotation to remove
        :returns: a dict with initial text values, prior to removal (use case: restore)
        """
        return TextHandler.remove(self, by_id=by_id)

    def formatAxisLabel(self, label):
        """
        Returns a string for label according to user preference.
        'Some quantity [unit]' may become 'Some quantify (unit)'
        Other possible input is ['This is a Quantity', 'Q', 'unit']
        """
        # retrieve user preference
        symbol = bool(self.config("graph_labels_symbols", False))
        units = self.config("graph_labels_units", default="[]", astype=str)
        units = units.replace("unit", "").replace(" ", "")
        # format input
        if isinstance(label, str):
            if units != "[]":  # that is default, no need to do anything
                expr = r"^(.* )\[(.*)\](.*)$"  # [ ] as characters, not a set
                if label != "":
                    f = refindall(expr, label)
                    if isinstance(f, list) and len(f) == 1 and len(f[0]) == 3:
                        if units in ["DIN", "/"]:
                            return f[0][0].strip(" ") + " / " + f[0][1] + "" + f[0][2]
                        elif units == "()":
                            return f[0][0].strip(" ") + "(" + f[0][1] + ")" + f[0][2]
        elif isinstance(label, list):
            while len(label) < 3:
                label += [""]
            out = label[0]
            if symbol and len(label[1]) > 0:
                out += " ${}$".format(label[1])
            if label[2] not in [None, ""]:
                if units == "/":
                    out += " / {}".format(label[2])
                elif units == "()":
                    out += " ({})".format(label[2])
                else:
                    out += " [{}]".format(label[2])
            return out.replace("  ", " ")
        return label

    # Configuration file
    def config(self, key: str, default=MetadataContainer.VALUE_DEFAULT, astype="auto"):
        """
        Returns the value corresponding to key in the configuration file.
        If 'key' is not defined in config file, returns default.
        """
        if self._config is not None and len(self._config) > 0:
            out = self._config[0].attr(key, default=default)
            if astype in [str, "str"]:
                return str(out)
            else:
                return strToVar(out)
        return default

    def config_all(self) -> (dict, str):
        """Returns all key-value pairs in config file, and filename"""
        if self._config is not None and len(self._config) > 0:
            return {
                "attributes": self._config[0].get_attributes(),
                "filename": self._config.filename,
            }
        return {"attributes": {}, "filename": None}

    def filenamewithpath(self, filename):
        # if filename is relative, join the path of the file with
        if os.path.isabs(filename):
            return filename
        path = ""
        if (
            hasattr(self, "filename")
            and isinstance(self.filename, str)
            and len(self.filename) > 0
        ):
            path = os.path.dirname(os.path.abspath(self.filename))
        return os.path.join(path, filename)

    # For convenience, a shortcut to graphIO export
    def export(
        self,
        filesave="",
        save_altered=False,
        if_template=False,
        if_compact=True,
        if_only_labels=False,
        if_clipboard_export=False,
    ):
        """Exports content of Grah object into a human- and machine-readable format."""
        return export(
            self,
            filesave=filesave,
            save_altered=save_altered,
            if_template=if_template,
            if_compact=if_compact,
            if_only_labels=if_only_labels,
            if_clipboard_export=if_clipboard_export,
        )

    # For convenience, we offer a shortcut to plot
    def plot(
        self,
        filesave="",
        img_format="",
        figsize=(0, 0),
        if_save="auto",
        if_export="auto",
        fig_ax=None,
        if_subplot="auto",
    ):
        """
        Plot the content of the Graph object.

        :param filesave: filename for the saved graph image and export text file.
        :param img_format: by default image format will be .png. Possible formats are
               the ones supported by plt.savefig() and possibly .emf
        :param figsize: figure size in inch. The actual default value is from a class
               constant
        :param if_save: [True/False/'auto'] if True, save the Graph as an image. If
               left to default, saves the image only if filesave is provided
        :param if_export: [True/False/'auto'] if True, create a human- and machine-
               readable .txt file containing all information of the graph.
               By default, export only if filesave is provided.
        :param fig_ax: [fig, ax]. the graph will be plotted in the provided figure
               and ax.
               Providing None or [fig, None] will erase all existing axes and create
               what is needed
        :param if_subplot: [True/False/'auto'] If True, prevents deletion of the
               existing axes in the figure.
               By default, keeps axes if axes are provided in figAx.
        """
        from grapa.utils.plot_graph import plot_graph  # otherwise circular dependency

        if if_save not in [True, False]:
            if_save = False if filesave == "" else True
        if if_export not in [True, False]:
            if_export = False if filesave == "" else True

        if if_subplot not in [True, False]:
            if_subplot = False
            if (
                fig_ax is not None
                and isinstance(fig_ax, list)
                and len(fig_ax) > 1
                and fig_ax[1] is not None
            ):
                if_subplot = True
        return plot_graph(
            self,
            filesave=filesave,
            img_format=img_format,
            figsize=figsize,
            if_save=if_save,
            if_export=if_export,
            fig_ax=fig_ax,
            if_subplot=if_subplot,
        )

    @classmethod
    def get_list_subclasses_parse(cls):
        """
        Returns the child classes of Graph which have implemented the methods required
        to parse files "isFileReadable", "readDataFromFile", and class attribute
        FILEIO_GRAPHTYPE.
        Seeps through files in folder datatypes whose filename start with 'curve' or
        'graph', check the presence of a class with appropriate name and required
        methods, and returns a list of Graph child class.
        """
        # do not reimport everything each time, only first time
        if cls._list_subclasses_parse is not None:
            return cls._list_subclasses_parse

        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datatypes")
        required = ["FILEIO_GRAPHTYPE", "isFileReadable", "readDataFromFile"]
        subclasses = []
        for filestart in ["graph", "curve"]:
            for file in os.listdir(folder):
                fname, fext = os.path.splitext(file)
                if (
                    fext == ".py"
                    and fname.startswith(filestart)
                    and len(fname) > len(filestart)
                ):
                    end = fname[len(filestart) :]
                    module = importlib.import_module("grapa.datatypes." + fname)
                    if not hasattr(module, "Graph" + end):
                        continue

                    class_to_test = getattr(module, "Graph" + end)
                    is_valid = True
                    for attr in required:
                        if not hasattr(class_to_test, attr):
                            is_valid = False
                            break
                    if is_valid:
                        subclasses.append(class_to_test)
        cls._list_subclasses_parse = subclasses  # stored as class variable
        return subclasses

    # DEPRECATED METHODS, mostly renamed
    def length(self):
        """Returns the number of Curve objects in the list. Can also use len(graph)

        :meta private:
        """
        return len(self.data)

    def iterCurves(self):
        """. Deprecated. Returns an iterator over the different Curves.
        Not great, deprecateds. Rather use: for curve in graph:

        :meta private:
        """
        for curve in self:
            yield curve

    def getAttribute(self, key: str, default=MetadataContainer.VALUE_DEFAULT) -> dict:
        """. Deprecated. alias to attr.

        :meta private:"""
        return self.attr(key, default=default)

    def getAttributes(self, keys_list: list = None) -> dict:
        """. Deprecated. Use get_attributes

        :meta private:"""
        return self.get_attributes(keys_list=keys_list)

    def deleteCurve(self, i):
        """. Deprecated. Use curve_delete instead.

        :meta private:
        """
        return self.curve_delete(i)

    def replaceCurve(self, new_curve, idx):
        """. Deprecated. Use curve_replace instead

        :meta private:
        """
        return self.curve_replace(new_curve, idx)

    def moveCurveToIndex(self, idxsource, idxtarget):
        """. Deprecated. Use curve_move_to_index instread.

        :meta private:
        """
        return self.curve_move_to_index(idxsource, idxtarget)

    def duplicateCurve(self, idx1):
        """. Deprecated. Use curve_duplicate instead

        :meta private:
        """
        return self.curve_duplicate(idx1)

    def swapCurves(self, idx1, idx2, relative=False):
        """. Deprecated. Use curves_swap instead

        :meta private:
        """
        return self.curves_swap(idx1, idx2, relative=relative)

    def reverseCurves(self):
        """. Deprecated. Use curves_reverse instead

        :meta private:
        """
        return self.curves_reverse()

    # def delete(self, key: str) -> dict:
    #     """Alias of attr_pop. Historical reasons. Should remove.
    #
    #     :meta private:"""
    #     return self.attr_pop(key)


class ConditionalPropertyApplier:
    """
    Apply changes to attributes of Curves within a Graph, for the Curves that satisfy a
    test
    """

    MODES_BYINPUTTYPE = {
        "any": ["==", "!=", ">", ">=", "<", "<="],
        "str": [
            "startswith",
            "endswith",
            "contains",
            "not startswith",
            "not endswith",
            "not contains",
        ],
    }
    # flatten in a single list
    MODES_VALUES = [x for _, xs in MODES_BYINPUTTYPE.items() for x in xs]

    @staticmethod
    def _evaluate_values(valref, valtest, mode):
        if mode == "==":
            return valref == valtest
        elif mode == "!=":
            return valref != valtest
        elif mode == ">":
            return valref > valtest
        elif mode == ">=":
            return valref >= valtest
        elif mode == "<":
            return valref < valtest
        elif mode == "<=":
            return valref <= valtest
        elif mode == "startswith":
            return str(valref).startswith(str(valtest))
        elif mode == "endswith":
            return str(valref).endswith(str(valtest))
        elif mode == "contains":
            return str(valtest) in str(valref)
        elif mode == "not startswith":
            return not str(valref).startswith(str(valtest))
        elif mode == "not endswith":
            return not str(valref).endswith(str(valtest))
        elif mode == "not contains":
            return str(valtest) not in str(valref)
        msg = (
            "GraphConditionalPropertyApplier._evaluate_values: unsupported mode "
            "{}, return False. Input values {}, {}."
        )
        logger.warning(msg.format(mode, valref, valtest))
        return False

    @classmethod
    def _coerce_mode(cls, mode):
        if mode in cls.MODES_VALUES:
            return mode
        new = "=="
        msg = (
            "GraphConditionalProperty: unsupported mode, changed '{}' for '{}'. "
            "Possible values: {}."
        )
        logger.warning(msg.format(mode, new, cls.MODES_VALUES))
        return new

    @classmethod
    def apply(
        cls, graphorcurve, test_prop, test_mode, test_value, apply_prop, apply_value
    ):
        """Changes values of property to a given value, for all curves satisfying
        test condition."""
        mode = cls._coerce_mode(test_mode)
        # if Graph: loop and execute over the curves
        if isinstance(graphorcurve, Graph):
            for curve in graphorcurve:
                cls.apply(curve, test_prop, mode, test_value, apply_prop, apply_value)
            return
        # supposedly a Curve object: execute behavior
        val = graphorcurve.attr(test_prop)
        try:
            test = cls._evaluate_values(val, test_value, mode)
        except (AttributeError, TypeError) as e:
            msg = (
                "GraphConditionalProperty.apply: Error during evaluation: {}. "
                "Comparison: {} ({}), {}, {}."
            )
            logger.warning(msg.format(e, val, test_prop, mode, test_value))
        else:
            if test:
                graphorcurve.update({apply_prop: apply_value})
