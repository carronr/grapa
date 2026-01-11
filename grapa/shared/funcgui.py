"""Provides the FuncGUI class, and related functions to suggest to the user
possible actions depending on the Curve typeplot."""

import logging
from enum import Enum
from typing import Optional, TYPE_CHECKING

import numpy as np

from grapa.shared.string_manipulations import listToString, restructuredtext_to_text
from grapa.shared.error_management import issue_warning

if TYPE_CHECKING:
    from grapa.graph import Graph
    from grapa.curve import Curve


logger = logging.getLogger(__name__)


class CurveActionRequestToGui(Enum):
    """Special keywords that can be sent by a Curve as FuncGUI func
    e.g. request the GUI to open specific file"""

    OPEN_FILE = "_OPEN_FILE_"


def funclistgui_graph_axislabels(
    curve: "Curve",
    lookup_x: Optional[dict] = None,
    lookup_y: Optional[dict] = None,
    **kwargs
) -> list:
    """To suggests the user appropriate graph axis xlabel and ylabel.
    Relies on CurveSubClass.AXISLABELS_X and CurveSubClass.AXISLABELS_Y, with form:
    {"": ["Radius", "r", "m"]}. Key are the corresponding alter keywords.
    Superseded if the curve instance has attribute Curve.KEY_AXISLABEL_X and/or
    Curve.KEY_AXISLABEL_Y

    :param curve: the Curve selected by the user
    :param lookup_x: a dict to lookup new value for x unit, with key alter[0].
            Use-case: for example, if unit is "%" instead of "", or "nF" vs "nF cm-2"
    :param lookup_y: a dict to lookup new value for y unit, with key alter[1]
    :param kwargs: must contain kwargs "graph"
    :return: a list containing one FuncGUI element if appropriate, empty otherwise
    """
    out = []
    if "graph" not in kwargs:
        return out

    if lookup_x is None:
        lookup_x = {}
    if lookup_y is None:
        lookup_y = {}
    graph: "Graph" = kwargs["graph"]
    xlabel = graph.format_axis_label(graph.attr("xlabel"))
    ylabel = graph.format_axis_label(graph.attr("ylabel"))
    alter = graph.get_alter()
    attrs = {}
    # x axis
    x_raw = curve.attr(curve.KEY_AXISLABEL_X)
    if x_raw == "" and alter[0] in curve.AXISLABELS_X:
        x_raw = list(curve.AXISLABELS_X[alter[0]])
        if isinstance(x_raw, list) and alter[0] in lookup_x:
            x_raw[2] = lookup_x[alter[0]]
    if x_raw != "":
        if graph.format_axis_label(x_raw) != xlabel:
            attrs["xlabel"] = x_raw
    # yaxis
    y_raw = curve.attr(curve.KEY_AXISLABEL_Y)
    if y_raw == "" and alter[1] in curve.AXISLABELS_Y:
        y_raw = list(curve.AXISLABELS_Y[alter[1]])
        if isinstance(y_raw, list) and alter[1] in lookup_y:
            y_raw[2] = lookup_y[alter[1]]
    if y_raw != "":
        if graph.format_axis_label(y_raw) != ylabel:
            attrs["ylabel"] = y_raw
    # instructions for widgets
    if len(attrs) > 0:
        item = FuncGUI(graph.update_values_keys, "Save axis labels")
        item.set_hiddenvars({"keys": list(attrs.keys())})
        for key, value in attrs.items():
            item.append(key, value)
        out.append(item)
    return out


class FuncGUI:
    """
    The GUI request from the displayed Curve a list of FuncGUI objects.
    Each FuncGUI objects contains a possible action the user can perform.
    The class FuncGUI rationalizes the following: ::

      out.append([self.currentCalc,
                  'EQE current',
                  ['ROI', 'interpolate', 'spectrum'],
                  [[min(self.x()), max(self.x())], 'linear', fnames[0]],
                  {},
                  [{'width': 15},
                   {'width': 8, 'field': 'Combobox', 'values': ['linear', 'quadratic']},
                   {'width': 8, 'field': 'Combobox', 'values': fnames}]])

    Becomes: ::

       line = FuncGUI(self.currentCalc, 'EQE current')
       line.append('ROI', [min(self.x()), max(self.x())], {'width': 15})
       line.appendcbb('interpolate', 'linear', ['linear', 'quadratic'], {'width': 8})
       line.append("", "", "Frame")  # new line
       line.appendcbb('spectrum', fnames[0], fnames, options={'width': 8})
       out.append(line)

    """

    ASKFORFILENAME = "askforfilename"

    def __init__(self, func, textsave, hiddenvars=None, tooltiptext=""):
        """
        In principe, create FuncGUI with func, textsave, hiddenvars,
        then create fields with .append() one after the other
        One may also do FuncGUI(None, None).initLegacy(oldformat)
        """
        # store information
        self.func = func
        self.textsave = textsave
        self.hiddenvars = {}
        self.fields = []
        self.tooltiptext = None
        self.funcdocstring_alt = None
        # set if already provided
        self.set_hiddenvars(hiddenvars)
        self.set_tooltiptext(tooltiptext)

    def set_hiddenvars(self, hiddenvars):
        """hiddenvars is a dict of variables that will be provided to func
        when called by the GUI, in addition to the values provided by the user"""
        if isinstance(hiddenvars, dict):
            self.hiddenvars = dict(hiddenvars)
        elif hiddenvars is not None:
            msg = "FuncGUI: hiddenvars must be a dict, will be ignored: {}"
            issue_warning(logger, msg.format(hiddenvars))

    def set_tooltiptext(self, tooltiptext):
        """A short text that will be displayed as tooltip in the GUI"""
        if tooltiptext is None or isinstance(tooltiptext, str):
            self.tooltiptext = tooltiptext
        else:
            msg = "FuncGUI: tooltiptext must be a str, will be ignored: {}"
            issue_warning(logger, msg.format(tooltiptext))

    def set_funcdocstring_alt(self, documentation):
        """for use in printHelp"""
        if isinstance(documentation, str):
            self.funcdocstring_alt = documentation
        else:
            msg = "FuncGUI: documentation must be a str, will be ignored: {}."
            issue_warning(logger, msg.format(documentation))

    def append(
        self,
        label,
        value,
        widgetclass="Entry",
        bind=None,
        keyword=None,
        options=None,
        **kwargs
    ):
        """
        Examples

        :param label: 'my choice'
        :param value: 'b'
        :param widgetclass: 'Entry', 'Combobox'... a tk widget (or other library if
               implemented).
               Frame: new line.
               None: to not show in the GUI and not provide a value (e.g. display label)
        :param keyword: for the parameter to be submitted as keyword variable
        :param bind: a (validation) function on the widget value. Examples:

               - 'beforespace': <<ComboboxSelected>> remove all before first space

               - 'previouswidgettogglereadonly': toggle previous widget readonly state

               - other: <<ComboboxSelected>> call the function provided as 'bind'

        :param options: {'width': 8, 'values': ['a', 'b', 'c']}
        :param kwargs: additional keywords will be added into options
        """
        if options is None:
            options = {}
        options = dict(options)  # work on a copy as would modify the content
        for key, val in kwargs.items():
            options.update({key: val})
        # _propsFromOptions can modify options
        cls, bind, kw = self._opts_to_props(options, widgetclass, bind, keyword)
        if isinstance(value, np.ndarray):
            value = list(value)
        self.fields.append(
            {
                "label": label,
                "value": value,
                "options": options,
                "widgetclass": cls,
                "bind": bind,
                "keyword": kw,
            }
        )

    def appendcbb(
        self, label, value, values, bind=None, keyword=None, options=None, **kwargs
    ):
        """Notation shortcut to .append(), append a Combobox:
        .appendcbb("my value", "a", ["a", "b", "c"])"""
        opts = {} if options is None else dict(options)
        opts.update({"field": "Combobox", "values": values})
        return self.append(
            label, value, bind=bind, keyword=keyword, options=opts, **kwargs
        )

    def append_pickfile(self, label, value, initialdir: Optional[str]):
        """Notation shortcut to .append() an entry and a button to choose a file"""
        opts = {"field": self.ASKFORFILENAME, "initialdir": initialdir}
        return self.append(label, value, options=opts)

    def append_newline(self):
        """Shortcut to append a new line in the GUI"""
        self.append("", "", widgetclass="Frame")

    def move_item(self, old_idx, new_idx):
        """Use-case: a posteriori append a field, want to place it not at last position.
        For example, CurveMCA fit which line are prepared in base class of FitHandler
        Beware index if moving after old_idx."""
        self.fields.insert(new_idx, self.fields.pop(old_idx))

    def init_legacy(self, line):
        """Initialize a FuncGUI object from the legacy list format."""
        self.func = line[0]
        self.textsave = line[1]
        if len(line) > 4 and isinstance(line[4], dict):
            self.hiddenvars = line[4]
        for i in range(len(line[2])):
            label = line[2][i]
            value = line[3][i] if len(line) > 3 else ""
            if isinstance(value, (list, np.ndarray)):
                value = listToString(value)
            options = line[5][i] if len(line) > 5 else {}
            self.append(label, value, options=options)
        return self

    @staticmethod
    def _opts_to_props(options, widgetclass, bind, keyword):
        # to be called as
        # widgetclass, bind, keyword = self._optsToProps(options, widgetclass,
        #                                                bind, keyword)
        if "field" in options:  # overrides provided value
            widgetclass = options["field"]
            del options["field"]
        if "bind" in options:
            bind = options["bind"]
            del options["bind"]
        if "keyword" in options:
            keyword = options["keyword"]
            del options["keyword"]
        # some further checks
        if widgetclass == "Combobox":
            if "width" not in options:
                width = int(1.1 * max([len(v) for v in options["values"]]))
                options.update({"width": width})
        return widgetclass, bind, keyword

    def is_similar(self, other):
        """
        Determines if 2 FuncGUI objects are similar
        Checks for function name, textsave, hidden vars, fields labels
        Does NOT check for field values - user force same value for all
        """
        if (
            isinstance(other, FuncGUI)
            and self.textsave == other.textsave
            and len(self.fields) == len(other.fields)
            and len(self.hiddenvars) == len(other.hiddenvars)
        ):
            if (
                self.func in CurveActionRequestToGui
                and other.func in CurveActionRequestToGui
                and self.func == other.func
            ) or (self.func.__name__ == other.func.__name__):
                for i, fieldi in enumerate(self.fields):
                    if fieldi["label"] != other.fields[i]["label"]:
                        return False
                return True
        return False

    def func_docstring_to_text(self) -> list:
        """Returns str lines, with slight reformatting from function docstring"""
        legit = [f for f in self.fields if f["widgetclass"] not in [None, "Frame"]]
        nparammax = len(legit)
        docstring = self.funcdocstring_alt
        if docstring is None:
            docstring = self.func.__doc__
        lines = restructuredtext_to_text(docstring, nparammax=nparammax)
        if len(lines) == 0:
            lines.append("")
        lines = ["  " + li for li in lines]

        funclabel = self.textsave
        if len(self.fields) > 0 and self.fields[0]["widgetclass"] is not None:
            if "." in self.fields[0]["label"]:
                funclabel += " " + self.fields[0]["label"].split(".")[0]
            elif ":" in self.fields[0]["label"]:
                funclabel += " " + self.fields[0]["label"].split(":")[0]
        lines.insert(0, "- " + funclabel)
        return lines


class AlterListItem:
    r"""One item in the list of possible data transforms a Curve does offer.

    In essence a dataclass, yet need backwards compatibility.
    Replaces syntax ["label", ["alterx", "altery"], "semilogy"].
    Can be instantiated by AlterListItem(\*legacy_value).

    :param label: The label to shown in the GUI
    :param alter: a length-2 list ["alter_x", "alter_y"]. Can be ["", ""].
    :param typeplot: one of TYPEPLOTS items
    :param doc: a short human-readable description
    """

    # vs list: custom equality, ensure typeplot valid, TYPEPLOTS not part of gui

    TYPEPLOTS = [
        "",
        "plot",
        "plot norm.",
        "semilogy",
        "semilogy norm.",
        "semilogx",
        "loglog",
    ]
    TYPEPLOT_DEF = TYPEPLOTS[0]

    def __init__(self, label: str, alter: list, typeplot: str, doc: str = ""):
        self.label = str(label)
        self.alter = alter
        self.typeplot = typeplot
        if typeplot not in self.TYPEPLOTS:
            self.typeplot = self.TYPEPLOT_DEF
            msg = "AlterListItem: typeplot not in autorized list ({}), replace with {}."
            issue_warning(logger, msg.format(typeplot, self.typeplot))
        self.doc = str(doc)

    def __getitem__(self, idx):
        """So this class also behaves as a dict - its legacy implementation"""
        content = [self.label, self.alter, self.typeplot, self.doc]
        return content[idx]

    def __eq__(self, other):
        """So it is possible to test: if item in itemlist"""
        return (
            self.label == other[0]
            and self.alter == other[1]
            and self.typeplot == other[2]
        )  # no test on doc, intentional

    @classmethod
    def item_neutral(cls):
        """

        :return: an AlterListItem instance that does not transform the data.
        """
        return AlterListItem("no transform", ["", ""], "", "y vs x")
