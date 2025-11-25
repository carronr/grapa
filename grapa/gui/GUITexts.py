# -*- coding: utf-8 -*-
"""
A popup window ti edit graph text legends, titles and text annotations.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, font as tkfont
from copy import deepcopy
import logging


path = os.path.abspath(os.path.join(".", "..", ".."))
if path not in sys.path:
    sys.path.append(path)

from grapa.utils.string_manipulations import strToVar, strUnescapeIter, varToStr
from grapa.utils.error_management import issue_warning
from grapa.gui.widgets_tooltip import CreateToolTip
from grapa.gui.widgets_custom import (
    EntryVar,
    CheckbuttonVar,
    OptionMenuVar,
    ComboboxVar,
    FrameScrollable,
)

logger = logging.getLogger(__name__)


class GuiManagerAnnotations(tk.Frame):
    """
    Provides a popup window for quick and easier modification of text
    annotations, legends, legend titles and graph titles.
    """

    # Implementation started with a not-great code for textx, then a more
    # tunable solution was designed for the rest. Previous not modified as it
    # was working

    def __init__(self, master, graph, funcupdate):
        self.master = master
        tk.Frame.__init__(self, master)
        self.funcupdate = funcupdate
        self.graph = graph
        # save intial values
        self.oldattr = {
            "text": deepcopy(graph.attr("text")),
            "textxy": deepcopy(graph.attr("textxy", [""])),
            "textargs": deepcopy(graph.attr("textargs", [{}])),
            "legendtitle": deepcopy(graph.attr("legendtitle")),
        }
        self.attr = {}
        # prepare
        self.legtit_fields = []
        self.legtit_vars = []
        self.legend_fields = []
        self.legend_vars = []
        self.text_fields = []
        self.text_vars = []
        self.annotations, self.annotationsnew = [], None
        self._config_text()
        self._config_leg_tit()
        self._config_legend()
        # fill GUI
        self.main = FrameScrollable(self.master)
        self.fontBold = None
        self._init_fonts(self.main)
        self.main.pack(side="top", fill="both", expand=True)
        self._fill_ui_main(self.main.child)
        self.master.bind("<Return>", lambda event: self.go())

    def _init_fonts(self, frame):
        a = tk.Label(frame, text="")
        self.fontBold = tkfont.Font(font=a["font"])
        self.fontBold.configure(weight="bold")

    def _config_text(self):
        self.text_fields = []
        self.text_fields.append(
            {
                "label": "visible",
                "field": "Checkbutton",
                "fromkwargs": "visible",
                "tooltip": "Visible?",
                "default": True,
            }
        )
        self.text_fields.append(
            {"label": "text", "width": 20, "tooltip": "Enter a text"}
        )
        self.text_fields.append(
            {
                "label": "color",
                "width": 7,
                "fromkwargs": "color",
                "tooltip": 'Color. Example: "r", "[1,0,1]", or "pink"',
                "field": "Combobox",
                "values": ["r", "pink", "[1,0,1]"],
            }
        )
        self.text_fields.append(
            {
                "label": "fontsize",
                "width": 7,
                "fromkwargs": "fontsize",
                "tooltip": 'Fontsize. Example: "16"',
            }
        )
        self.text_fields.append(
            {
                "label": "xytext",
                "width": 12,
                "fromkwargs": "xytext",
                "tooltip": 'Position of the text. Example: "(0.2,0.3)"',
            }
        )
        self.text_fields.append(
            {
                "label": "textcoords",
                "field": "OptionMenu",
                "fromkwargs": "textcoords",
                "tooltip": 'By default "figure fraction". Can also be "axes fraction", "figure pixels", "data", etc.',
                "values": [
                    "figure fraction",
                    "figure pixels",
                    "figure points",
                    "axes fraction",
                    "axes pixels",
                    "axes points",
                    "data",
                ],
            }
        )
        self.text_fields.append(
            {
                "label": "other properties",
                "width": 50,
                "tooltip": "kwargs to ax.annotate. Example: \"{'verticalalignment':'center', 'xy':(0.5,0.5), 'xycoords': 'figure fraction', 'rotation': 90, 'arrowprops':{'facecolor':'blue', 'shrink':0.05}}\"\nSee https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.annotate.html",
                "default": {},
                "field": "Combobox",
                "values": [
                    "{}",
                    "{'verticalalignment':'center'}",
                    "{'xy':(0.5,0.5), 'xycoords': 'figure fraction', 'arrowprops':{'facecolor':'b', 'shrink':0.05}}",
                    "{'rotation': 90}",
                ],
            }
        )
        for field in self.text_fields:
            if "field" not in field:
                field["field"] = "Entry"
            if "width" not in field:
                field["width"] = -1
            if "fromkwargs" not in field:
                field["fromkwargs"] = ""
        self.text_vars = []
        for _field in self.text_fields:
            self.text_vars.append([])

    def _config_leg_tit(self):
        self.legtit_fields = []
        self.legtit_fields.append({"label": "Legend title", "width": 0})
        self.legtit_fields.append(
            {"label": "title", "width": 20, "tooltip": "Enter a title to the legend"}
        )
        self.legtit_fields.append(
            {
                "label": "color",
                "width": 7,
                "fromkwargs": "color",
                "tooltip": 'Color. Example: "r", "[1,0,1]", or "pink"',
                "field": "Combobox",
                "values": ["r", "pink", "[1,0,1]"],
            }
        )
        self.legtit_fields.append(
            {
                "label": "fontsize",
                "width": 7,
                "fromkwargs": "fontsize",
                "tooltip": 'Numeric, eg. "20"',
            }
        )
        self.legtit_fields.append(
            {
                "label": "align",
                "field": "OptionMenu",
                "fromkwargs": "align",
                "tooltip": 'Horizontal alignment of title. eg. "left", "right".\nFor legendtitle: not a regular matplotlib keyword.',
                "values": ["left", "center", "right"],
            }
        )
        self.legtit_fields.append(
            {
                "label": "position (x,y)",
                "width": 10,
                "fromkwargs": "position",
                "tooltip": 'legendtitle: (x,y) position shift in pixels, eg. "(20,0)"\ntitle: (x,y) relative position, eg. "(0,1)"',
                "field": "Combobox",
                "values": ["(10,0)", "(-10,20)"],
            }
        )
        self.legtit_fields.append(
            {
                "label": "other properties",
                "width": 50,
                "tooltip": "kwargs to legend.set_title()",
            }
        )
        for field in self.legtit_fields:
            if "field" not in field:
                field["field"] = "Entry"
            if "width" not in field:
                field["width"] = -1
        self.legtit_vars = []
        for _field in self.legtit_fields:
            self.legtit_vars.append([])

    def _config_legend(self):
        self.legend_fields = []
        self.legend_fields.append({"label": "Legend", "width": 0})
        self.legend_fields.append(
            {
                "label": "color",
                "width": 7,
                "tooltip": 'Color. Example: "r", "[1,0,1]", or "pink"\n"curve" will give same color as trace',
                "fromkwargs": "color",
                "field": "Combobox",
                "values": ["r", "pink", "[1,0,1]", "curve"],
            }
        )
        self.legend_fields.append(
            {
                "label": "fontsize",
                "width": 7,
                "tooltip": 'Numeric, eg. "20"',
                "fromkwargs": "fontsize",
            }
        )
        self.legend_fields.append(
            {
                "label": "loc",
                "field": "OptionMenu",
                "tooltip": 'Location, eg. "nw" (north west), "se", "center"',
                "fromkwargs": "loc",
                "default": "best",
                "values": [
                    "best",
                    "nw",
                    "n",
                    "ne",
                    "w",
                    "center",
                    "e",
                    "sw",
                    "s",
                    "se",
                ],
            }
        )
        self.legend_fields.append(
            {
                "label": "ncol",
                "width": 6,
                "tooltip": 'Number of columns, eg. "2"',
                "fromkwargs": "ncol",
                "casttype": int,
            }
        )
        self.legend_fields.append(
            {
                "label": "bbox_to_anchor",
                "width": 12,
                "tooltip": '(x,y) shift to base position in axes fraction, eg. "(0.1, 0.2)"',
                "fromkwargs": "bbox_to_anchor",
                "field": "Combobox",
                "values": ["(0,0.1)", "(0.1,0.2)"],
            }
        )
        self.legend_fields.append(
            {
                "label": "other properties",
                "width": 55,
                "tooltip": "kwargs to ax.legend(), see https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.legend.html",
                "field": "Combobox",
                "values": [
                    "{}",
                    "{'frameon': True, 'framealpha': 1}",
                    "{'framealpha': 1}",
                    "{'numpoints': 2}",
                ],
            }
        )
        for field in self.legend_fields:
            if "field" not in field:
                field["field"] = "Entry"
            if "width" not in field:
                field["width"] = -1
        self.legend_vars = []
        for _field in self.legend_fields:
            self.legend_vars.append([])

    def _parse_graph(self):
        # retrieve data from Graph object, ensure formatting is ok
        self.attr = {}
        self.attr["text"] = deepcopy(self.graph.attr("text"))
        self.attr["textxy"] = deepcopy(self.graph.attr("textxy", [""]))
        self.attr["textargs"] = deepcopy(self.graph.attr("textargs", [{}]))
        if not isinstance(self.attr["text"], list):
            self.attr["text"] = [self.attr["text"]]
        if not isinstance(self.attr["textxy"], list):
            self.attr["textxy"] = [self.attr["textxy"]]
        if not isinstance(self.attr["textargs"], list):
            self.attr["textargs"] = [self.attr["textargs"]]
        while len(self.attr["textxy"]) < len(self.attr["text"]):
            self.attr["textxy"].append("")
        while len(self.attr["textargs"]) < len(self.attr["text"]):
            self.attr["textargs"].append({})
        # override xytext in textargs with content of textxy - later can ignore
        # textxy
        for i in range(len(self.attr["textxy"])):
            if self.attr["textxy"][i] != "":
                self.attr["textargs"][i].update({"xytext": self.attr["textxy"][i]})
        if self.attr["text"] == [""]:
            self.attr["text"] = []

    def _fill_ui_main(self, frame):
        self.legend_grid = tk.Frame(frame)
        self.legend_grid.pack(side="top", anchor="w")
        self._fill_ui_legend_grid(self.legend_grid)

        self.leg_tit_grid = tk.Frame(frame, pady=10)
        self.leg_tit_grid.pack(side="top", anchor="w")
        self._fill_ui_legend_title_grid(self.leg_tit_grid)

        self.frame_grid = tk.Frame(frame)
        self.frame_grid.pack(side="top", anchor="w")
        self._fill_ui_text_grid(self.frame_grid)

        frm = tk.Frame(frame)
        frm.pack()
        self._fill_ui_buttons(frm)

    def _fill_ui_buttons(self, frame):
        self.ButtonRevert = tk.Button(
            frame, text="Revert to initial", fg="grey", command=self._revert
        )
        self.ButtonRevert.pack(side="left")
        self.ButtonQuit = tk.Button(
            frame, text="Quit", fg="grey", command=self.close_windows
        )
        self.ButtonQuit.pack(side="left", padx=10)
        self.ButtonGo = tk.Button(
            frame, text="Update graph", command=self.go, font=self.fontBold
        )
        self.ButtonGo.pack(side="right", padx=30)

    def _fill_ui_text_grid(self, frame):
        self._parse_graph()
        # remove previsously created widgets, start afresh
        if self.annotationsnew is not None:
            self.annotationsnew.destroy()
            self.annotationsnew = None
        for row in self.annotations:
            for field in row:
                if hasattr(field, "destroy"):
                    field.destroy()
        self.annotations = []
        args = []
        for j in range(len(self.attr["text"])):
            self.annotations.append([""] * len(self.text_fields))
            args.append(deepcopy(self.attr["textargs"][j]))
        self.annotations.append([""] * len(self.text_fields))  # for new text
        args.append({})  # defautl new
        for j in range(len(self.text_fields)):
            field = self.text_fields[j]
            e = tk.Label(frame, text=field["label"])
            e.grid(row=0, column=(j + 1))
            if "tooltip" in field:
                CreateToolTip(e, field["tooltip"])
            # data fields and new text fields
            kw = field["fromkwargs"]
            for i in range(len(self.attr["text"]) + 1):
                value = ""
                if i == len(self.attr["text"]):  # "new" text field
                    value = ""
                elif field["label"] == "text":  # text: from attr['text']
                    value = varToStr(self.attr["text"][i])
                else:  # otherwise must grab it from the textargs dict
                    if kw != "" and kw in args[i]:
                        value = args[i][kw]
                        del args[i][kw]
                    elif j == len(self.text_fields) - 1:
                        value = varToStr(args[i])
                if value == "" and "default" in field:
                    value = field["default"]
                e = None
                if field["field"] == "OptionMenu":
                    e = OptionMenuVar(frame, values=field["values"], default=str(value))
                elif field["field"] == "Checkbutton":
                    defval = bool(value) if value != "" else False
                    e = CheckbuttonVar(frame, "", default=defval)
                if field["field"] == "Combobox":
                    e = ComboboxVar(
                        frame, field["values"], default=str(value), width=field["width"]
                    )
                elif field["width"] > 0:
                    e = EntryVar(frame, str(value), width=field["width"])
                if e is not None:
                    self.annotations[i][j] = e
                    e.grid(column=(j + 1), row=(i + 1), pady=0, ipady=0, padx=1)
        tk.Label(frame, text="Text", font=self.fontBold).grid(row=0, column=0)
        self.annotationsnew = tk.Label(frame, text="New", font=self.fontBold)
        self.annotationsnew.grid(row=(len(self.attr["text"]) + 1), column=0)
        for j in range(len(self.text_fields)):
            if self.text_fields[j]["label"] == "xytext":
                self.annotations[-1][j].set("(0.05, 0.95)")

    def _fill_ui_legend_grid(self, frame):
        Labels = []
        for j in range(len(self.legend_fields)):
            field = self.legend_fields[j]
            Labels.append(tk.Label(frame, text=field["label"]))
            if field["width"] == 0:
                Labels[-1].configure(font=self.fontBold)
            Labels[j].grid(row=(0 if field["width"] != 0 else 1), column=j)
            if "tooltip" in field:
                CreateToolTip(Labels[j], field["tooltip"])
            self.legend_vars[j].append(tk.StringVar())
            e = None
            if field["field"] == "OptionMenu":
                e = tk.OptionMenu(frame, self.legend_vars[j][-1], *field["values"])
            if field["field"] == "Combobox":
                e = ttk.Combobox(
                    frame,
                    textvariable=self.legend_vars[j][-1],
                    values=field["values"],
                    width=field["width"],
                )
            elif field["width"] > 0:
                e = tk.Entry(
                    frame, width=field["width"], textvariable=self.legend_vars[j][-1]
                )
            if e is not None:
                e.grid(column=j, row=len(self.legend_vars[j]), padx=1)
        self._legend_title_fill_values(
            "legendproperties", self.legend_fields, self.legend_vars
        )

    def _fill_ui_legend_title_grid(self, frame):
        Labels = []
        # legend title
        for j in range(len(self.legtit_fields)):
            field = self.legtit_fields[j]
            if field["width"] != 0:
                Labels.append(tk.Label(frame, text=field["label"]))
                Labels[-1].grid(row=0, column=j)
            if "tooltip" in field:
                CreateToolTip(Labels[-1], field["tooltip"])
            self.legtit_vars[j].append(tk.StringVar())
            e = None
            if field["field"] == "OptionMenu":
                e = tk.OptionMenu(frame, self.legtit_vars[j][-1], *field["values"])
            if field["field"] == "Combobox":
                e = ttk.Combobox(
                    frame,
                    textvariable=self.legtit_vars[j][-1],
                    values=field["values"],
                    width=field["width"],
                )
            elif field["width"] > 0:
                e = tk.Entry(
                    frame, width=field["width"], textvariable=self.legtit_vars[j][-1]
                )
            if e is not None:
                e.grid(column=j, row=len(self.legtit_vars[j]), ipady=0, pady=0, padx=1)
        # graph title
        self.legtit_fields[5].update({"values": ["(0,1)", "(0.5, 0.95)"]})
        for j in range(len(self.legtit_fields)):
            field = self.legtit_fields[j]
            # graph title
            self.legtit_vars[j].append(tk.StringVar())
            if "field" not in field:
                field["field"] = "Entry"
            e = None
            if field["field"] == "OptionMenu":
                e = tk.OptionMenu(frame, self.legtit_vars[j][-1], *field["values"])
            if field["field"] == "Combobox":
                e = ttk.Combobox(
                    frame,
                    textvariable=self.legtit_vars[j][-1],
                    values=field["values"],
                    width=field["width"],
                )
            elif field["width"] > 0:
                e = tk.Entry(
                    frame, width=field["width"], textvariable=self.legtit_vars[j][-1]
                )
            if e is not None:
                e.grid(column=j, row=len(self.legtit_vars[j]), ipady=0, pady=0, padx=1)
        tk.Label(frame, text="Legend title", font=self.fontBold).grid(row=1, column=0)
        tk.Label(frame, text="Graph title", font=self.fontBold).grid(
            row=2, column=0, sticky="W"
        )
        self._legend_title_fill_values(
            "legendtitle", self.legtit_fields, self.legtit_vars, row=0
        )
        self._legend_title_fill_values(
            "title", self.legtit_fields, self.legtit_vars, row=1
        )

    def _legend_title_fill_values(self, attribute, fields, vars_, row=0):
        attr = deepcopy(self.graph.attr(attribute))
        vals = [""] * len(fields)
        vals[-1] = attr
        if attribute in ["title", "legendtitle"]:
            if not isinstance(attr, list):
                attr = [attr, {}]
            vals[1] = attr[0]
            vals[-1] = attr[1]
        elif attribute in ["legendproperties"]:
            if vals[-1] == "":
                vals[-1] = {}
        for i in range(len(fields)):
            if "fromkwargs" in fields[i]:
                key = fields[i]["fromkwargs"]
                if attribute == "title" and key == "align":
                    key = "loc"  # different keyword for same stuff
                if key in vals[-1]:
                    vals[i] = str(vals[-1][key])
                    del vals[-1][key]
            if vals[i] == "" and "default" in fields[i]:
                vals[i] = fields[i]["default"]
        vals[1] = vals[1].replace("\n", "\\n")
        for j in range(0, len(vars_)):
            vars_[j][row].set(vals[j])

    def go(self):
        # text annotations
        out = {"text": [], "textxy": [], "textargs": []}
        i = 0
        for ann in self.annotations:
            if ann[1].get() != "":
                out["text"].append(strUnescapeIter(ann[1].get()))  # as str
                args = strToVar(ann[-1].get())
                if not isinstance(args, dict):
                    msg = "GuiManagerAnnotations invalid input: {} should be a dict."
                    issue_warning(logger, msg.format(args))
                    args = {}
                for j in range(len(self.text_fields) - 1):
                    if self.text_fields[j]["fromkwargs"] != "":
                        val = strToVar(ann[j].get())
                        if self.text_fields[j]["label"] == "xytext":
                            if not isinstance(val, (list, tuple)):
                                msg = (
                                    "GuiManagerAnnotations invalid input: (xytext) {}"
                                    "should be a tuple or dict with 2 elements"
                                    "(coordinates)."
                                )
                                issue_warning(logger, msg.format(val))
                        if self.text_fields[j]["label"] == "visible?" and val:
                            val = ""
                        if val != "":
                            args.update({self.text_fields[j]["fromkwargs"]: val})
                out["textxy"].append("")
                out["textargs"].append(args)
            i += 1
        # legend title, graph title
        keywords = ["legendtitle", "title"]
        for j in range(len(keywords)):
            tit = strUnescapeIter(self.legtit_vars[1][j].get())  # as str
            kw = strToVar(self.legtit_vars[-1][j].get())
            if not isinstance(kw, dict):
                msg = (
                    "GuiManagerAnnotations invalid input: {} should be a dict"
                    "(legendtitle)."
                )
                issue_warning(logger, msg.format(kw))
                kw = {}
            for i in range(len(self.legtit_fields)):
                if "fromkwargs" in self.legtit_fields[i]:
                    if self.legtit_vars[i][j].get() != "":
                        val = strToVar(self.legtit_vars[i][j].get())
                        if "casttype" in self.legtit_fields[i]:
                            val = self.legtit_fields[i]["casttype"](val)
                        kw.update({self.legtit_fields[i]["fromkwargs"]: val})
            if keywords[j] == "title" and "align" in kw:
                kw["loc"] = kw["align"]
                del kw["align"]
            legtit = tit if kw in ["", {}] else [tit, kw]
            out.update({keywords[j]: legtit})
        # legend properties
        kw = strToVar(self.legend_vars[-1][0].get())
        if not isinstance(kw, dict):
            msg = "GuiManagerAnnotations invalid input: {} should be a dict (legend)."
            issue_warning(logger, msg.format(kw))
            kw = {}
        for i in range(len(self.legend_fields)):
            if "fromkwargs" in self.legend_fields[i]:
                if self.legend_vars[i][0].get() != "":
                    val = strToVar(self.legend_vars[i][0].get())
                    if "casttype" in self.legend_fields[i]:
                        val = self.legend_fields[i]["casttype"](val)
                    kw.update({self.legend_fields[i]["fromkwargs"]: val})
        out.update({"legendproperties": kw})
        # graph title
        # perform update, finish
        # print('sent to GUI', out)
        self.funcupdate(out)
        self.refresh_ui()

    def _revert(self):
        self.funcupdate(self.oldattr)
        self.refresh_ui()

    def refresh_ui(self):
        self._fill_ui_text_grid(self.frame_grid)
        self._legend_title_fill_values(
            "legendtitle", self.legtit_fields, self.legtit_vars
        )

    def close_windows(self):
        self.master.destroy()


def build_ui():
    from grapa.graph import Graph

    root = tk.Tk()
    graph = Graph(r".\..\examples\fancyAnnotations.txt")

    def funcupdate(*args):
        for a in args:
            graph.update(a)

    app = GuiManagerAnnotations(root, graph, funcupdate)
    app.mainloop()


if __name__ == "__main__":
    build_ui()
