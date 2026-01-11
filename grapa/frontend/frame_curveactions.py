# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import os
import logging
import warnings
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
from typing import TYPE_CHECKING

import numpy as np

from grapa.curve import Curve
from grapa.graph import Graph
from grapa.frontend.widgets_custom import FrameTitleContentHide
from grapa.frontend.widgets_tooltip import CreateToolTip
from grapa.shared.error_management import GrapaError, issue_warning, FileNotReadError
from grapa.shared.funcgui import FuncGUI, CurveActionRequestToGui
from grapa.shared.string_manipulations import strToVar

if TYPE_CHECKING:
    from grapa.GUI import Application

logger = logging.getLogger(__name__)


class GUIFrameActionsCurves:
    """
    Handles part with actions specific to curves
    Contains a Frame, to be embedded in a wider application (.frame.pack())
    Some methods will call methods of application
    """

    def __init__(self, master, application: "Application", **kwargs):
        self.frame = tk.Frame(master, **kwargs)
        self.app = application
        self.list_callinfos = []
        self.frame_action = None  # define later, here only defines the name
        self._widgets = []
        self._create_widgets(self.frame)
        self.previous = {"curve": None, "type": None, "le": None}
        self.app.observables["focusTree"].register(self._cw_frame_action)

    def update_ui(self):
        """Update widgets"""
        curve = self.app.get_selected_curves(multiple=False)
        if len(curve) > 0 and curve[0] > -1:
            curve = self.app.graph()[curve[0]]
        self._cw_frame_action(curve, force=True)

    def _create_widgets(self, frame):
        """Create widgets: itle frame"""
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w", fill=tk.X)
        lbl = tk.Label(
            fr, text="Actions specific to selected Curve", font=self.app.fonts["bold"]
        )
        lbl.pack(side="left", anchor="center")
        hline = FrameTitleContentHide.frame_hline(fr)
        hline.pack(side="left", anchor="center", fill=tk.X, expand=1, padx=5)
        # content
        self.frame_action = tk.Frame(frame)
        self.frame_action.pack(side="top", fill=tk.X, padx=5)
        # argsFunc=[-1])

    def _cw_frame_action(self, curve, *_args, force=False):
        # identify index of active curve
        graph = self.app.graph()
        graph_i = None
        for c, curve2 in enumerate(graph):
            if curve2 == curve:
                graph_i = c
                break
        # retrieve list of actions
        func_list = []
        if graph_i is not None:
            try:
                func_list = curve.funcListGUI(graph=graph, graph_i=graph_i)
            except Exception as e:
                msg = "{}.funcListGUI, graph_i {}, graph {}. {}: {}."
                msg = msg.format(type(curve), graph_i, graph, type(e), e)
                logger.error(msg, exc_info=True)
                return  # no need to raise here

        # args: specific attribute selected. Useless here
        if (
            not force
            and curve == self.previous["curve"]
            and isinstance(curve, self.previous["type"])
            and len(func_list) == self.previous["le"]
        ):
            # no need to change anything, do not update anything
            # print('GUIFrameActionsCurves: can think to not update actions')
            pass
        self.previous = {"curve": curve, "type": type(curve), "le": len(func_list)}
        # destroy all existing widgets
        for line in self._widgets:  # widgets
            for widget in line:
                widget.destroy()
        self.list_callinfos = []
        self._widgets = []
        # create new widgets
        for j, line in enumerate(func_list):
            if not isinstance(line, FuncGUI):
                line = FuncGUI(None, None).init_legacy(line)
            # create widgets
            callinfo, widgets = self._define_widgets_funcgui(
                line, self.frame_action, self.call_action, j
            )
            self.list_callinfos.append(callinfo)
            self._widgets.append(widgets)
        if len(self.list_callinfos) == 0:
            widget = tk.Label(self.frame_action, text="No possible action.")
            widget.pack(side="top", anchor="w")
            self._widgets.append([widget])

    def _define_widgets_funcgui(self, funcgui, frame, callback, callbackarg):
        """
        Creates a frame and widgets inside. Returns:
        callinfo: [func, StringVar1, StringVar2, ..., {hiddenvars}]
        widgets: [innerFrame, widget1, widget2, ...]
        frame: where the widgets are created. substructure will be provided
        callback: function to call when user validates his input
        callbackarg: index of guiAction. callback(callbackarg)
        """
        callinfo = {
            "func": funcgui.func,
            "args": [],
            "kwargs": dict(funcgui.hiddenvars),
        }
        widgets = []  # list of widgets, to be able to destroy later
        folderbase = self.app.get_folder()
        # create inner frame
        fr = tk.Frame(frame)
        fr.pack(side="top", anchor="w", fill=tk.X)
        widgets.append(fr)

        # validation button
        _cw_buttonvalidation(fr, funcgui, widgets, callback, callbackarg)

        # list of widgets
        lookuppreviouswidget = {}
        for field in funcgui.fields:
            options = dict(field["options"])
            widgetclass = field["widgetclass"]
            to_create = []

            # first, a Label widget for to help the user
            _cw_field_label(fr, field["label"], widgets)

            if widgetclass in [None, "None"]:
                # do not create any widget; purpose: show the label
                continue
            if widgetclass in ["Frame"]:  # 'Frame' interpreted as to create a new line
                fr = tk.Frame(frame)
                fr.pack(side="top", anchor="w", fill=tk.X)
                widgets.append(fr)  # inner Frame
                continue  # stop there, go to next widget

            # widgetname: tranform into reference to class
            try:
                if widgetclass in ["Combobox"]:  # Combobox
                    widgetclass = getattr(ttk, widgetclass)
                elif widgetclass in [FuncGUI.ASKFORFILENAME]:
                    widgetclass = getattr(tk, "Entry")
                    kwtmp = {}
                    for key in ["initialdir"]:
                        if key in options:
                            kwtmp[key] = options[key]
                            del options[key]
                    to_create.append((FuncGUI.ASKFORFILENAME, (), kwtmp))
                else:
                    widgetclass = getattr(tk, widgetclass)
            except Exception as e:
                msg = "ERROR Actions Curves, cannot create. class {}. Exception {} {}"
                print(msg.format(widgetclass, type(e), e))
                continue

            # create stringvar
            stringvar = _cw_field_stringvar(widgetclass, field["value"])
            if field["keyword"] is None:
                callinfo["args"].append(stringvar)
            else:
                callinfo["kwargs"].update({field["keyword"]: stringvar})
            _cw_field_options(options, widgetclass, stringvar, funcgui, field["label"])

            # create widget
            try:
                widget = widgetclass(fr, **options)
            except Exception as e:
                msg = "Could not create widget %s, %s, %s. %s: %s."
                msga = (field["label"], widgetclass.__name__, options, type(e), e)
                issue_warning(logger, msg, *msga)
                continue

            widget.pack(side="left", anchor="w")
            widgets.append(widget)
            _cw_field_tocreate(to_create, fr, stringvar, widget, widgets, folderbase)
            # bind
            _cw_field_bind(
                field["bind"], widget, stringvar, widgets, lookuppreviouswidget, frame
            )
            widget.bind("<Return>", lambda event, j_=callbackarg: callback(j_))
        # end of loop, return
        return callinfo, widgets

    def call_action(self, j):
        """when user validates his input for curve action"""
        self.app.store_selected_curves()
        curve = self.app.get_selected_curves(multiple=False)[0]
        graph = self.app.graph()
        info = self.list_callinfos[j]
        # retrieve function to call
        func, args, kwargs = info["func"], info["args"], dict(info["kwargs"])
        # retrieve values of user-adjustable parameters
        args = [strToVar(a.get()) for a in args]
        for key in kwargs:
            if isinstance(kwargs[key], tk.StringVar):
                kwargs[key] = strToVar(kwargs[key].get())
            elif isinstance(kwargs[key], tk.BooleanVar):
                kwargs[key] = strToVar(kwargs[key].get())
            # else: hiddenvars
        # check if func is method of Graph object and not of the Curve
        if not hasattr(func, "__self__"):
            args = [self.app.graph()] + args
            # to get call as Graph.func(graph, *args)

        def execute_func(curve, func, args, kwargs):
            # execute curve action
            if func == CurveActionRequestToGui.OPEN_FILE:
                try:
                    self.app.open_file(*args[1:])  # by construction args[0] is graph
                except FileNotReadError as e:
                    msg = "%s: %s."
                    print(msg % (type(e), e))
                return True

            graph = self.app.graph()
            try:
                with warnings.catch_warnings(record=True) as w:
                    res = func(*args, **kwargs)
                    # now must ignore, otherwise issue_warning() result in infinite loop
                    warnings.simplefilter("ignore")
                    for wa in w:
                        ms = "While executing function {}. {}: {}."
                        msg = ms.format(func.__name__, wa.category.__name__, wa.message)
                        issue_warning(logger, msg)
            except GrapaError as e:
                if e.report_in_gui:
                    ms = "While executing function {} with args {} kwargs {}. {}: {}."
                    msg = ms.format(func, args, kwargs, type(e), e)
                    logger.error(msg, exc_info=True)
                return False
            except Exception as e:  # cannot afford to fail
                ms = "While executing function {} with args {} kwargs {}. {}: {}."
                msg = ms.format(func, args, kwargs, type(e), e)
                logger.error(msg, exc_info=True)
                # raise GrapaError(msg) from e
                return False

            if self.app.if_print_commands():
                try:
                    subject = func.__module__
                    if "graph" in subject:
                        subject = "graph"
                    elif "curve" in subject:
                        subject = "graph[" + str(curve) + "]"
                    else:
                        ms = "callAction print: subject not determined ({}, {}, {})"
                        issue_warning(logger, ms.format(subject, func.__name__, j))
                    ms = "curve = {}.{}({})"
                    msgformat = ms.format(
                        subject, func.__name__, self.app.args_to_str(*args, **kwargs)
                    )
                    print(msgformat)
                except Exception:
                    pass  # error while doing useless output does not matter
            # where to place new Curves
            idx = curve + 1
            while idx < len(graph):  # increments idx in specific cases
                type_ = graph[idx].attr("type")
                if not type_.startswith("scatter") and not type_.startswith("errorbar"):
                    break
                idx += 1
            # insert newly created Curves
            if isinstance(res, Curve):
                self.app.call_graph_method("append", res, idx=idx)
            elif (
                isinstance(res, list)  # if list of Curves
                and np.array([isinstance(c, Curve) for c in res]).all()
            ):
                self.app.call_graph_method("append", res, idx=idx)
            elif isinstance(res, Graph):  # if returns a Graph, create a new one
                folder = self.app.get_folder()
                dpi = self.app.get_tab_properties()["dpi"]
                path = graph.filename
                if not isinstance(path, str) or path == "":
                    title = "Curve action output"
                else:
                    title = os.path.splitext(os.path.basename(path))[0]
                self.app.graph(res, title=title, folder=folder, dpi=dpi)
            elif not isinstance(res, bool) or res is not True:  # don't, only if "True"
                print("Curve action output:")
                print(res)
            # TO CHECK: if want to print anything if True, etc.
            return True

        # handling actions on multiple Curves
        to_execute = {curve: func}
        curves = self.app.get_selected_curves(multiple=True)
        if len(curves) > 1:
            func_gui0 = graph[curve].funcListGUI(graph=graph, graph_i=curve)[j]
            if not isinstance(func_gui0, FuncGUI):
                func_gui0 = FuncGUI(None, None).init_legacy(func_gui0)
            for c in curves:
                if c != curve:
                    # check that function is offered by other selected curves
                    list1 = graph[c].funcListGUI(graph=graph, graph_i=c)
                    if len(list1) > j:
                        func_gui1 = list1[j]
                        if not isinstance(func_gui1, FuncGUI):
                            func_gui1 = FuncGUI(None, None).init_legacy(func_gui1)
                        if func_gui0.is_similar(func_gui1):
                            to_execute.update({c: func_gui1.func})
        keys = list(to_execute.keys())
        keys.sort(reverse=True)
        # start execution by last one - to handle new curves
        for c in keys:
            if len(keys) > 1:
                lbl = graph[c].attr("label", "")
                if len(lbl) > 0:
                    lbl = "(" + lbl + ")"
                print("Action on Curve", c, lbl)
            execute_func(c, to_execute[c], args, kwargs)
        # after execution
        self.app.update_ui()


def _cw_buttonvalidation(fr, funcgui, widgets: list, callback, callbackarg):
    if funcgui.func is None:
        widget_button = tk.Label(fr, text=funcgui.textsave)
    else:
        widget_button = tk.Button(
            fr, text=funcgui.textsave, command=lambda j_=callbackarg: callback(j_)
        )
    widget_button.pack(side="left", anchor="w")
    widgets.append(widget_button)
    if len(funcgui.tooltiptext) > 0:
        CreateToolTip(widget_button, funcgui.tooltiptext)


def _cw_field_label(fr, text: str, widgets: list):
    widget = tk.Label(fr, text=text)
    widget.pack(side="left", anchor="w", fill=tk.X)
    widgets.append(widget)


def _cw_field_stringvar(widgetclass, value):
    # create stringvar
    if widgetclass == tk.Checkbutton:
        stringvar = tk.BooleanVar()
        stringvar.set(bool(value))
    else:
        stringvar = tk.StringVar()
        stringvar.set(str(value))
    return stringvar


def _cw_field_options(
    options, widgetclass, stringvar, funcgui, label
):  # fiueld["label"]
    # default size if widgetclass is Entry
    if widgetclass == tk.Entry and "width" not in options:
        widthtest = int(
            (40 - len(label) / 3 - len(funcgui.textsave) / 2) / len(funcgui.fields)
        )
        width = max(8, widthtest)
        if len(stringvar.get()) < 2:
            width = int(0.3 * width + 0.7)
        options.update({"width": width})
    # link to StringVar
    if widgetclass == tk.Checkbutton:
        options.update({"variable": stringvar})
    else:
        options.update({"textvariable": stringvar})


def _cw_field_bind(
    bind, widget, stringvar, widgets: list, lookuppreviouswidget: dict, frame
):
    if bind is None:
        return

    if bind == "beforespace":
        widget.bind(
            "<<ComboboxSelected>>",
            lambda event: event.widget.set(event.widget.get().split(" ")[0]),
        )
    elif bind == "previouswidgettogglereadonly":
        # -1 is itfuncgui, -2 its Label, previous widget is -3
        lookuppreviouswidget.update({stringvar._name: widgets[-3]})
        stringvar.trace_add(
            "write",
            lambda w, _1, _2: lookuppreviouswidget[w].config(
                state="readonly" if int(frame.getvar(w)) else "normal"
            ),
        )
    else:
        widget.bind("<<ComboboxSelected>>", bind)


def _cw_field_tocreate(to_create, fr, stringvar, widget, widgets: list, folder):

    def _browse_trigger(stringvar, kw, widget, folder):
        new = filedialog.askopenfilename(**kw)
        if new == "":
            return
        stringvar.set(os.path.relpath(new, folder))
        widget.focus_set()
        widget.event_generate("<Return>")

    emoji_fonts = ("Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji")
    for what, args, kw in to_create:
        if what == FuncGUI.ASKFORFILENAME:
            button = tk.Button(
                fr,
                text="\U0001f50d",
                font=(emoji_fonts, 12),
                padx=1,
                pady=1,
                borderwidth=1,
                highlightthickness=0,
                command=lambda kw=kw: _browse_trigger(stringvar, kw, widget, folder),
            )
            button.pack(side="left", anchor="w")
            widgets.append(button)
        else:
            msg = "Not implemented create widget to_create: %s, %s"
            issue_warning(logger, msg, what, args, kw)
