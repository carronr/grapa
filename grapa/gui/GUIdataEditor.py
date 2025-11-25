# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:43:32 2018

@author: Romain
"""

import sys
import os
import numpy as np
import tkinter as tk

path = os.path.abspath(os.path.join(".", "..", ".."))
if path not in sys.path:
    sys.path.append(path)

# from grapa.mathModule import strToVar
# from grapa.gui.createToolTip import CreateToolTip
from grapa.gui.widgets_custom import EntryVar, LabelVar, ComboboxVar


class GuiDataEditor(tk.Frame):
    """
    This class creates a window to edit data of a Graph
    """

    fieldWidth = 55
    fieldHeight = 15
    curveWidth = 2 * fieldWidth + 25

    def __init__(self, master, graph, callback, curve_i=0):
        tk.Frame.__init__(self, master)
        self.master = master
        self.graph = graph
        self.callback = callback
        self._fields = {"idx": [], "curve": []}
        self._fieldsNum = 0
        # fill GUI
        # width = min(800, max(200, len(self.graph) * 200))
        # height = 400

        self._init_fonts(self.master)
        self.frameHead = tk.Frame(self.master, borderwidth=2, relief="raised")
        self.frameHead.pack(side="top", fill=tk.X)
        frame = tk.Frame(self.master)
        frame.pack(side="top", fill=tk.BOTH, expand=True)

        # --- create canvas with scrollbar ---
        def _on_configure(_event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.ca_top.configure(scrollregion=self.ca_top.bbox("all"))
            self.ca_lef.configure(scrollregion=self.ca_lef.bbox("all"))

        def _on_click_canvas(event):
            self._click_canvas(event)

        def _scrollx(*args):
            self.canvas.xview(*args)
            self.ca_top.xview(*args)
            self.ca_lef.xview(*args)

        def _scrolly(*args):
            self.canvas.yview(*args)
            self.ca_top.yview(*args)
            self.ca_lef.yview(*args)

        scrollbarx = tk.Scrollbar(frame, command=_scrollx, orient=tk.HORIZONTAL)
        scrollbary = tk.Scrollbar(frame, command=_scrolly, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(frame, bg="white")
        self.ca_top = tk.Canvas(frame, height=30)
        self.ca_lef = tk.Canvas(frame, width=45)
        self.ca_tol = tk.Canvas(frame, width=45, height=30)
        self.canvas.configure(yscrollcommand=scrollbary.set)
        self.canvas.configure(xscrollcommand=scrollbarx.set)
        self.ca_top.configure(xscrollcommand=scrollbarx.set)
        self.ca_lef.configure(yscrollcommand=scrollbary.set)
        scrollbarx.grid(row=2, column=1, sticky="we")
        scrollbary.grid(row=1, column=2, sticky="ns")
        self.canvas.grid(row=1, column=1, sticky="nsew")
        self.ca_top.grid(row=0, column=1, sticky="we")
        self.ca_lef.grid(row=1, column=0, sticky="ns")
        self.ca_tol.grid(row=0, column=0, sticky="")
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)
        # update scrollregion after starting 'mainloop'
        # when all widgets are in canvas
        self.canvas.bind("<Configure>", _on_configure)
        self.ca_top.bind("<Configure>", _on_configure)
        self.ca_lef.bind("<Configure>", _on_configure)
        self.canvas.bind("<Button-1>", _on_click_canvas)
        # populate different canvas
        self._fill_ui_head()
        self._fill_ui_curves()

    def _init_fonts(self, frame):
        import tkinter.font as font

        a = tk.Label(frame, text="")
        self.fontbold = font.Font(font=a["font"])
        self.fontbold.configure(weight="bold")

    def _click_canvas(self, event, c=None, i=None):
        if event is not None:
            c = 0
            for c_ in range(len(self._fields["curve"])):
                if self.canvas.canvasx(event.x) > self._fields["curve"][c_]["dx"]:
                    c = c_
            i = int(
                np.floor(
                    (self.canvas.canvasy(event.y) - self._fields["curve"][c]["dy0"])
                    / self.fieldHeight
                )
            )
        elif c is not None and i is not None:
            pass
        else:
            return
        x = self.graph[c].x()
        y = self.graph[c].y()
        if i < len(x):
            x, y = x[i], y[i]
        else:
            x, y = "", ""
        self._headCurve1.set(str(c))
        self._headCurve3.set(str(i))
        self._headCurve5.set(str(x))
        self._headCurve7.set(str(y))
        self._headCurBo1.set(str(c))
        self._headCurBo3.set(str(i))

    def save_data(self):
        c = int(self._headCurve1.get())
        i = int(self._headCurve3.get())
        x = float(self._headCurve5.get())
        y = float(self._headCurve7.get())
        x_, y_ = self.graph[c].x(), self.graph[c].y()
        x_[i] = x
        y_[i] = y
        self.graph[c].setX(x_)
        self.graph[c].setY(y_)
        self._fill_ui_curve_data(c)
        self._click_canvas(None, c=c, i=i)
        if self.callback is not None:
            self.callback()

    def delete_point(self):
        c = int(self._headCurve1.get())
        i = int(self._headCurve3.get())
        curve = self.graph[c]
        data = np.delete(curve.getData(), i, axis=1)
        curve.set_data(data)
        self._fill_ui_curve_data(c)
        self._fill_ui_curves_index()
        self._click_canvas(None, c=c, i=i)
        if self.callback is not None:
            self.callback()

    def insert_point(self):
        c = int(self._headCurBo1.get())
        i = int(self._headCurBo3.get())
        x = float(self._headCurBo5.get())
        y = float(self._headCurBo7.get())
        curve = self.graph[c]
        x_, y_ = curve.x(), curve.y()
        x_ = np.insert(x_, i, x)
        y_ = np.insert(y_, i, y)
        curve.set_data(np.array([x_, y_]))
        self._fill_ui_curve_data(c)
        self._fill_ui_curves_index()
        self._click_canvas(None, c=c, i=i)
        if self.callback is not None:
            self.callback()

    # user interface
    def _fill_ui_head(self):
        frame = self.frameHead
        frameup = tk.Frame(frame)
        framedw = tk.Frame(frame)
        frameup.pack(side="top", fill="x")
        framedw.pack(side="top", fill="x")
        self._fill_ui_head_top(frameup)
        self._fill_ui_head_bot(framedw)

    def _fill_ui_head_top(self, frame):
        self._headCurve0 = tk.Label(frame, text="Modify data, curve no")
        self._headCurve1 = LabelVar(frame, "")
        self._headCurve2 = tk.Label(frame, text="index")
        self._headCurve3 = LabelVar(frame, "", width=2)
        self._headCurve4 = tk.Label(frame, text="x")
        self._headCurve5 = EntryVar(frame, "", width=10)
        self._headCurve6 = tk.Label(frame, text="y")
        self._headCurve7 = EntryVar(frame, "", width=10)
        self._headCurve8 = tk.Button(frame, text="Save", command=self.save_data)
        self._headCurve9 = tk.Button(
            frame, text="Delete Point", command=self.delete_point
        )
        self._headCurve0.pack(side="left")
        self._headCurve1.pack(side="left")
        self._headCurve2.pack(side="left")
        self._headCurve3.pack(side="left")
        self._headCurve4.pack(side="left")
        self._headCurve5.pack(side="left")
        self._headCurve6.pack(side="left")
        self._headCurve7.pack(side="left")
        self._headCurve8.pack(side="left")
        self._headCurve9.pack(side="left")

    def _fill_ui_head_bot(self, frame):
        self._headCurBo0 = tk.Label(frame, text="New data point, curve no")
        self._headCurBo1 = LabelVar(frame, "0")
        self._headCurBo2 = tk.Label(frame, text="index")
        self._headCurBo3 = ComboboxVar(frame, [0, "-1", "", 1, 2], default="0", width=3)
        self._headCurBo4 = tk.Label(frame, text="x")
        self._headCurBo5 = EntryVar(frame, "", width=10)
        self._headCurBo6 = tk.Label(frame, text="y")
        self._headCurBo7 = EntryVar(frame, "", width=10)
        self._headCurBo8 = tk.Button(frame, text="Save", command=self.insert_point)
        self._headCurBo0.pack(side="left")
        self._headCurBo1.pack(side="left")
        self._headCurBo2.pack(side="left")
        self._headCurBo3.pack(side="left")
        self._headCurBo4.pack(side="left")
        self._headCurBo5.pack(side="left")
        self._headCurBo6.pack(side="left")
        self._headCurBo7.pack(side="left")
        self._headCurBo8.pack(side="left")

    def _fill_ui_curves(self):
        self._fieldsNum = 0
        for c in range(len(self.graph)):
            if self._fieldsNum + len(self.graph[c].x()) > 100000:
                print(
                    "Data editor: too many data, "
                    + (
                        "stopped display after curve " + str(c - 1)
                        if c > 0
                        else "did not display anything"
                    )
                    + "."
                )
                break
            if c >= len(self._fields["curve"]):
                self._fields["curve"].append({"data": [], "head": []})
            self._fill_ui_curve_data(c)
        # clean possibly useless fields related to nonexistent curves in memory
        while len(self._fields["curve"]) > len(self.graph):
            for column in self._fields["curve"][-1]["data"]:
                for f in column:
                    self.canvas.delete(f)
        self._fill_ui_curves_index()

    def _fill_ui_curves_index(self):
        maxlen = 0
        for c in range(len(self._fields["curve"])):
            maxlen = max(maxlen, len(self.graph[c].x()))
        if len(self._fields["idx"]) > 0:
            for f in self._fields["idx"]:
                self.ca_lef.delete(f)
        self.ca_tol.create_text(
            0, 0 * self.fieldHeight, anchor="nw", font=self.fontbold, text=" Curve"
        )
        self.ca_tol.create_text(
            0, 1 * self.fieldHeight, anchor="nw", font=self.fontbold, text=" Label"
        )
        for i in range(maxlen):
            self._fields["idx"].append(
                self.ca_lef.create_text(
                    0,
                    (i) * self.fieldHeight,
                    anchor="nw",
                    font=self.fontbold,
                    text=" " + str(i),
                )
            )
        # empty text at far right to ensure same width of different canvas
        self.canvas.create_text(
            len(self.graph) * self.curveWidth, 0, anchor="nw", text=" "
        )
        self.ca_top.create_text(
            len(self.graph) * self.curveWidth, 0, anchor="nw", text=" "
        )

    def _fill_ui_curve_data(self, c):
        # clear memory, explicitely destroy writings (to avoid memory leak)
        for column in self._fields["curve"][c]["data"]:
            for f in column:
                self.canvas.delete(f)
        for f in self._fields["curve"][c]["head"]:
            self.ca_top.delete(f)
        self._fields["curve"][c]["data"] = [[], []]
        self._fields["curve"][c]["head"] = []
        # make new display
        curve = self.graph[c]
        lbl = str(curve.attr("label"))
        if len(lbl) > 25:
            lbl = lbl[:20] + "..."
        x, y = curve.x(), curve.y()
        dx = c * self.curveWidth + 10
        self._fields["curve"][c]["dx"] = dx
        self._fields["curve"][c]["dy0"] = (0) * self.fieldHeight
        self._fields["curve"][c]["head"].append(
            self.ca_top.create_text(dx, 0, anchor="nw", font=self.fontbold, text=str(c))
        )
        self._fields["curve"][c]["head"].append(
            self.ca_top.create_text(
                dx, self.fieldHeight, anchor="nw", font=self.fontbold, text=lbl
            )
        )
        for i in range(len(x)):
            dy = (i) * self.fieldHeight
            self._fields["curve"][c]["data"][0].append(
                self.canvas.create_text(dx, dy, anchor="nw", text=str(x[i]))
            )
            self._fields["curve"][c]["data"][1].append(
                self.canvas.create_text(
                    dx + self.fieldWidth, dy, anchor="nw", text=str(y[i])
                )
            )
        # update number of created fields
        self._fieldsNum += len(x)


def build_ui():
    import grapa
    from grapa.graph import Graph

    root = tk.Tk()

    graph = Graph(r"./../examples/EQE/SAMPLE_A_d1_1.sr")
    curve = graph[0]
    from copy import deepcopy

    for i in range(6):
        graph.append(deepcopy(curve))

    app = GuiDataEditor(root, graph, None)
    app.master.title("Grapa software v" + grapa.__version__ + " Data editor")
    app.mainloop()


if __name__ == "__main__":
    build_ui()
