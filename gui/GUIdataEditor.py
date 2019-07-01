# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:43:32 2018

@author: Romain
"""

import tkinter as tk
from tkinter import ttk

import numpy as np

import sys
import os
path = os.path.abspath(os.path.join('.', '..', '..'))
if path not in sys.path:
    sys.path.append(path)

from grapa.mathModule import stringToVariable
from grapa.gui.createToolTip import CreateToolTip
from grapa.gui.GUImisc import EntryVar, CheckbuttonVar, OptionMenuVar, ComboboxVar, ButtonSmall


class GuiDataEditor(tk.Frame):
    """
    This class creates a window to edit data of a Curve (of a Graph?)
    """
    
    def __init__(self, master, graph, callback, curve_i=0):
        tk.Frame.__init__(self, master)
        self.master = master
        self.graph = graph
        self.callback = callback
        self._fields = []
        self._fieldsNum = 0
        # fill GUI
        width = min(800, max(200, self.graph.length() * 200))
        self.frame = tk.Frame(self.master, width=width)
        self.initFonts(self.frame)
        self.frame.pack(side='top', fill=tk.BOTH, expand=True)
        self.fillUIMain(self.frame)
        
        
    def initFonts(self, frame):
        import tkinter.font as font
        a = tk.Label(frame, text='')
        self.fontBold = font.Font(font=a['font'])
        self.fontBold.configure(weight='bold')
    
    def fillUIMain(self, frame):
        def on_configure(event):
            # update scrollregion after starting 'mainloop'
            # when all widgets are in canvas
            canvas.configure(scrollregion=canvas.bbox('all'))
        # --- create canvas with scrollbar ---
        canvas = tk.Canvas(frame)
        scrollbarx = tk.Scrollbar(frame, command=canvas.xview, orient=tk.HORIZONTAL)
        scrollbary = tk.Scrollbar(frame, command=canvas.yview, orient=tk.VERTICAL)
        canvas.configure(xscrollcommand = scrollbarx.set)
        canvas.configure(yscrollcommand = scrollbary.set)
        scrollbarx.pack(side=tk.BOTTOM, fill='x')
        scrollbary.pack(side=tk.RIGHT, fill='y')
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # update scrollregion after starting 'mainloop'
        # when all widgets are in canvas
        canvas.bind('<Configure>', on_configure)

        self.frameData = tk.Frame(canvas)
        canvas.create_window((0,0), window=self.frameData, anchor='nw')
#        self.frameData.pack(side='top', anchor='w')

        self.fillUIData()
        
    
    def fillUIData(self):
        frame = self.frameData
        self._fieldsNum = 0
        for c in range(self.graph.length()):
            curve = self.graph.curve(c)
            if self._fieldsNum + len(curve.x()) > 1500:
                print('Data editor: too many data, '+('stopped display after curve '+str(c-1) if c>0 else 'did not display anything')+'.')
                break
            if c >= len(self._fields):
                self._fields.append({'data':[], 'new': [], 'frame': None, 'headers':None})
                self._fields[c]['frame'] = tk.Frame(frame)
                self._fields[c]['frame'].grid(row=0, column=c, padx=10, sticky='n')
            self.fillUICurve(self._fields[c]['frame'], c, curve)
        # clean possibly useless fields related to nonexistent curves from memory
        while len(self._fields) > self.graph.length():
            for what in ['data', 'new']:
                for column in self._fields[-1]['data']:
                    for f in column:
                        if hasattr(f, 'destroy'):
                            f.destroy()
            for f in self._fields[-1]['headers']:
                if hasattr(f, 'destroy'):
                    f.destroy()
            if hasattr(self._fields[-1]['frame'], 'destroy'):
                self._fields[-1]['frame'].destroy()
            
    def fillUICurve(self, frame, c, curve):
        lbl = curve.getAttribute('label')
        if len(lbl) > 20:
            lbl = lbl[0:15]+ '...'
        if self._fields[c]['headers'] is None:
            self._fields[c]['headers'] = []
            self._fields[c]['headers'].append(tk.Label (frame, text='Curve '+str(c)+': '+lbl))
            self._fields[c]['headers'].append(tk.Button(frame, text='Save data', command=lambda x=c:self.saveDataCurve(x)))
            self._fields[c]['headers'].append(tk.Frame(frame))
            self._fields[c]['headers'][0].pack(side='top', anchor='w')
            self._fields[c]['headers'][1].pack(side='top', anchor='w')
            self._fields[c]['headers'][2].pack(side='top', anchor='w')
        self.fillUICurveData(self._fields[c]['headers'][2], c, curve)
        
    def fillUICurveData(self, frame, c, curve):
        # clear memory (to avoid memeory leak). Not optimal, should try to reuse existing fields
        if len(self._fields[c]['data']) > 0:
            for column in self._fields[c]['data']:
                for f in column:
                    if hasattr(f, 'destroy'):
                        f.destroy()
        if len(self._fields[c]['new']) > 0: # want to erase these fields anyways, lenght (thus position) may have changed
            for column in self._fields[c]['new']:
                for f in column:
                    if hasattr(f, 'destroy'):
                        f.destroy()
        # make new display
        # new value at index 0
        self._fields[c]['new'] = [[], []]
        self._fields[c]['new'][0].append(EntryVar(frame, '', width=10))
        self._fields[c]['new'][1].append(EntryVar(frame, '', width=10))
        self._fields[c]['new'][0][-1].grid(row=0, column=0)
        self._fields[c]['new'][1][-1].grid(row=0, column=1)
        # data
        self._fields[c]['data'] = [[], [], []]
        x, y = curve.x(), curve.y()
        for i in range(len(x)):
            self._fields[c]['data'][0].append(EntryVar(frame, str(x[i]), width=10))
            self._fields[c]['data'][0][i].grid(row=i+2, column=0)
        for i in range(len(x)):
            self._fields[c]['data'][1].append(EntryVar(frame, str(y[i]), width=10))
            self._fields[c]['data'][1][i].grid(row=i+2, column=1)
        for i in range(len(x)):
            self._fields[c]['data'][2].append(ButtonSmall(frame, text='X', command=lambda x=c, y=i: self.deleteXYPair(x, y)))
            self._fields[c]['data'][2][i].grid(row=i+2, column=2)
        # new value at last index
        self._fields[c]['new'][0].append(EntryVar(frame, '', width=10))
        self._fields[c]['new'][1].append(EntryVar(frame, '', width=10))
        self._fields[c]['new'][0][-1].grid(row=len(x)+3, column=0)
        self._fields[c]['new'][1][-1].grid(row=len(x)+3, column=1)
        self._fields[c]['new'][0].append(tk.Frame(frame, height=8, width=8))
        self._fields[c]['new'][0].append(tk.Frame(frame, height=8, width=8))
        self._fields[c]['new'][0][2].grid(row=1, column=0)
        self._fields[c]['new'][0][3].grid(row=len(x)+2, column=1)
        # update number of created fields
        self._fieldsNum += len(x)

        
    def deleteXYPair(self, c, i):
        curve = self.graph.curve(c)
        data = np.delete(curve.getData(), i, axis=1)
        curve.data = data
        if self.callback is not None:
            self.callback()
        self.fillUIData()
        
    def saveDataCurve(self, c):
        x = [float(f.get()) for f in self._fields[c]['data'][0]]
        y = [float(f.get()) for f in self._fields[c]['data'][1]]
        x_, y_ = self._fields[c]['new'][0][0].get(), self._fields[c]['new'][1][0].get()
        if x_ is not '' and y_ is not '':
            x = [float(x_)] + x
            y = [float(y_)] + y
        x_, y_ = self._fields[c]['new'][0][1].get(), self._fields[c]['new'][1][1].get()
        if x_ is not '' and y_ is not '':
            x = x + [float(x_)]
            y = y + [float(y_)]            
        self.graph.curve(c).data = np.array([x, y])
        if self.callback is not None:
            self.callback()
        self.fillUIData()
        
        
def buildUI():
    import grapa
    from grapa.graph import Graph
    root = tk.Tk()
    graph = Graph(r'./../examples/EQE/SAMPLE_A_d1_1.sr')
    graph.duplicateCurve(0)
    graph.duplicateCurve(0)
    graph.duplicateCurve(0)
    graph.duplicateCurve(0)
    graph.duplicateCurve(0)
    graph.duplicateCurve(0)
    app = GuiDataEditor(root, graph, None)
    app.master.title('Grapa software v'+grapa.__version__+' Data editor')
    app.mainloop()

    
if __name__ == "__main__":
    buildUI()
    
    
    
    
    