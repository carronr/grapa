# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:18:07 2017

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import tkinter as tk
from tkinter import ttk
from copy import deepcopy

import sys
import os
path = os.path.abspath(os.path.join('.', '..', '..'))
if path not in sys.path:
    sys.path.append(path)

from grapa.mathModule import stringToVariable
from grapa.gui.createToolTip import CreateToolTip
from grapa.gui.GUImisc import EntryVar, CheckbuttonVar, OptionMenuVar, ComboboxVar


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
        self.oldattr = {}
        self.oldattr['text']     = deepcopy(graph.getAttribute('text'))
        self.oldattr['textxy']   = deepcopy(graph.getAttribute('textxy', ['']))
        self.oldattr['textargs'] = deepcopy(graph.getAttribute('textargs', [{}]))
        self.oldattr['legendtitle'] = deepcopy(graph.getAttribute('legendtitle'))
        # prepare
        self.annotations = []
        self.configText()
        self.configLegTit()
        self.configLegend()
        # fill GUI
        self.frame = tk.Frame(self.master)
        self.initFonts(self.frame)
        
        self.fillUIMain(self.frame)
        self.frame.pack(side='top')
        self.master.bind("<Return>", lambda event: self.go())

    def initFonts(self, frame):
        import tkinter.font as font
        a = tk.Label(frame, text='')
        self.fontBold = font.Font(font=a['font'])
        self.fontBold.configure(weight='bold')
    

    def configText(self):
        self.textfields = []
        self.textfields.append({'label': 'visible','field': 'Checkbutton', 'fromkwargs': 'visible', 'tooltip': 'Visible?', 'default': True})
        self.textfields.append({'label': 'text',        'width': 15,  'tooltip': 'Enter a text'})
        self.textfields.append({'label': 'color',       'width': 7,  'fromkwargs': 'color',     'tooltip':'Color. Example: "r", "[1,0,1]", or "pink"', 'field': 'Combobox', 'values': ['r', 'pink', '[1,0,1]']})
        self.textfields.append({'label': 'fontsize',    'width': 7,  'fromkwargs': 'fontsize',  'tooltip':'Fontsize. Example: "16"'})
        self.textfields.append({'label': 'xytext',      'width': 12, 'fromkwargs': 'xytext',    'tooltip':'Position of the text. Example: "(0.2,0.3)"'})
        self.textfields.append({'label': 'textcoords', 'field': 'OptionMenu', 'fromkwargs': 'textcoords','tooltip':'By default "figure fraction". Can also be "axes fraction", "figure pixels", "data", etc.', 'values': ['figure fraction', 'figure pixels', 'figure points', 'axes fraction', 'axes pixels', 'axes points', 'data']})
        self.textfields.append({'label':'other properties','width': 38,
                                'tooltip':"kwargs to ax.annotate. Example: \"{'verticalalignment':'center', 'xy':(0.5,0.5), 'xycoords': 'figure fraction', 'rotation': 90, 'arrowprops':{'facecolor':'blue', 'shrink':0.05}}\"\nSee https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.annotate.html",
                                'default':{}, 'field': 'Combobox',
                                'values': ["{}",
                                           "{'verticalalignment':'center'}",
                                           "{'xy':(0.5,0.5), 'xycoords': 'figure fraction', 'arrowprops':{'facecolor':'b', 'shrink':0.05}}",
                                           "{'rotation': 90}"]})
        for field in self.textfields:
            if 'field' not in field:
                field['field'] = 'Entry'
            if 'width' not in field:
                field['width'] = -1
            if 'fromkwargs' not in field:
                field['fromkwargs'] = ''
        self.textvars = []
        for field in self.textfields:
            self.textvars.append([])
    def configLegTit(self):
        self.legtitfields = []
        self.legtitfields.append({'label': 'Legend title','width': 0})
        self.legtitfields.append({'label': 'title',       'width': 15,  'tooltip': 'Enter a title to the legend'})
        self.legtitfields.append({'label': 'color',       'width': 7,   'fromkwargs': 'color',    'tooltip':'Color. Example: "r", "[1,0,1]", or "pink"', 'field': 'Combobox', 'values': ['r', 'pink', '[1,0,1]']})
        self.legtitfields.append({'label': 'fontsize',    'width': 7,   'fromkwargs': 'fontsize', 'tooltip':'Numeric, eg. "20"'})
        self.legtitfields.append({'label': 'align', 'field': 'OptionMenu', 'fromkwargs': 'align',    'tooltip':'Horizontal alignment of title. eg. "left", "right".\nFor legendtitle: not a regular matplotlib keyword.', 'values': ['left', 'center', 'right']})
        self.legtitfields.append({'label': 'position (x,y)','width': 10,'fromkwargs': 'position',
                                  'tooltip':'legendtitle: (x,y) position shift in pixels, eg. "(20,0)"\ntitle: (x,y) relative position, eg. "(0,1)"', 'field': 'Combobox', 'values': ['(10,0)', '(-10,20)']})
        self.legtitfields.append({'label':'other properties','width': 40,'tooltip':'kwargs to legend.set_title()'})
        for field in self.legtitfields:
            if 'field' not in field:
                field['field'] = 'Entry'
            if 'width' not in field:
                field['width'] = -1
        self.legtitvars = []
        for field in self.legtitfields:
            self.legtitvars.append([])
    def configLegend(self):
        self.legendfields = []
        self.legendfields.append({'label': 'Legend',    'width': 0})
        self.legendfields.append({'label': 'color',     'width': 7, 'tooltip':'Color. Example: "r", "[1,0,1]", or "pink"\n"curve" will give same color as trace', 'fromkwargs': 'color', 'field': 'Combobox', 'values': ['r', 'pink', '[1,0,1]', 'curve']})
        self.legendfields.append({'label': 'fontsize',  'width': 7, 'tooltip':'Numeric, eg. "20"', 'fromkwargs': 'fontsize'})
        self.legendfields.append({'label': 'loc',  'field': 'OptionMenu', 'tooltip':'Location, eg. "nw" (north west), "se", "center"', 'fromkwargs': 'loc', 'default': 'best', 'values': ['best', 'nw', 'n', 'ne', 'w', 'center', 'e', 'sw', 's', 'se']})
        self.legendfields.append({'label': 'ncol',      'width': 6, 'tooltip':'Number of columns, eg. "2"', 'fromkwargs': 'ncol', 'casttype': int})
        self.legendfields.append({'label': 'bbox_to_anchor','width': 12, 'tooltip':'(x,y) shift to base position in axes fraction, eg. "(0.1, 0.2)"', 'fromkwargs': 'bbox_to_anchor', 'field': 'Combobox', 'values': ['(0,0.1)', '(0.1,0.2)']})
        self.legendfields.append({'label':'other properties','width': 40,
                                  'tooltip':'kwargs to ax.legend(), see https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.legend.html',
                                  'field': 'Combobox',
                                  'values': ["{}", "{'frameon': True, 'framealpha':1}", "{'framealpha': 1}", "{'numpoints': 2}"]})
        for field in self.legendfields:
            if 'field' not in field:
                field['field'] = 'Entry'
            if 'width' not in field:
                field['width'] = -1
        self.legendvars = []
        for field in self.legendfields:
            self.legendvars.append([])
        
    
    def parseGraph(self):
        # retrieve data from Graph object, ensure formatting is ok
        self.attr = {}
        self.attr['text']     = deepcopy(self.graph.getAttribute('text'))
        self.attr['textxy']   = deepcopy(self.graph.getAttribute('textxy', ['']))
        self.attr['textargs'] = deepcopy(self.graph.getAttribute('textargs', [{}]))
        if not isinstance(self.attr['text'], list):
            self.attr['text'] = [self.attr['text']]
        if not isinstance(self.attr['textxy'], list):
            self.attr['textxy'] = [self.attr['textxy']]
        if not isinstance(self.attr['textargs'], list):
            self.attr['textargs'] = [self.attr['textargs']]
        while len(self.attr['textxy']) < len(self.attr['text']):
            self.attr['textxy'].append('')
        while len(self.attr['textargs']) < len(self.attr['text']):
            self.attr['textargs'].append({})
        # override xytext in textargs with content of textxy - later can ignore textxy
        for i in range(len(self.attr['textxy'])):
            if self.attr['textxy'][i] != '':
                self.attr['textargs'].update({'xytext': self.attr['textxy'][i]})
        if self.attr['text'] == ['']:
            self.attr['text'] = []
        
    def fillUIMain(self, frame):
        self.frameGrid = tk.Frame(frame)
        self.frameGrid.pack(side='top', anchor='w')
        self.fillUITextGrid(self.frameGrid)
        
        self.legendGrid = tk.Frame(frame)
        self.legendGrid.pack(side='top', anchor='w')
        self.fillUILegendGrid(self.legendGrid)
        
        self.legTitGrid = tk.Frame(frame)
        self.legTitGrid.pack(side='top', anchor='w')
        self.fillUILegendTitleGrid(self.legTitGrid)
        
        frm = tk.Frame(frame)
        frm.pack()
        self.fillUIButtons(frm)
    
    def fillUIButtons(self, frame):
        self.ButtonRevert = tk.Button(frame, text='Revert to initial', fg='grey', command=self.revert)
        self.ButtonRevert.pack(side='left')
        self.ButtonQuit = tk.Button(frame, text='Quit', fg='grey', command=self.close_windows)
        self.ButtonQuit.pack(side='left', padx=10)
        self.ButtonGo = tk.Button(frame, text='Update graph', command=self.go, font=self.fontBold)
        self.ButtonGo.pack(side='right', padx=30)
    
    def fillUITextGrid(self, frame):
        self.parseGraph()
        # remove previsously created widgets, start afresh
        for row in self.annotations:
            for field in row:
                if hasattr(field, 'destroy'):
                    field.destroy()
        self.annotations = []
        args = []
        for j in range(len(self.attr['text'])):
            self.annotations.append([''] * len(self.textfields))
            args.append(deepcopy(self.attr['textargs'][j]))
        self.annotations.append([''] * len(self.textfields)) # for new text
        args.append({}) # defautl new
        for j in range(len(self.textfields)):
            field = self.textfields[j]
            e = tk.Label(frame, text=field['label'])
            e.grid(row=0, column=(j+1))
            if 'tooltip' in field:
                CreateToolTip(e, field['tooltip'])
            # data fields and new text fields
            kw = field['fromkwargs']
            for i in range(len(self.attr['text'])+1):
                value = ''
                if i == len(self.attr['text']): # "new" text field
                    value = ''
                elif field['label'] == 'text': # text: take it from attr['text']
                    value = self.attr['text'][i].replace('\n', '\\n')
                else: # otherwise must grab it from the textargs dict
                    if kw != '' and kw in args[i]:
                        value = args[i][kw]
                        del args[i][kw]
                    elif j == len(self.textfields) - 1:
                        value = str(args[i])
                if value == '' and 'default' in field:
                    value = field['default']
                e = None
                if field['field'] == 'OptionMenu':
                    e = OptionMenuVar(frame, values=field['values'], default=str(value))
                elif field['field'] == 'Checkbutton':
                    e = CheckbuttonVar(frame, '', default=(bool(value) if value != '' else False))
                if field['field'] == 'Combobox':
                    e = ComboboxVar(frame, field['values'], default=str(value), width=field['width'])
                elif field['width'] > 0:
                    e = EntryVar(frame, str(value), width=field['width'])
                if e is not None:
                    self.annotations[i][j] = e
                    e.grid(column=(j+1), row=(i+1), pady=0, ipady=0, padx=1)
        tk.Label(frame, text='Text', font=self.fontBold).grid(row=0, column=0)
        tk.Label(frame, text='New',  font=self.fontBold).grid(row=(len(self.attr['text'])+1), column=0)

    def fillUILegendGrid(self, frame):
        Labels = []
        for j in range(len(self.legendfields)):
            field = self.legendfields[j]
            Labels.append(tk.Label(frame, text=field['label']))
            if field['width'] == 0:
                Labels[-1].configure(font=self.fontBold)
            Labels[j].grid(row=(0 if field['width']!=0 else 1), column=j)
            if 'tooltip' in field:
                CreateToolTip(Labels[j], field['tooltip'])
            self.legendvars[j].append(tk.StringVar())
            e = None
            if field['field'] == 'OptionMenu':
                e = tk.OptionMenu(frame, self.legendvars[j][-1], *field['values'])
            if field['field'] == 'Combobox':
                e = ttk.Combobox(frame, textvariable=self.legendvars[j][-1], values=field['values'], width=field['width'])
            elif field['width'] > 0:
                e = tk.Entry(frame, width=field['width'], textvariable=self.legendvars[j][-1])
            if e is not None:
                e.grid(column=j, row=len(self.legendvars[j]), padx=1)
        self.legendTitleFillValues('legendproperties', self.legendfields, self.legendvars)
    def fillUILegendTitleGrid(self, frame):
        Labels = []
        # legend title
        for j in range(len(self.legtitfields)):
            field = self.legtitfields[j]
            if field['width'] != 0:
                Labels.append(tk.Label(frame, text=field['label']))
                Labels[-1].grid(row=0, column=j)
            if 'tooltip' in field:
                CreateToolTip(Labels[-1], field['tooltip'])
            self.legtitvars[j].append(tk.StringVar())
            e = None
            if field['field'] == 'OptionMenu':
                e = tk.OptionMenu(frame, self.legtitvars[j][-1], *field['values'])
            if field['field'] == 'Combobox':
                e = ttk.Combobox(frame, textvariable=self.legtitvars[j][-1], values=field['values'], width=field['width'])
            elif field['width'] > 0:
                e = tk.Entry(frame, width=field['width'], textvariable=self.legtitvars[j][-1])
            if e is not None:
                e.grid(column=j, row=len(self.legtitvars[j]), ipady=0, pady=0, padx=1)
        # graph title
        self.legtitfields[5].update({'values': ["(0,1)", "(0.5, 0.95)"]})
        for j in range(len(self.legtitfields)):
            field = self.legtitfields[j]
            # graph title
            self.legtitvars[j].append(tk.StringVar())
            if 'field' not in field:
                field['field'] = 'Entry'
            e = None
            if field['field'] == 'OptionMenu':
                e = tk.OptionMenu(frame, self.legtitvars[j][-1], *field['values'])
            if field['field'] == 'Combobox':
                e = ttk.Combobox(frame, textvariable=self.legtitvars[j][-1], values=field['values'], width=field['width'])
            elif field['width'] > 0:
                e = tk.Entry(frame, width=field['width'], textvariable=self.legtitvars[j][-1])
            if e is not None:
                e.grid(column=j, row=len(self.legtitvars[j]), ipady=0, pady=0, padx=1)
        tk.Label(frame, text='Legend title', font=self.fontBold).grid(row=1, column=0)
        tk.Label(frame, text='Graph title',  font=self.fontBold).grid(row=2, column=0, sticky='W')
        self.legendTitleFillValues('legendtitle', self.legtitfields, self.legtitvars, row=0)
        self.legendTitleFillValues('title', self.legtitfields, self.legtitvars, row=1)
            
        
    def legendTitleFillValues(self, attribute, fields, vars_, row=0):
        attr = deepcopy(self.graph.getAttribute(attribute))
        vals = [''] * len(fields)
        vals[-1] = attr
        if attribute in ['title', 'legendtitle']:
            if not isinstance(attr, list):
                attr = [attr, {}]
            vals[1] = attr[0]
            vals[-1] = attr[1]
        elif attribute in ['legendproperties']:
            if vals[-1] == '':
                vals[-1] = {}
        for i in range(len(fields)):
            if 'fromkwargs' in fields[i]:
                key = fields[i]['fromkwargs']
                if attribute == 'title' and key == 'align':
                    key = 'loc' # different keyword for same stuff
                if key in vals[-1]:
                    vals[i] = str(vals[-1][key])
                    del vals[-1][key]
            if vals[i] == '' and 'default' in fields[i]:
                vals[i] = fields[i]['default']
        vals[1] = vals[1].replace('\n','\\n')
        for j in range(0, len(vars_)):
            vars_[j][row].set(vals[j])

    
    def go(self):
        # text annotations
        out = {'text': [], 'textxy': [], 'textargs': []}
        i = 0
        for ann in self.annotations:
            if ann[1].get() is not '':
                out['text'].append(ann[1].get().replace('\\n', '\n'))
                args = stringToVariable(ann[-1].get())
                if not isinstance(args, dict):
                    print('GuiManagerAnnotations invalid input:', args, 'should be a dict')
                    args = {}
                for j in range(len(self.textfields)-1):
                    if self.textfields[j]['fromkwargs'] != '':
                        val = stringToVariable(ann[j].get())
                        if self.textfields[j]['label'] == 'xytext':
                            if not isinstance(val, (list, tuple)):
                                print('GuiManagerAnnotations invalid input: (xytext)', val, 'should be a tuple or dict with 2 elements (coordinates)')
                        if self.textfields[j]['label'] == 'visible?' and val:
                            val = ''
                        if val != '':
                            args.update({self.textfields[j]['fromkwargs']: val})
                out['textxy'].append('')
                out['textargs'].append(args)
            i += 1
        # legend title, graph title
        keywords = ['legendtitle', 'title']
        for j in range(len(keywords)):
            tit = self.legtitvars[1][j].get().replace('\\n','\n')
            kw = stringToVariable(self.legtitvars[-1][j].get())
            if not isinstance(kw, dict):
                print('GuiManagerAnnotations invalid input:', kw, 'should be a dict (legendtitle)')
                kw = {}
            for i in range(len(self.legtitfields)):
                if 'fromkwargs' in self.legtitfields[i]:
                    if self.legtitvars[i][j].get() != '':
                        val = stringToVariable(self.legtitvars[i][j].get())
                        if 'casttype' in self.legtitfields[i]:
                            val = self.legtitfields[i]['casttype'](val)
                        kw.update({self.legtitfields[i]['fromkwargs']: val})
            if keywords[j] == 'title' and 'align' in kw:
                kw['loc'] = kw['align']
                del kw['align']
            legtit = tit if kw in ['', {}] else [tit, kw]
            out.update({keywords[j]: legtit})
        # legend properties
        kw = stringToVariable(self.legendvars[-1][0].get())
        if not isinstance(kw, dict):
            print('GuiManagerAnnotations invalid input:', kw, 'should be a dict (legend)')
            kw = {}
        for i in range(len(self.legendfields)):
            if 'fromkwargs' in self.legendfields[i]:
                if self.legendvars[i][0].get() != '':
                    val = stringToVariable(self.legendvars[i][0].get())
                    if 'casttype' in self.legendfields[i]:
                        val = self.legendfields[i]['casttype'](val)
                    kw.update({self.legendfields[i]['fromkwargs']: val})
        out.update({'legendproperties': kw})
        # graph title
        
        # perform update, finish
        #print('sent to GUI', out)
        self.funcupdate(out)
        self.refreshUI()
    def revert(self):
        self.funcupdate(self.oldattr)
        self.refreshUI()
    def refreshUI(self):
        self.fillUITextGrid(self.frameGrid)
        self.legendTitleFillValues('legendtitle', self.legtitfields, self.legtitvars)
    def close_windows(self):
        self.master.destroy()
    

    
        
        
def buildUI():
    from grapa.graph import Graph
    root = tk.Tk()
    app = GuiManagerAnnotations(root, Graph(r'.\..\examples\fancyAnnotations.txt'), None)
    app.mainloop()

if __name__ == "__main__":
    buildUI()