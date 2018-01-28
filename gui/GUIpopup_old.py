# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:18:07 2017

@author: Romain
"""
import tkinter as tk
from copy import deepcopy

from grapa.mathModule import stringToVariable
from grapa.gui.createToolTip import CreateToolTip


class GuiManagerAnnotations:
    """
    Provides a popup window for quick and easier modification of text
    annotations, legends, legend titles and graph titles.
    """
    # Implementation started with a not-great code for textx, then a more
    # tunable solution was designed for the rest. Previous not modified as it
    # was working
    
    def __init__(self, master, graph, funcupdate):
        self.master = master
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
    
    def configLegTit(self):
        self.legtitfields = []
        self.legtitfields.append({'label': 'Legend title','width': 0})
        self.legtitfields.append({'label': 'title',       'width': 15,  'tooltip': 'Enter a title to the legend'})
        self.legtitfields.append({'label': 'color',       'width': 9,   'fromkwargs': 'color',    'tooltip':'Color. Example: "r", "[1,0,1]", or "pink"'})
        self.legtitfields.append({'label': 'fontsize',    'width': 7,   'fromkwargs': 'fontsize', 'tooltip':'Numeric, eg. "20"'})
        self.legtitfields.append({'label': 'align', 'field': 'OptionMenu', 'fromkwargs': 'align',    'tooltip':'Horizontal alignment of title. eg. "left", "right".\nFor legendtitle: not a regular matplotlib keyword.', 'values': ['left', 'center', 'right']})
        self.legtitfields.append({'label': 'position (x,y)','width': 12,'fromkwargs': 'position', 'tooltip':'legendtitle: (x,y) position shift in pixels, eg. "(20,0)"\ntitle: (x,y) relative position, eg. "(0,1)"'})
        self.legtitfields.append({'label':'other properties','width': 35,'tooltip':'kwargs to legend.set_title()'})
        self.legtitvars = []
        for field in self.legtitfields:
            self.legtitvars.append([])
    def configLegend(self):
        self.legendfields = []
        self.legendfields.append({'label': 'Legend',    'width': 0})
        self.legendfields.append({'label': 'loc',  'field': 'OptionMenu', 'tooltip':'Location, eg. "nw" (north west), "se", "center"', 'fromkwargs': 'loc', 'default': 'best', 'values': ['best', 'nw', 'n', 'ne', 'w', 'center', 'e', 'sw', 's', 'se']})
        self.legendfields.append({'label': 'color',     'width': 9, 'tooltip':'Color. Example: "r", "[1,0,1]", or "pink"\n"curve" will give same color as trace', 'fromkwargs': 'color'})
        self.legendfields.append({'label': 'fontsize',  'width': 7, 'tooltip':'Numeric, eg. "20"', 'fromkwargs': 'fontsize'})
        self.legendfields.append({'label': 'ncol',      'width': 6, 'tooltip':'Number of columns, eg. "2"', 'fromkwargs': 'ncol', 'casttype': int})
        self.legendfields.append({'label': 'bbox_to_anchor','width': 14, 'tooltip':'(x,y) shift to base position in axes fraction, eg. "(0.1, 0.2)"', 'fromkwargs': 'bbox_to_anchor'})
        self.legendfields.append({'label':'other properties','width': 35,'tooltip':'kwargs to ax.legend(), see https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.legend.html'})
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
        for line in self.annotations:
            for widget in line:
                widget.destroy()
        self.annotations = []
        self.annotatvars = []
        text = self.attr['text']
        texy = self.attr['textxy']
        targ = self.attr['textargs']
        ttco = ['' for t in text]
        # headers
        tk.Label(frame, text='').grid(row=0, column=0)
        Labels = []
        Labels.append(tk.Label(frame, text='visible'))
        Labels.append(tk.Label(frame, text='text'))
        Labels.append(tk.Label(frame, text='xytext'))
        Labels.append(tk.Label(frame, text='textcoords'))
        Labels.append(tk.Label(frame, text='color'))
        Labels.append(tk.Label(frame, text='fontsize'))
        Labels.append(tk.Label(frame, text='other properties'))
        for j in range(len(Labels)):
            Labels[j].grid(row=0, column=j)
        CreateToolTip(Labels[0], "Visible?")
        CreateToolTip(Labels[1], "Enter a text")
        CreateToolTip(Labels[2], 'Position of the text. Example: "(0.2,0.3)"')
        CreateToolTip(Labels[3], 'By default "figure fraction". Can also be "axes fraction", "figure pixels", "data", etc.')
        CreateToolTip(Labels[4], 'Color. Example: "r", "[1,0,1]", or "pink"')
        CreateToolTip(Labels[5], 'Fontsize. Example: "16"')
        CreateToolTip(Labels[6], "kwargs to ax.annotate. Example: \"{'verticalalignment':'center', 'xy':(0.5,0.5), 'xycoords': 'figure fraction', 'rotation': 90, 'arrowprops':{'facecolor':'blue', 'shrink':0.05}}\"\nSee https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.annotate.html", wraplength=300)
        # content
        for i in range(len(text)):
            if texy[i] == '' and 'xytext' in targ[i]:
                texy[i] = targ[i]['xytext']
                del targ[i]['xytext']
            if 'textcoords' in targ[i]:
                ttco[i] = targ[i]['textcoords']
                del targ[i]['textcoords']
            visible = True
            if 'visible' in targ[i]:
                visible = int(targ[i]['visible'])
                del targ[i]['visible']
            color = ''
            if 'color' in targ[i]:
                color = targ[i]['color']
                del targ[i]['color']
            fontsize = ''
            if 'fontsize' in targ[i]:
                fontsize = targ[i]['fontsize']
                del targ[i]['fontsize']
            self.annotations.append([])
            self.annotatvars.append([])
            self.annotatvars[-1].append(tk.IntVar())
            self.annotatvars[-1][-1].set(visible)
            self.annotations[-1].append(tk.Checkbutton(frame, variable=self.annotatvars[-1][-1]))
            self.annotations[-1].append(tk.Entry(frame, width=15))
            self.annotations[-1].append(tk.Entry(frame, width=12))
            self.annotations[-1].append(tk.Entry(frame, width=12)
            self.annotations[-1].append(tk.Entry(frame, width=9))
            self.annotations[-1].append(tk.Entry(frame, width=7))
            self.annotations[-1].append(tk.Entry(frame, width=35))
            self.annotations[-1][1].insert(0, str(text[i]).replace('\n','\\n'))
            self.annotations[-1][2].insert(0, str(texy[i]))
            self.annotations[-1][3].insert(0, str(ttco[i]))
            self.annotations[-1][4].insert(0, str(color))
            self.annotations[-1][5].insert(0, str(fontsize))
            self.annotations[-1][6].insert(0, str(targ[i]))
            for j in range(len(self.annotations[-1])):
                self.annotations[-1][j].grid(row=(i+1), column=j)
        # new annotation
        self.annotations.append([])
        self.annotatvars.append([])
        self.annotatvars[-1].append(tk.IntVar())
        self.annotatvars[-1][-1].set(1)
        self.annotations[-1].append(tk.Checkbutton(frame, variable=self.annotatvars[-1][-1]))
        self.annotations[-1].append(tk.Entry(frame, width=15))
        self.annotations[-1].append(tk.Entry(frame, width=12))
        self.annotations[-1].append(tk.Entry(frame, width=12))
        self.annotations[-1].append(tk.Entry(frame, width=9))
        self.annotations[-1].append(tk.Entry(frame, width=7))
        self.annotations[-1].append(tk.Entry(frame, width=35))
        self.annotations[-1][0].select()
        self.annotations[-1][6].insert(0, '{}')
        i = len(self.annotations)
        for j in range(len(self.annotations[-1])):
            self.annotations[-1][j].grid(row=i, column=j)
    
    def fillUILegendGrid(self, frame):
        Labels = []
        for j in range(len(self.legendfields)):
            field = self.legendfields[j]
            if 'width' not in field:
                field['width'] = -1
            Labels.append(tk.Label(frame, text=field['label']))
            if field['width'] == 0:
                Labels[-1].configure(font=self.fontBold)
            Labels[j].grid(row=(0 if field['width']!=0 else 1), column=j)
            if 'tooltip' in field:
                CreateToolTip(Labels[j], field['tooltip'])
            self.legendvars[j].append(tk.StringVar())
            if 'field' not in field:
                field['field'] = 'Entry'
            e = None
            if field['field'] == 'OptionMenu':
                e = tk.OptionMenu(frame, self.legendvars[j][-1], *field['values'])
            elif field['width'] > 0:
                e = tk.Entry(frame, width=field['width'], textvariable=self.legendvars[j][-1])
            if e is not None:
                e.grid(column=j, row=len(self.legendvars[j]))
        self.legendTitleFillValues('legendproperties', self.legendfields, self.legendvars)
    def fillUILegendTitleGrid(self, frame):
        Labels = []
        # legend title
        for j in range(len(self.legtitfields)):
            field = self.legtitfields[j]
            if 'width' not in field:
                field['width'] = -1
            if field['width'] != 0:
                Labels.append(tk.Label(frame, text=field['label']))
                Labels[-1].grid(row=0, column=j)
            if 'tooltip' in field:
                CreateToolTip(Labels[-1], field['tooltip'])
            self.legtitvars[j].append(tk.StringVar())
            if 'field' not in field:
                field['field'] = 'Entry'
            e = None
            if field['field'] == 'OptionMenu':
                e = tk.OptionMenu(frame, self.legtitvars[j][-1], *field['values'])
            elif field['width'] > 0:
                e = tk.Entry(frame, width=field['width'], textvariable=self.legtitvars[j][-1])
            if e is not None:
                e.grid(column=j, row=len(self.legtitvars[j]), ipady=0, pady=0)
        # graph title
        for j in range(len(self.legtitfields)):
            field = self.legtitfields[j]
            # graph title
            self.legtitvars[j].append(tk.StringVar())
            if 'field' not in field:
                field['field'] = 'Entry'
            e = None
            if field['field'] == 'OptionMenu':
                e = tk.OptionMenu(frame, self.legtitvars[j][-1], *field['values'])
            elif field['width'] > 0:
                e = tk.Entry(frame, width=field['width'], textvariable=self.legtitvars[j][-1])
            if e is not None:
                e.grid(column=j, row=len(self.legtitvars[j]), ipady=0, pady=0)
        tk.Label(frame, text='Legend title', font=self.fontBold).grid(row=1, column=0)
        tk.Label(frame, text='Graph title',  font=self.fontBold).grid(row=2, column=0)
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
                tvis = self.annotatvars[i][0].get()
                targ = stringToVariable(ann[6].get())
                texy = stringToVariable(ann[2].get())
                colo = stringToVariable(ann[4].get())
                fsiz = stringToVariable(ann[5].get())
                textcoords = ann[3].get()
                if not isinstance(targ, dict):
                    if targ is not '':
                        print('GuiManagerAnnotations invalid input:', targ, 'should be a dict')
                    targ = {}
                if isinstance(texy, (list, tuple)) and 'xytext' not in targ:
                    targ.update({'xytext': texy})
                    texy = ''
                if textcoords != '':
                    targ.update({'textcoords': textcoords})
                if not tvis and 'visible' not in targ:
                    targ.update({'visible': tvis})
                if colo != '' and 'color' not in targ:
                    targ.update({'color': colo})
                if fsiz != '' and 'fontsize' not in targ:
                    targ.update({'fontsize': fsiz})
                out['textxy'].append(texy)
                out['textargs'].append(targ)
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
    

        