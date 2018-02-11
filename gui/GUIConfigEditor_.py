# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:35:58 2018

@author: Romain
"""

import tk

import sys
import os
path = os.path.abspath(os.path.join('.', '..', '..'))
if path not in sys.path:
    sys.path.append(path)

from grapa.mathModule import stringToVariable
from grapa.gui.createToolTip import CreateToolTip
from grapa.gui.GUImisc import EntryVar, CheckbuttonVar, OptionMenuVar, ComboboxVar


class GuiConfigEditor(tk.Frame):
    """
    This class creates a window to edit the configuration file.
    """
    
    def __init__(self, master, fileConfig):
        self.fileConfig = fileConfig
        # fill GUI
        self.frame = tk.Frame(self.master)
        self.initFonts(self.frame)        
        self.initKeywords()
        self.fillUIMain(self.frame)
        self.frame.pack(side='top')
        
        
    def initFonts(self, frame):
        import tkinter.font as font
        a = tk.Label(frame, text='')
        self.fontBold = font.Font(font=a['font'])
        self.fontBold.configure(weight='bold')
    
    def fillUIMain(self, frame):
        self.frameGrid = tk.Frame(frame)
        self.frameGrid.pack(side='top', anchor='w')
        self.fillUIKeywords(self.frameGrid)
        self.frameActions = tk.Frame(frame)
        self.frameActions.pack(side='top', anchor='w')
        self.fillUIActions(self.frameGrid)
        
    def fillUIActions(self, frame):
        tk.Button(frame, text='reload', command=self.loadFromFile).pack(side='left')
        tk.Button(frame, text='save', command=self.saveConfig).pack(side='left')
    
    def saveConfig(self):
        print('saveConfig: TO IMPLEMENT')
    def loadFromFile(self):
        
        
    def fillUIKeywords(self, frame):
        for field in self.keywords:
            pass
        
        
        
        
        
    def initKeywords(self):
        self.keywords = []
        
        # graph labels default unit presentation: [unit], (unit), / unit, or [], (), /
        self.keyword.append(['graph_labels_units',
                             {'field': 'Combobox',
                             'values': ['[unit]', '(unit)', '/ unit']}])
        # graph labels presence of symbols (ie: $C$ in 'Capacitance $C$ [nF]')
        self.keywords.append(['graph_labels_symbols',
                              {'field': 'Combobox',
                              'values': ['False', 'True']}])
        # path to inkscape executable, to export in .emf image format. Can be a string, or a list of strings
        self.keywords.append(['inkscape_path',
                              {'field': 'openfilename'}])
        # GUI default colorscales. Each colorscale is represented as a string (matplotlib colorscales), or a list of colors.
        # Each color can be a [r,g,b] triplet, a [r,g,b,a] quadruplet, or a [h,l,s,'hls'] quadruplet. rgb, rgba, hls, hsv colorscape are supported.
        self.keywords.append(['GUI_colorscale',
                              {'multiple': True})
#             GUI_colorscale00	[[1,0,0], [1,1,0.5], [0,1,0]]
#             GUI_colorscale01	[[1,0.3,0], [0.7,0,0.7], [0,0.3,1]]
#             GUI_colorscale02	[[1,0.43,0], [0,0,1]]
#             GUI_colorscale03	[[0.91,0.25,1], [1.09,0.75,1], 'hls']
#             GUI_colorscale04	[[0.70,0.25,1], [0.50,0.75,1], 'hls']
#             GUI_colorscale05	[[1,0,1], [1,0.5,1], [0.5,0.75,1], 'hls']
#             GUI_colorscale07	'inferno'
#             GUI_colorscale10	'gnuplot2'
#             GUI_colorscale11	'viridis'
#             GUI_colorscale12	'afmhot'
#             GUI_colorscale13	'YlGn'
    # default saving image format
    self.keywords.append(['save_imgformat',
                          {'field': 'combobox',
                           'values': ['.png', '.tif', '.pdf', 'svg', '.emf']}])
                           # TODO: all possibilities
            
    # some checks
    for vals in self.keywords:
        if 'field' in vals[1]:
            vals[1]['field'] = vals[1].fields.lower()
        else:
            vals[1]['fields'] = 'entry'

                           
        
def buildUI():
    from grapa.graph import Graph
    root = tk.Tk()
    app = GuiConfigEditor(root, Graph(r'.\config.txt'), None)
    app.mainloop()

if __name__ == "__main__":
    buildUI()