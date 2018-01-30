# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:19:52 2017

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
from copy import deepcopy
import numpy as np

from grapa.graph import Graph
from grapa.curve import Curve

class Curve_Image(Curve):
    """
    The purpose is this class is to provid GUI support to create insets
    """

    CURVE = 'image'

    def __init__(self, *args, **kwargs):
        Curve.__init__(self, *args, **kwargs)
        # define default values for important parameters
        if self.getAttribute('type') not in ['imshow', 'contour', 'contourf']:
            self.update ({'type': 'imshow'})
        # legacy keyword
        imagefile = deepcopy(self.getAttribute('imagefile', None))
        if imagefile is not None:
            self.update({'datafile': imagefile, 'imagefile': ''})
        self.update({'Curve': Curve_Image.CURVE})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        texttype, default = "keyword 'type'", self.getAttribute('type')
        if default not in ['imshow', 'contour', 'contourf']:
            texttype = "issue: keyword 'type' should be:"
            default = 'imshow'
        out.append([self.updateValuesDictkeys, 'Set', [texttype], [default],
                    {'keys': ['type']},
                    [{'field':'Combobox','values':['imshow','contour','contourf']}]])
        interpolation = ['', 'none', 'nearest', 'bilinear', 'bicubic',
            'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
            'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
            'lanczos']
        aspect = ['', '1 scalar', 'auto', 'equal']
        datafile_XY1rowcol = bool(self.getAttribute('datafile_XY1rowcol'))
        # file; X,Y 1st row column
        out.append([self.updateValuesDictkeys, 'Set',
                    ['data file', 'first row, column as coordinates'],
                    [self.getAttribute('datafile'), datafile_XY1rowcol],
                    {'keys': ['datafile', 'datafile_XY1rowcol']},
                    [{'width':25},{'field':'Checkbutton'}]])
        # transpose, rotate
        at = ['transpose', 'rotate']
        out.append([self.updateValuesDictkeys, 'Set', at,
                    [self.getAttribute(a) for a in at], {'keys': at},
                    [{'field':'Combobox', 'values':['','True','False']},{}]])
        # extent
        if not datafile_XY1rowcol:
            extent = list(self.getAttribute('extent'))
            while len(extent) < 4:
                extent.append('')
            out.append([self.updateExtent, 'Set', ['extent left', 'right', 'bottom', 'top'], extent])
        else:
            out.append([None, '(extent not active if first row, column as coordinates)', [], []])
        # colormap
        attxt = ['cmap (ignored if color image)', 'vmin', 'vmax']
        at = [a.split(' ')[0] for a in attxt]
        out.append([self.updateValuesDictkeys, 'Set', attxt, [self.getAttribute(a) for a in at], {'keys': at}])
        if self.getAttribute('type') == 'imshow':
            at = ['aspect', 'interpolation']
            out.append([self.updateValuesDictkeys, 'Set',
                        ['aspect ratio', 'interpolation'],
                        [self.getAttribute(a) for a in at], {'keys': at},
                        [{'field':'Combobox', 'values':aspect, 'bind':'beforespace'},
                         {'field':'Combobox', 'values':interpolation}]])
        else: # contour, contourf
            at = ['levels']
            out.append([self.updateValuesDictkeys, 'Set',
                        ['levels (list of values)'],
                        [self.getAttribute(a) for a in at], {'keys': at}])
        
        return out

    def updateExtent(self, *args):
        flag = False
        for a in args:
            if a != '':
                flag = True
        self.update({'extent': (list(args) if flag else '')})
    
    def printHelp(self):
        print('*** *** ***')
        print('Class Curve_Image facilitates the creation and customization of insets inside a Graph')
        print('Important parameters:')
        print('- insetfile: a path to a saved Graph, either absolute or relative to the main Graph.')
        print('  if set to ' ' (whitespace) or if no Curve are found in the given file, the next Curves of the main graph will be displayed in the inset.')
        print('- insetcoords: coordinates of the inset axis, relative to the main graph. Prototype: [left, bottom, width, height]')
        print('- insetupdate: a dict which will be applied to the inset graph. Provides basic support for run-time customization of the inset graph.')
        print("  Examples: {'fontsize':8}, or {'xlabel': 'An updated label'}")
        return True

    
    def getImageData(self, graph, graph_i, alter, ignoreNext=0):
        """
        graph: the Graph the Curve is in
        graph_i: index of Curve in graph
        alter: 2-elements list for alter
        """
        transpose = self.getAttribute('transpose', False)
        rotate    = self.getAttribute('rotate', False)
        datafile  = self.getAttribute('datafile', None)
        
        data = np.zeros((2,2))
        X, Y = None, None
        if datafile is not None:
            if graph is not None:
                datafile = graph.filenamewithpath(datafile)
            try:
                from PIL import Image as PILimage
                data = PILimage.open(datafile)
                if rotate:
                    data = data.rotate(rotate)
                if transpose:
                    data = data.transpose(PILimage.TRANSPOSE)
            except OSError: # file is not an image -> assume is data
                complement = {'readas': 'generic', '_singlecurve':True}
                graphtmp = Graph(datafile, complement)
                if graphtmp.length() > 0:
                    data = graphtmp.curve(0).getData()
                    if transpose:
                        data = np.transpose(data)
                    if self.getAttribute('datafile_XY1rowcol', False):
                        X = data[0,1:]
                        Y = data[1:,0]
                        data = data[1:,1:]
                    if rotate:
                        data = np.rot90(data, k=int(rotate)) # only by 90°
                else:
                    import os
                    print('Curve image cannot find data.', ('File '+datafile+' does not seem to exist.' if not os.path.isfile(datafile) else ''))
        else: # greedily aggregate data of following curves, provided x match
            x = self.x_offsets(alter=alter[0])
            y = self.y_offsets(alter=alter[1])
            data = [x, y]
            if graph is not None:
                for j in range(graph_i+1, graph.length()):
                    if np.array_equal(x, graph.curve(j).x_offsets(alter=alter[0])):
                        data.append(graph.curve(j).y_offsets(alter=alter[0]))
                        ignoreNext += 1
                    else:
                        break
            data = np.transpose(data) if transpose else np.array(data)
            if rotate:
                data = np.rot90(data, k=int(rotate)) # only by 90°
        return data, ignoreNext, X, Y
    