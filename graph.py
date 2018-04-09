# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:46:13 2016

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import os
import numpy as np
from copy import deepcopy
from re import findall as refindall
import matplotlib as mpl

from grapa.curve import Curve
from grapa.mathModule import is_number, stringToVariable


class Graph:
    """
## TO UPDATE
     content of the class
    'filename': str
    'data': list of Curve
    'headers': dict containing various information about the data
    'graphInfo': dict containing information related to graph.
                 Example: xlim, xlabel, etc.
    'sampleInfo': Deprecated. Dict containing various information about the
                  sample. Generally is empty, as the relevant data are stored
                  in the Curve object.
    
    List of useful methods
    __init__(self, filename, complement='', silent=False)
    __str__(self)

    length(self)
    merge(self, graph)
    append(self, curve)
	
    curve(self, index)
    curves(self, attr, value, strLower=False, strStartWith=False):
    iterCurves(self)
    deleteCurve(self, i)
    replaceCurve(self, newCurve, idx)
    swapCurves(self, idx1, idx2, relative=False)
    duplicateCurve(self, idx1)
    castCurve(self, newType, idx, silentSuccess=False)

    update(self, attributes, ifAll=False)
    getAttribute(self, attr, default='')
	replaceLabels(self, old, new)
    colorize(self, colorscale, sameIfEmptyLabel=False, avoidWhite=False)
    applyTemplate(self, graph)
    
    formatAxisLabel(self, default)
    
    config(self, key, default='', astype='auto')

	
    export(self, filesave='', saveAltered=False, ifTemplate=False, ifCompact=True, ifClipboardExport=False)
    plot(self, filesave='', imgFormat='', figsize=(0, 0), ifSave=True, ifExport=True, figAx=None, ifSubPlot=False)
    """
    # some class constants
    FIGSIZE_DEFAULT = (6.0, 4.0)
    FILEIO_DPI = 80
    DEFAULT = {}
    DEFAULT['subplots_adjust'] = 0.15
    DEFAULT['fontsize'] = mpl.rcParams['font.size']
    # Default axis labels. Normally not used, but can be overidden and used in
    # subclasses. Possible data formats are:
    # ['x axis label [unit]', 'y axis label [unit]']
    # [['Quantity x', 'Q', 'unit'], ['Quantity y', 'Q', 'unit']]. Q will be placed in between $ $
    AXISLABELS = ['', ''] 
    
    
    headersKeys = ['meastype', 'sample', 'collabels'] + ['savesilent']
    graphInfoKeysData = []
    graphInfoKeysData.append(['== Figure ==', ''])
    graphInfoKeysData.append(['figsize',         'Figure size (inch).\nExample: "(6.0, 4.0)"'])
    graphInfoKeysData.append(['subplots_adjust', 'Margins (relative).\nExamples: "0.15" (bottom only), or "[0.125, 0.1, 0.9, 0.9]" left,b,r,top, or "[1,1,5,3.5,\'abs\']"'])
    graphInfoKeysData.append(['dpi',             'Resolution dots-per-inch.\nExample: "300"'])
    graphInfoKeysData.append(['fontsize',        'Font size of titles, annotations, etc.\nExample: "12"'])
    graphInfoKeysData.append(['title',           'Graph title, based on ax.set_title().\nExample: "My data", or "[\'A dataset\', {\'color\':\'r\'}]"'])
    graphInfoKeysData.append(['== Axes ==', ''])
    graphInfoKeysData.append(['xlim',            'Limits of x axis, based on ax.set_xlim().\nExamples: "[2,9]", or "[\'\',4]"'])
    graphInfoKeysData.append(['ylim',            'Limits of y axis, based on ax.set_ylim().\nExamples: "[0,100]", or "[0,\'\']"'])
    graphInfoKeysData.append(['xlabel',          'Label of x axis, based on ax.set_xlabel().\nExample: "Axis x [unit]", "[\'My label\', {\'size\':6, \'color\':\'r\'}]"'])
    graphInfoKeysData.append(['ylabel',          'Label of y axis, based on ax.set_ylabel().\nExample: "Axis y [unit]", "[\'My label\', {\'size\':6, \'color\':\'r\'}]"'])
    graphInfoKeysData.append(['xticksstep',      'Value difference between ticks on x axis, or ticks positions.\nExample: "0.01", or "[0,1,2]"']) # ax.xaxis.set_ticks
    graphInfoKeysData.append(['yticksstep',      'Value difference between ticks on y axis, or ticks positions.\nExample: "0.01", or "[0,1,2]"']) # ax.yaxis.set_ticks
    graphInfoKeysData.append(['xtickslabels',    'Customized ticks. First is a list of values, then a list of labels, then possibly options.\nExamples: "[[0,1],[\'some\',\'value\']]", or "[None, None, {\'rotation\':45, \'size\': 6, \'color\':\'r\'}]"']) # plt.xticks
    graphInfoKeysData.append(['ytickslabels',    'Customized ticks. First is a list of values, then a list of labels, then possibly options.\nExamples: "[[0,1],[\'some\',\'value\']]", or "[None, None, {\'rotation\':45, \'size\': 6, \'color\':\'r\'}]"']) # plt.yticks
    graphInfoKeysData.append(['xlabel_coords',   'Position of xlabel, based on ax.xaxis.set_label_coords().\nExamples: "-0.1", or "[0.5,-0.15]"']) 
    graphInfoKeysData.append(['ylabel_coords',   'Position of ylabel, based on ax.yaxis.set_label_coords().\nExamples: "-0.1", or "[-0.1,0.5]"'])
    graphInfoKeysData.append(['== Legends ==', ''])
    graphInfoKeysData.append(['legendproperties','Position, or keywords to ax.legend(). Examples: "best", "sw", or\n"{\'bbox_to_anchor\':(0.2,0.8), \'ncol\':2, \'fontsize\':8}"'])
    graphInfoKeysData.append(['legendtitle',     'Legend title. Example: "Some title", or "[\'Some title\', {\'size\':25}]"']) # messy implementation
    graphInfoKeysData.append(['== Annotations ==', ''])
    graphInfoKeysData.append(['axhline',         'Horizontal line(s), based on ax.axhline().\nExample: "0", or "[-1, 1]"'])
    graphInfoKeysData.append(['axvline',         'Vertical line(s), based on ax.axvline().\nExample: "0", or "[-1, 1]"'])
    graphInfoKeysData.append(['text',            'Annotations, use GUI window if possible. "Some text", or "[\'Here\', \'There\']"'])
    graphInfoKeysData.append(['textxy',          'Use GUI window if possible. "(0.05, 0.95)" , or "[(0.2, 0.3), (0.8, 0.9)]"'])
    graphInfoKeysData.append(['textargs',        'Use GUI window if possible. "{\'fontsize\':15}", or "[{\'horizontalalignment\': \'right\',\n\'xytext\': (0.4, 0.65), \'arrowprops\': {\'shrink\': 0.05}, \'xy\': (0.46, 0.32)}, {}]"'])
    graphInfoKeysData.append(['== Secondary axes ==', ''])
    graphInfoKeysData.append(['twiny_xlabel',    'Secondary x axis label.\nExamples: "Other axis x [unit]", "[\'My label\', {\'size\':6, \'color\':\'r\'}]"'])
    graphInfoKeysData.append(['twinx_ylabel',    'Secondary y axis label.\nExamples: "Other axis x [unit]", "[\'My label\', {\'size\':6, \'color\':\'r\'}]"'])
    graphInfoKeysData.append(['twiny_xlim',      'Secondary x axis limits.\nExamples: "[2,9]", or "[\'\',4]"'])
    graphInfoKeysData.append(['twinx_ylim',      'Secondary y axis limits.\nExamples: "[0,100]", or "[0,\'\']"'])
    graphInfoKeysData.append(['== Misc ==', ''])
    graphInfoKeysData.append(['alter',           'Data transform keyword, specific to the type of manipulated data.\nExamples: "linear", or "[\'nmeV\', \'tauc\']"'])
    graphInfoKeysData.append(['typeplot',        'General graph plotting instruction, based on ax.set_xscale() and ax.set_yscale().\nExamples: "plot", "semilogx", etc.'])
    graphInfoKeysData.append(['arbitraryfunctions', 'A list of instructions. Each instruction is a list as\n[method of ax, list of arguments, dict of keyword arguments]', ["[[\'xaxis.set_ticks\',[[1.5, 5.5]],{\'minor\':True}], [\'set_xticklabels\',[[\'a\',\'b\']],{\'minor\':True}]]", "[['set_axis_off', [], {}]]", "[['yaxis.set_major_formatter', ['StrMethodFormatter({x:.2f})'], {}]]", "[['xaxis.set_minor_locator', ['MultipleLocator(0.5)'], {}]]"]])
    graphInfoKeysData_ = np.array([d[0:2] for d in graphInfoKeysData])
    graphInfoKeys = list(graphInfoKeysData_[:,0])
    graphInfoKeysExample = list(graphInfoKeysData_[:,1])
    graphInfoKeysExalist = []
    for i in range(len(graphInfoKeysData)):
        if len(graphInfoKeysData[i]) < 3:
            split = graphInfoKeysData[i][1].split('"')
            graphInfoKeysExalist.append(['']+split[1::2])
        else:
            graphInfoKeysExalist.append(graphInfoKeysData[i][2])
    
#    graphInfoKeys = ['xlim', 'ylim', 'figsize', 'title', 'xlabel', 'ylabel'] + ['text', 'textxy', 'textargs'] + ['xticksstep', 'yticksstep'] + ['axhline', 'axvline'] + ['legendproperties', 'legendtitle'] + ['alter', 'typeplot'] + ['twinx_ylabel', 'twinx_ylim', 'twiny_xlabel', 'twiny_xlim'] + ['xlabel_coords', 'ylabel_coords'] + ['subplots_adjust', 'fontsize'] + ['xtickslabels', 'ytickslabels'] + ['arbitraryfunctions'] + ['dpi']
#    graphInfoKeysExample = ["[2,9], or ['',4]", "[0,100], or [0,'']", str(FIGSIZE_DEFAULT)+' (inch)', 'Some title', 'Axis x [unit]', 'Axis y [unit]'] + ['Some text, or [\'Here\', \'There\']', "(0.05, 0.95) , or [(0.2, 0.3), (0.8, 0.9)] if multiple text", "{'fontsize':15}, or \n[{'horizontalalignment': 'right', 'xytext': (0.4, 0.65),\n'arrowprops': {'shrink': 0.05}, 'xy': (0.46, 0.32)}, {}]"] + ['0.01, or [0,1,2]', '0.01, or [0,1,2]'] + ['0, or [-1, 1]', '0, or [-1, 1]'] + ["best, sw, etc., or\n{'bbox_to_anchor':(0.2,0.8),'ncol':2, 'fontsize':8} kwargs to ax.legend()", 'Some title, or [\'Some title\', {\'size\':25}] font properties prop'] + ["linear, or ['nmeV', 'tauc']", 'plot, fill, scatter, boxplot, etc.'] + ['Secondary y axis label [unit]', "[0,100], or [0,'']", 'Secondary x axis label [unit]', "[2,9], or ['',4]"] + ['-0.15 or [0.5,-0.15]', '-0.1 or [-0.1,0.5]'] + ['0.15 (bottom only), or [0.125, 0.1, 0.9, 0.9] left,b,r,top, or [1,1,5,3.5,\'abs\']', '12'] + ["[[0,1],['some','value']], or [None, None, {'rotation':45, 'color':'r'}]", "[[0,1],['some','value']], or [None, None, {'rotation':45, 'color':'r'}]"] + ["[['xaxis.set_ticks',[[1.5, 5.5]],{'minor':True}],\n['set_xticklabels',[['a','b']],{'minor':True}]]"] + [300]
    type_examples = ['', '== usual methods ==']
    # "simple" plotting methods, with prototype similar to plot()
    type_examples += ['plot', 'fill', 'errorbar', 'scatter', 'boxplot']
    type_examples += ['== similar to plot ==', 'semilogx', 'semilogy', 'loglog', 'plot_date', 'stem', 'step', 'triplot']
    # plotting methods not accepting formatting string as 3rd argument
    type_examples += ['== (no linespec) ==', 'bar', 'barbs', 'barh', 'cohere', 'csd', 'fill_between', 'fill_betweenx', 'hexbin', 'hist2d', 'quiver', 'xcorr']
    #  plotting of single vector data
    type_examples += ['== 1D vector data ==', 'acorr', 'angle_spectrum', 'eventplot', 'hist', 'magnitude_spectrum', 'phase_spectrum', 'pie', 'psd', 'specgram']
    type_examples += ['== other ==', 'spy', 'stackplot', 'violinplot']
    type_examples += ['imshow', 'contour', 'contourf']
    # list of attributes recognized for plotting. These are typically attributes of the curves. Can be extended if needed.
    dataInfoKeysGraphData = []
    dataInfoKeysGraphData.append(['== Basic properties ==',  ''])
    dataInfoKeysGraphData.append(['type',           'Plotting method of Axes, ie. "plot", "scatter", "fill", "boxplot", "errorbar", etc.\nTip: after "scatter", set next Curve as "scatter_c" or "scatter_s"', type_examples])
    dataInfoKeysGraphData.append(['linespec',       'Format string controlling the line style or marker.\nExamples: "r", "--", ".-or", "^k"'])
    dataInfoKeysGraphData.append(['label',          'Curve label to be shown in legend. Example: "Experiment C"'])
    dataInfoKeysGraphData.append(['== Display ==',  ''])
    dataInfoKeysGraphData.append(['color',          'Curve color. Examples: "r", "purple", or "[0.5,0,0]" (rgb or rgba notations)'])
    dataInfoKeysGraphData.append(['alpha',          'Transparency. Examples. "0.5", "1"'])
    dataInfoKeysGraphData.append(['linewidth',      'Linewidth in points. Example: "1.5"'])
    dataInfoKeysGraphData.append(['marker',         'Examples: "o", "s", "x"'])
    dataInfoKeysGraphData.append(['markersize',     'Size of marker, in points. Example: "2.5"'])
    dataInfoKeysGraphData.append(['markerfacecolor','Marker inner color. Example: "r", "[0.5,0,0]"'])
    dataInfoKeysGraphData.append(['markeredgecolor','Marker border color. Example: "r", "[0.5,0,0]"'])
    dataInfoKeysGraphData.append(['markeredgewidth','Marker border width, in points. Example: "1.5"'])
    dataInfoKeysGraphData.append(['zorder',         'Determines the drawing order (float), highest is drawn on top.\nExample: "2", "3"'])
    dataInfoKeysGraphData.append(['== Offsets ==',  ''])
    dataInfoKeysGraphData.append(['offset',         'Offset to data. Examples: "-10" (on y values), "[2,\'1/20\']" for (x,y) offset.\nSpecial keywords: "[\'minmax\', \'0max\']" and combinations.'])
    dataInfoKeysGraphData.append(['muloffset',      'Multiplicative offset to data. Examples: "0.01" for y values, or\n"[10, 1e2]" for (x,y) multiplicative offsets'])
    dataInfoKeysGraphData.append(['== For specific curve types ==', ''])
    dataInfoKeysGraphData.append(['facecolor',      'Color of "fill" Curve types. Examples: "r", "[0.5,0,0]"'])
    dataInfoKeysGraphData.append(['cmap',           'Colormap, for Curve types which accept this keyword such as scatter). Examples:\n"afmhot", "inferno", or "[[0.91,0.25,1], [1.09,0.75,1], \'hls\']" (see Colorize options)'])
    dataInfoKeysGraphData.append(['vminmax',        'Bounds values for cmap. Examples: "[0,7]", or "[3, \'\']"'])
    dataInfoKeysGraphData.append(['colorbar',       'If not empty display a colorbar according to keyword cmap. Example: "1",\n"{\'ticks\': [-1, 0, 2]}", or "{\'orientation\':\'horizontal\', \'adjust\':[0.1, 0.1, 0.7, 0.1]}"'])
    dataInfoKeysGraphData.append(['xerr',           'x error for curve type "errorbar". Example: "1", or "5"'])
    dataInfoKeysGraphData.append(['yerr',           'y error for curve type "errorbar". Example: "1", or "5"'])
    dataInfoKeysGraphData.append(['== Misc ==',     ''])
    dataInfoKeysGraphData.append(['labelhide',      'Use "1" to hide label in the graph'])
    dataInfoKeysGraphData.append(['legend',         'Deprecated. Curve label in legend.'])
    dataInfoKeysGraphData.append(['ax_twinx',       'Plot curve on secondary axis. Example: "True", or anything'])
    dataInfoKeysGraphData.append(['ax_twiny',       'Plot curve on secondary axis. Example: "True", or anything'])
    dataInfoKeysGraphData.append(['linestyle',      'Use "none" to hide a Curve'])
    dataInfoKeysGraphData.append(["['key', value]", 'User-defined keyword-values pairs. Will be fed to the plotting method if possible.\nExamples: "[\'fillstyle\', \'top\']" for half-filled markers, "[\'comment\', \'a valuable info\']]'])
    dataInfoKeysGraphData_ = np.array([d[0:2] for d in dataInfoKeysGraphData])
    dataInfoKeysGraph = list(dataInfoKeysGraphData_[:,0])
    dataInfoKeysGraphExample = list(dataInfoKeysGraphData_[:,1])
    dataInfoKeysGraphExalist = []
    for i in range(len(dataInfoKeysGraphData)):
        if len(dataInfoKeysGraphData[i]) < 3:
            split = dataInfoKeysGraphData[i][1].split('"')
            dataInfoKeysGraphExalist.append(['']+split[1::2])
        else:
            dataInfoKeysGraphExalist.append(dataInfoKeysGraphData[i][2])
        
#    dataInfoKeysGraph = ['type', 'linespec', 'color', 'legend', 'label', 'labelhide', 'linewidth', 'facecolor', 'linestyle', 'alpha', 'marker', 'markersize', 'markerfacecolor', 'markeredgewidth', 'markeredgecolor', 'cmap', 'vminmax', 'colorbar', 'xerr', 'yerr', 'offset', 'muloffset', 'ax_twinx', 'ax_twiny', 'insetfile', "['key', value]"]
#    dataInfoKeysGraphExample = ['plot, scatter, fill, boxplot, errorbar, etc. (scatter_c after previous scatter)', '--r', 'r, [0.5,0,0], etc.', 'Some legend', 'Some legend', '1 to hide label in graph', '1.5', 'r, [0.5,0,0], etc.', '"" or none', '0.5, 1, etc.', 'o, or s, etc.', '2', 'r, [0.5,0,0], etc.', '1.', 'r, [0.5,0,0], etc.', 'afmhot, or inferno, or color list as in Colorize', 'Imposes bounds for cmap. Ex: [0,7], or [3, \'\']', '1, or {\'ticks\': [-1, 0, 2]}, or {\'orientation\':\'horizontal\', \'adjust\':[0.1, 0.1, 0.7, 0.1]}', '1, or 5. Requires \'type\': \'errorbar\'.', '1, or 5. Requires \'type\': \'errorbar\'.', '-10, or [2,20]', '0.01, or [10, 1e2]', 'True, or anything', 'True, or anything', 'file to place as inset. Related keywords: insetcoords and insetupdate', "['color', [0,0.2,0.7], or ['comment', 'a valuable info']]"]

    # constructor
    def __init__(self, filename='', complement='', silent=True, config='config.txt'):
        """ default constructor. Calls method reset. """
        # load preferences
        if config is not None:
            if config == 'config.txt':
                config = os.path.join(os.path.dirname(os.path.realpath(__file__)), config)
            self._config = Graph(config, complement={'readas': 'database'}, config=None)
        # actually load the file
        self.reset(filename, complement=complement, silent=silent)
    

    def reset(self, filename, complement='', silent=True):
        # complement: special keywords: 'readas', 'isfilecontent'
        # default values
        from grapa.graphIO import GraphIO
        self.filename = filename
        self.silent = silent
        self.data = []
        self.headers = {}
        self.graphInfo = {}
        self.sampleInfo = {}
        # try identify and parse the datafile
        if isinstance(filename, str) and filename == '':
            if not self.silent:
                print('Empty Graph object created')
            return
        if isinstance(filename, str):
            # if single file was provided - a string, or if filename is the content if the file - complement['isfilecontent'] must be true
            GraphIO.readDataFile(self, complement=complement)
        else:
            if len(filename) == 2 and len(filename[0]) == len(filename[1]) and is_number(filename[0][0]):
                # filename is actually the data, filename is a list
                if not self.silent:
                    print('Class Graph: interpret "filename" as data content')
                self.filename = ''
                GraphIO.dataFromVariable(self, filename, attributes=complement)
            else:
                # if a list was provided - open first file, then merge the others one by one
                if not isinstance(complement, list):
                    if complement != '':
                        print('WARNING class Graph: complement must be a list if filename is a list.')
                        print('   filename:', len(filename), 'element, complement:', complement)
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
        if hasattr(self, 'filename'):
            self.filename = os.path.abspath(self.filename)

# RELATED TO GUI
    def alterListGUI(self):
        out = []
        out.append(['Linear', ['', ''], '']) # do not want to leave this completely empty (if no curve left)
        for c in range(self.length()):
            for j in self.curve(c).alterListGUI():
                if j not in out:
                    out.append(j)
        return out




# "USUAL" CLASS METHODS
    def __str__(self):
        """ Returns some information about the class instance. """
        out = 'Content of Graph stored in file ' + self.filename + '\n'
        out += 'Content of headers:' + '\n'
        out += str(self.headers)
        out += '\nNumber of Curves: '+str(self.length())
        return out


    # interactions with other Graph objects
    def merge(self, graph):
        """ Add in a Graph object the content of another Graph object """
        # keep previous self.filename
        # copy data
        for x in graph.data:
            self.data.append(x)
        # copy headers, unless already exists (in this case information is forgotten)
        for key in graph.headers:
            if not key in self.headers:
                self.headers.update({key: graph.headers[key]})
        # copy graphInfo, unless already exists (in this case information is forgotten)
        for key in graph.graphInfo:
            if not key in self.graphInfo:
                self.graphInfo.update({key: graph.graphInfo[key]})
        # copy sampleInfo, unless already exists (in this case information is forgotten)
        for key in graph.sampleInfo:
            if not key in self.sampleInfo:
                self.sampleInfo.update({key: graph.sampleInfo[key]})

    
    # methods handling the bunch of curves
    def length(self):
        """ Returns the number of Curve objects in the list. """
        return len(self.data)

    def curve(self, index):
        """ Returns the Curve object at index i in the list. """
        if index >= self.length() or self.length() == 0:
            print ('ERROR Class Graph method Curve: cannot find Curve (index',index,', max possible',self.length()-1,')')
            return 
        return self.data[index]

    def curves(self, attr, value, strLower=False, strStartWith=False):
        """ 
        Returns a list of Curves which attribute attr == value (also same type)
        attr: the attribute to check. By default 'label'.
        strLower: if True, 
        strStartWith: is both value and the attribute are str, only look at the
            first characters
        """
        out = []
        for c in self.iterCurves():
            lbl = c.getAttribute(attr)
            if isinstance(lbl, type(value)):
                if strLower:
                    if isinstance(lbl, str):
                        lbl = lbl.lower()
                    if isinstance(value, str):
                        value = value.lower()
                if lbl == value:
                    out.append(c)
                elif isinstance(lbl, str) and strStartWith and lbl[:len(value)] == value:
                    out.append(c)
        return out

    def iterCurves(self):
        """ Returns an iterator over the different Curves """
        for c in range(self.length()):
            yield self.curve(c)
    
    def append(self, curve, idx=None):
        """
        Add into the object list a Curve, a list or Curves, or every Curve in a
        Graph object.
        """
        insert = False if idx is None else True
        if isinstance(curve, Graph):
            for c in curve.iterCurves():
                if insert:
                    self.data.insert(idx, c)
                    idx += 1
                else:
                    self.data.append(c) # c are Curves, we can do like that
        elif isinstance(curve, list):
            for c in curve:
                if insert:
                    self.data.insert(idx, c)
                    idx += 1
                else:
                    self.append(c) # call itself, must check if c is a Curve
        elif isinstance(curve, Curve):
            if insert:
                self.data.insert(idx, curve)
            else:
                self.data.append(curve)
        elif isinstance(curve, str) and curve == 'empty':
            curve = Curve([[np.inf],[np.inf]], {})
            if insert:
                self.data.insert(idx, curve)
            else:
                self.data.append(curve)
        else:
            print('Graph.append: failed (type:',type(curve),')')

    def deleteCurve(self, i):
        """ Delete a Curve at index i from the Graph object. """
        if isinstance(i, (list, tuple)):
            i = list(np.sort(i)[::-1])
            for k in i:
                self.deleteCurve(k)
        else:
            le0 = len(self.data)
            if is_number(i) and i < le0:
                # delete data
                del self.data[i]
                # delete in headers
                if 'collabels' in self.headers and i < len(self.headers['collabels']):
                    del self.headers['collabels'][i]
                if 'collabelsdetail' in self.headers: # certainly only useful for "databases"
                    for j in range(len(self.headers['collabelsdetail'])):
                        if i < len(self.headers['collabelsdetail'][j]):
                            del self.headers['collabelsdetail'][j][i]
                # nothing to delete in graphInfo
                # nothing to delete in sampleInfo

    def replaceCurve(self, newCurve, idx):
        """ Replaces a Curve with another. """
        if isinstance(newCurve, Curve):
            try:
                self.data[idx] = newCurve
                return True
            except Exception:
                print ('Graph.replaceCurve: cannot add Curve at index', idx, '.')
        else:
            print ('Graph.replaceCurve: newCurve is not a Curve (type', type(newCurve), ')')
        return False
    
    def swapCurves(self, idx1, idx2, relative=False):
        """
        Exchange the Curves at index idx1 and idx2.
        For example useful to modify the plot order (and order in the legend)
        """
        if idx1 < -self.length() or idx1 >= self.length():
            print ('Graph.swapCurves: idx1 not valid (value', idx1, ').')
            return False
        if relative:
            idx2 = idx1 + idx2
        if idx2 == idx1: # swap with itself
            return True
        if idx2 < -self.length() or idx2 >= self.length():
            print ('Graph.swapCurves: idx2 not valid (value', idx2, ').')
            return False
        swap = deepcopy(self.curve(idx1))
        self.data[idx1] = self.curve(idx2)
        self.data[idx2] = swap
        return True

    def duplicateCurve(self, idx1):
        """ Duplicate (clone) an existing curve and append it in the curves list."""
        if idx1 < -self.length() or idx1 >= self.length():
            print ('Graph.duplicateCurve: idx1 not valid (value', idx1, ').')
            return False
        curve = deepcopy(self.curve(idx1))
        self.data.insert(idx1+1, curve)

        
    # methods handling content of curves
    def getCurveData(self, idx, ifAltered=True):
        if ifAltered:
            alter = self._getAlter()
            x = self.curve(idx).x_offsets(alter=alter[0])
            y = self.curve(idx).y_offsets(alter=alter[1])
        else:
            x = self.curve(idx).x(alter='')
            y = self.curve(idx).y(alter='')
        return np.array([x,y])

    def update(self, attributes, ifAll=False, forceGraphInfo=False):
        """ Update the properties of the Graph object.
        The properties are stored in dict objects, so update works similarly as
        the dict.update() method.
        If a property is not known to belong to the Graph object, it is
        attributed to the 1st Curve object (unless ifAll=True), or if
        forceGraphInfo is True (then the undetermined keys are inserted in
        self.graphInfo)
        """
        for key in attributes:
            k = key.lower().replace('ï»¿','')
            try:
                if   k in self.headersKeys:
                    if attributes[key] != '':
                        self.headers.update({k: attributes[key]})
                    elif k in self.headers:
                        del self.headers[k]
                elif (k in self.graphInfoKeys or forceGraphInfo) or k.startswith('subplots'):
                    if attributes[key] is not '':
                        self.graphInfo.update({k: attributes[key]})
                    elif k in self.graphInfo:
                        del self.graphInfo[k]
                    # by default nothing in sampleInfo, everything in the curves
                else:
                    if ifAll:
                        for i in range(self.length()):
                            self.curve(i).update({k: attributes[key]})
                    else:
                        self.curve(-1).update({k: attributes[key]})
            except Exception as e:
                print('Error Graph.update: key', key, ' attributes', attributes, 'exception', e)

    def updateValuesDictkeys(self, *args, **kwargs):
        """
        Performs update({key1: value1, key2: value2, ...}) with
        arguments value1, value2, ... , (*args) and
        kwargument key=['key1', 'key2', ...]
        """
        if 'keys' not in kwargs:
            print('Error Graph updateValuesDictkeys: "keys" key must be provided, must be a list of keys corresponding to the values provided in *args.')
            return False
        if len(kwargs['keys']) != len(args):
            print('WARNING Graph updateValuesDictkeys: len of list "keys" argument must match the number of provided values (',len(args),' args provided, ',len(kwargs['keys']),' keys).')
        lenmax = min(len(kwargs['keys']), len(args))
        for i in range(lenmax):
            self.update({kwargs['keys'][i]: args[i]})
        return True
                
    def delete(self, key, ifAll=False):
        """
        Delete an attribute of the Graph if recognized as such, or of the curve.
        Return the deleted attribute (except for attributes of curves if ifAll=True)
        """
        k = key.lower()
        out = {}
        if k in self.headersKeys:
            if k in self.headers:
                out.update({key: self.headers[k]})
                del self.headers[k]
        elif k in self.graphInfoKeys:
            if k in self.graphInfo:
                out.update({key: self.graphInfo[k]})
                del self.graphInfo[k]
        else:
            if ifAll:
                for i in range(self.length()):
                    # cannot export in this case
                    self.data[i].delete(key)
            else:
                out.update(self.curve(-1).delete(key))
        return out

    def getAttribute(self, attr, default=''):
        at = attr.lower()
        if at in self.headers:
            return self.headers[at]
        if at in self.graphInfo:
            return self.graphInfo[at]
        if at in self.sampleInfo:
            return self.sampleInfo[at]
        if self.length() > 0:
            return self.curve(0).getAttribute(attr, default=default)
        return default

    def deleteAttr(self, attrList):
        out = {}
        for key in attrList:
            out.update({key: self.getAttribute(key)})
            self.delete(key)
        return out

    def castCurve(self, newtype, idx, silentSuccess=False):
        """
        Replace a Curve with another type of Curve with identical data and 
        properties.
        """
        if idx >= - self.length() and idx < self.length():
            newCurve = self.curve(idx).castCurve(newtype)
            if isinstance(newCurve, Curve):
                flag = self.replaceCurve(newCurve, idx)
                if flag:
                    if not silentSuccess:
                        print ('Graph.castCurve: new Curve type:', self.curve(idx).classNameGUI() + '.')
                else:
                    print ('Graph.castCurve')
                return flag
        else:
            print ('Graph.castCurve: idx not in suitable range (', idx,', max', self.length(),').')
        return False
    

    def colorize(self, colorscale, sameIfEmptyLabel=False, avoidWhite=False):
        """
        Colorize a graph, by coloring each Curve along a colorscale gradient.
        """
        from grapa.colorscale import Colorscale
        if not isinstance(colorscale, Colorscale):
            colorscale = Colorscale(colorscale)
        if self.length() == 1:
            self.curve(0).update({'color': colorscale.valuesToColor(0.0, avoidWhite=avoidWhite)})
            return
        show = np.arange(0, self.length())
        if sameIfEmptyLabel:
            show = np.array([0.0] * self.length())
            i = 0.0
            for c in range(self.length()):
                show[c] = i
                if self.curve(c).getAttribute('label') != '' and not self.curve(c).isHidden():
                    i += 1.0
                elif c > 0:
                    show[c] = show[c-1]
        cols = colorscale.valuesToColor(show/max(max(show),1.0), avoidWhite=avoidWhite)
        for c in range(self.length()):
            self.curve(c).update({'color': cols[c]})


    def applyTemplate(self, graph, alsoCurves=True):
        """
        Apply a template to the Graph object. The template is a Graph object
        which contains:
        - self.getAttributes listed in graphInfoKeys
        - self.curve(i).getAttributes listed in dataInfoKeysGraph
        alsoCurves: True also apply Curves properties, False only apply Graph
        properties
        """
        # strip default attributes
        for key in Graph.DEFAULT:
            if graph.getAttribute(key) == Graph.DEFAULT[key]:
                graph.update({key: ''})
        for key in graph.graphInfoKeys:
            val = graph.getAttribute(key)
            if val != '':
                self.update({key: val})
        if alsoCurves:
            for c in range(self.length()):
                if c >= graph.length():
                    break
                for key in self.dataInfoKeysGraph:
                    val = graph.curve(c).getAttribute(key)
                    if val != '':
                        self.curve(c).update({key: val})
    
    def replaceLabels(self, old, new):
        """
        Modify all labels of the Graph, by replacing 'old' by 'new'
        """
        for c in self.iterCurves():
            c.update({'label': c.getAttribute('label').replace(old, new)})

    def checkValidText(self):
        text = self.getAttribute('text', None)
        texy = self.getAttribute('textxy', '')
        targ = self.getAttribute('textargs', {})
        if text == None:
            self.update({'textxy': '', 'textargs': ''})
            return
        onlyfirst = False if isinstance(text, list) else True
        # transform everything into lists
        if not isinstance(text, list):
            text = [text]
        if not isinstance(targ, list):
            targ = [targ]
        if not isinstance(texy, list):
            texy = [texy]
        if len(texy) == 2 and not isinstance(texy[0], (list, tuple)) and not texy in [['',''], ('','')]:
            texy = [texy]  # if texy was like (0.5,0.8)
        for i in range(len(targ)):
            if not isinstance(targ[i], dict):
                #print('Graph.checkValidText targ set', i, '{} (previous', targ[i], ')')
                targ[i] = {}
        for i in range(len(texy)):
            if not isinstance(texy[i], (tuple, list)):
                #print('Graph.checkValidText texy set', i, '\'\' (previous', texy[i], ')')
                texy[i] = ''
        while len(texy) < len(text):
            texy.append(texy[-1])
        while len(targ) < len(text):
            targ.append(targ[-1])
        if onlyfirst:
            text, texy, targ = text[0], texy[0], targ[0]
        if text != self.getAttribute('text'):
            print('Corrected attribute text', text, '(former', self.getAttribute('text'),')')
        if texy != self.getAttribute('textxy'):
            print('Corrected attribute textxy', texy, '(former', self.getAttribute('textxy'),')')
        if targ != self.getAttribute('textargs'):
            print('Corrected attribute textargs', targ, '(former', self.getAttribute('textargs'),')')
        self.update({'text': text, 'textxy': texy, 'textargs': targ})
                    
    def addText(self, text, textxy, textargs=None):
        """
        Adds a text to be annotated in the plot, handling the not-so-nice
        internal implementation
        """
        if textargs is None:
            textargs = {}
        restore = []
        restore.append({'text':     self.getAttribute('text'),
                        'textxy':   self.getAttribute('textxy'),
                        'textargs': self.getAttribute('textargs')})
        attrs = {'text': text, 'textxy': textxy, 'textargs': textargs}
        # TODO: ensure text, textxy and textargs if list have same length
        if self.getAttribute('text', None) is None:
            self.update(attrs)
        else:
            if not isinstance(self.getAttribute('text'), list):
                for key in attrs.keys(): # ensures attributes are list
                    self.update({key: [self.getAttribute(key, '')]})
            for key in attrs.keys():
                self.update({key: self.getAttribute(key) + [attrs[key]]})
        return restore
    def removeText(self):
        restore = []
        restore.append({'text':     self.getAttribute('text'),
                        'textxy':   self.getAttribute('textxy'),
                        'textargs': self.getAttribute('textargs')})
        attrs = ['text', 'textxy', 'textargs']
        if self.getAttribute('text', None) is None:
            pass
        else:
            if not isinstance(self.getAttribute('text'), list):
                for key in attrs: # ensures attributes are list
                    self.update({key: [self.getAttribute(key, '')]})
            for key in attrs:
                self.update({key: self.getAttribute(key)[:-1]})
        return restore

            
            
    
    # some mathemetical operation on curves
    def curvesAdd(self, idx0, idx1, interpolate=0, **kwargs):
        """ addition operation when only knowing curves indices """
        kwargs.update({'interpolate': interpolate, 'sub': False})
        return self.curve(idx0).__add__(self.curve(idx1), **kwargs)
    def curvesSub(self, idx0, idx1, interpolate=0, **kwargs):
        """ substraction operation when only knowing curves indices """
        kwargs.update({'interpolate': interpolate, 'sub': True})
        return self.curve(idx0).__add__(self.curve(idx1), **kwargs)
        
    def _curveMethod_graphRef(self, *args, **kwargs):
        """
        Allows operations on the graph to be executed from inside the Curve
        object. See for example CurveMath.
        Expect argument 'curve': a Curve object
        Expect argument 'method': will call curve.method
        """
        if 'curve' in kwargs and 'method' in kwargs:
            curve = kwargs['curve']
            m = kwargs['method']
            del kwargs['curve'], kwargs['method']
            return getattr(curve, m)(*args, graph=self, **kwargs)

    
    def _getAlter(self):
        """ returns the formatted alter instruction of self """
        return self._getAlterToFormat(self.getAttribute('alter'))
    @classmethod
    def _getAlterToFormat(cls, alter):
        """ Return a formatted alter instruction """
        if alter == '':
            alter = ['', '']
        if isinstance(alter, str): # nothing to do if it is dict
            alter = ['', alter]
        return alter
        

            
    def formatAxisLabel(self, label):
        """
        Returns a string for label according to user preference.
        'Some quantity [unit]' may become 'Some quantify (unit)'
        Other possible input is ['This is a Quantity', 'Q', 'unit']
        """
        # retrieve user preference
        symbol= bool(self.config('graph_labels_symbols', False))
        units = self.config('graph_labels_units', default='[]', astype=str)
        units = units.replace('unit', '').replace(' ','')
        # format input
        if isinstance(label, str):
            if units != '[]': # that is default, no need to do anything
                expr = '^(.* )\[(.*)\](.*)$'
                if label != '':
                    f = refindall(expr, label)
                    if isinstance(f, list) and len(f) == 1 and len(f[0]) == 3:
                        if units in ['DIN', '/']:
                            return f[0][0].strip(' ') + ' / ' + f[0][1] + '' + f[0][2]
                        elif units == '()':
                            return f[0][0].strip(' ') + '(' + f[0][1] + ')' + f[0][2]
        elif isinstance(label, list):
            while len(label) < 3:
                label += ['']
            out = label[0]
            if symbol and len(label[1]) > 0:
                out += ' $' + label[1] + '$'
            if label[2] not in [None, ' ']:
                if units == '/':
                    out += ' / ' + label[2]
                elif units == '()':
                    out += ' (' + label[2] + ')'
                else:
                    out += ' [' + label[2] + ']'
            return out.replace('  ', ' ')
        return label
        
        
    def config(self, key, default='', astype='auto'):
        """
        returns the value corresponding to key in the configuration file.
        If key is not defined, returns default.
        """
        if hasattr(self, '_config') and self._config.length() > 0:
            out = self._config.curve(0).getAttribute(key, default=default)
            if astype in [str, 'str']:
                return str(out)
            else:
                return stringToVariable(out)
        return default

    
    def filenamewithpath(self, filename):
        # if relative, join the path of the file with
        if os.path.isabs(filename):
            return filename
        path = ''
        if hasattr(self, 'filename') and isinstance(self.filename, str) and len(self.filename) > 0:
            path = os.path.dirname(os.path.abspath(self.filename))
        return os.path.join(path, filename)
        
        
    # For convenience we offer a shortcut to GraphIO.export
    def export(self, filesave='', saveAltered=False, ifTemplate=False,
               ifCompact=True, ifOnlyLabels=False, ifClipboardExport=False):
        """
        Exports content of Grah object into a human- and machine- readable
        format.
        """
        from grapa.graphIO import GraphIO
        return GraphIO.export(self, filesave=filesave, saveAltered=saveAltered,
                              ifTemplate=ifTemplate, ifCompact=ifCompact,
                              ifOnlyLabels=ifOnlyLabels,
                              ifClipboardExport=ifClipboardExport)
        

    # For convenience we offer a shortcut to GraphIO.plot
    def plot(self, filesave='', imgFormat='', figsize=(0, 0), ifSave='auto',
             ifExport='auto', figAx=None, ifSubPlot='auto'):
        """
        Plot the content of the object.
        filesave: filename for the saved graph image and export text
            file.
        imgFormat: by default image format will be .png. Possible formats are
            the ones supported by plt.savefig() and possibly .emf
        figsize: figure size in inch. The actual default value is from a class
            constant
        ifSave: [True/False/'auto'] if True, save the Graph as an image. If
            left to default, saves the image only if filesave is provided
        ifExport: [True/False/'auto'] if True, create a human- and machine-
            readable .txt file containing all information of the graph.
            By default, only export if filesave is provided
        figAx as [fig, ax]: the graph will be plotted in the provided figure
            and ax.
            Providing [fig, None] will erase all existing axes and create what
            is needed
        ifSubPlot: [True/False/'auto'] If True, prevents deletion of the
            existing axes in the figure.
            By default keeps axes if axes are provided in figAx.
            
        """
        if ifSave not in [True, False]:
            ifSave = False if filesave == '' else True
        if ifExport not in [True, False]:
            ifExport = False if filesave == '' else True
        from grapa.graphIO import GraphIO
        if ifSubPlot not in [True, False]:
            ifSubPlot = False
            if (figAx is not None and isinstance(figAx, list)
                and len(figAx) > 1 and figAx[1] is not None):
                ifSubPlot = True
        return GraphIO.plot(self, filesave=filesave, imgFormat=imgFormat, figsize=figsize,
                            ifSave=ifSave, ifExport=ifExport, figAx=figAx,
                            ifSubPlot=ifSubPlot)
