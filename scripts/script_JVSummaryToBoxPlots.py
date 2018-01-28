# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:46:38 2016

@author: car
"""

import glob
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import sys

path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.curve import Curve
try:
    from grapa.scripts.script_processJV import writeFileAvgMax
except ImportError:
    pass



def replaceLabels(graph, label):
    print('label', label)
    if label in ['Voc [V]']:
        label = graph.formatAxisLabel(['Voc', '', 'V'])
    elif label in ['Jsc [mA/cm2]']:
        label = graph.formatAxisLabel(['Jsc', '', 'mA cm$^{-2}$'])
    elif label in ['FF [%]']:
        label = graph.formatAxisLabel(['FF', '', '%'])
    elif label in ['Eff. [%]']:
        label = graph.formatAxisLabel(['Eff.', '', '%'])
    elif label in ['Rp [Ohmcm2]']:
        label = graph.formatAxisLabel(['Rp', '', 'Ohm cm$^2$'])
    elif label in ['Rs [Ohmcm2]']:
        label = graph.formatAxisLabel(['Rs', '', 'Ohm cm$^2$'])
    elif label in ['n']:
        label = graph.formatAxisLabel(['n', '', ' '])
    elif label in ['J0 [A/cm2]']:
        label = graph.formatAxisLabel(['J$_0$', '', 'A cm$^{-2}$'])
    return label

def labelFromGraph(graph, silent=False, file=''):
    # label will be per priority:
    # 1. if a 'label' field is present at beginning of file (so all curve labels are identical)
    # 2. a 'sample' field of 'sample name' field at beginning of file 
    # 3. processed filename
    
    # if label set at beginning of the file -> same label for each curve
    lbl = graph.curve(0).getAttribute('label')
    for c in range(graph.length()):
        if graph.curve(c).getAttribute('label') != lbl:
            lbl = None
            break
    if lbl is not None:
        if not silent:
            print('label:', lbl, '(\'label\' attribute in file header, or unexpected input)')
    if lbl is None:
        # if not every labels are identical: look for 'sample' and 'sample name' fields
        # overrides if find 'sample' keyword in the headers of the file
        if graph.curve(0).getAttribute('sample name') != '':
            lbl = graph.curve(0).getAttribute('sample name')
            if isinstance(lbl, list):
                lbl = lbl[0]
                for c in range(graph.length()): # clean a bit the mess
                    if isinstance(graph.curve(c).getAttribute('sample name'), list):
                        graph.curve(c).update({'sample name': graph.curve(c).getAttribute('sample name')[0]})
            if not silent:
                print('label:', lbl, '(\'sample name\' attribute in file header)')
        elif graph.getAttribute('sample') != '':
            lbl = graph.getAttribute('sample')
            if not silent:
                print('label:', lbl, '(\'sample\' attribute in file header)')
    if lbl is None:
        lbl = os.path.basename(file).replace('export','').replace('summary','').replace('_',' ').replace('  ',' ')
        if not silent:
            print('label:', lbl, '(from file name - no \'label\', \'sample\' or \'sample name\' attribute found in file header)')
    return lbl

    
    
    
def JVSummaryToBoxPlots(folder='.', exportPrefix='boxplot_', replace=None, plotkwargs={}, silent=False, pltClose=True, newGraphKwargs={}):
    if replace is None:
        replace = [] # possible input: [['Oct1143\n', ''], ['\nafterLS','']]
    newGraphKwargs =  copy.deepcopy(newGraphKwargs)
    newGraphKwargs.update({'silent': silent})
    
    # establish list of files
    path = os.path.join(folder, '*.txt')
    fileList = sorted(glob.glob(path))
    files = []
    for file in fileList:
        filename = os.path.basename(file)
        # do notwant to open previously created export files
        if len(exportPrefix) > 0 and filename[:len(exportPrefix)] != exportPrefix:
            files += [file]
    if len(files) == 0:
        return None
    
    # default mode: every column is shown, grouped by columns
    mode = 'all'
    # test if JV mode: only a subset is shown, and some cosmetics is performed
    isJV = True
    graph = Graph(files[0], **newGraphKwargs)
    collabels = graph.getAttribute('collabels')
    if len(collabels) < 8:
        isJV = False
    else:
        collabels = graph.getAttribute('collabels')
        cols = [0, 1, 2, 3, 8, 9] # required columns
        titles = [['Voc'], ['Jsc'], ['FF'], ['Eff'], ['Rp'], ['Rs']]
        for c in range(len(cols)):
            flag = False
            for tmp in titles[c]:
                if tmp in collabels[cols[c]]:
                    flag = True
                    break
            if not flag:
                isJV = False
    if isJV:
        mode = 'JV'
        JVcols = [0, 1, 2, 3, 8, 9, 10, 11] # all columns of interest
        JVtitles = [['Voc'], ['Jsc'], ['FF'], ['Eff'], ['Rp'], ['Rs'], ['n'], ['J0','I0']]
        JVupdates = [{}] * len(JVcols)
        JVupdates[-1] = {'typeplot': 'semilogy'} # J0

    graphs = []
    titles = []
    
    nbCarriageReturnLabel = 0
    strStatistics = ''
    
    for file in files :
        # open the file corresponding to 1 given sample
        if not silent:
            print ('open file',file)
        graph = Graph(file, **newGraphKwargs)
        if graph.length() == 0:
            print('Cannot interpret file data. Go to next one.')
            continue
        
        complement = copy.deepcopy(graph.curve(0).getAttributes())
        complement.update({'type': 'boxplot'})
        
        if mode == 'JV':
            # gather statisics data to be placed in a text file
            try:
                strStatistics += writeFileAvgMax(graph, withHeader=(True if len(strStatistics) == 0 else False), colSample=True, filesave=None)
            except Exception as e:
                print('Exception += ', type(e), e)
                pass
        
        label = labelFromGraph(graph, silent=silent, file=file)
        # "clean" label according to replacement pairs provided by user
        for rep in replace:
            label = label.replace(rep[0], rep[1])
        complement.update({'label': label})
        
        # needed to estimate the graph height
        nbCarriageReturnLabel = max(nbCarriageReturnLabel, label.count('\n'))
    
        # construct one column of each box plot
        collabels = graph.getAttribute('collabels')
        curvesOfInterest = JVcols if mode == 'JV' else range(graph.length())
        for i in range(len(curvesOfInterest)):
            c = curvesOfInterest[i]
            data = graph.curve(c).getData()
            
            # prepare graph title
            while i >= len(titles):
                titles.append('')
            title = collabels[c] if titles[i] == '' else ''
                
            # if input not match JV file, discard it
            if mode == 'JV':
                flag = False
                if c < len(collabels):
                    for tit in JVtitles[i]:
                        if collabels[c].startswith(tit):
                            flag = True
                            break
                if not flag:
                    print('Wrong column name, no data plotted, file', file, 'i', i, 'column', c)
                    data = [[np.nan], [np.nan]]
                    title = ''
                title = title.replace('Voc_V','Voc [V]').replace('Jsc_mApcm2','Jsc [mA/cm2]').replace('FF_pc','FF [%]').replace('Eff_pc','Eff. [%]').replace('_Ohmcm2',' [Ohmcm2]')
                title = title.replace('A/cm2', 'A cm$^{-2}$').replace('J0 ','J$_0$ ').replace('Ohmcm2', '$\Omega$ cm$^2$')
            
            # prepare graph
            if title is not '':
                titles[i] = graph.formatAxisLabel(title)
            while i >= len(graphs):
                graphs.append(Graph('', **newGraphKwargs))
            graphs[i].append(Curve(data, complement))
        # go to next file
        
    # cosmetics, then save plots
    figSize = (1 + 0.9*len(files), 4)
    for i in range(len(graphs)):
        graph = graphs[i]
        if graph.length() > 0:
            # counts number of non-NaN element in each curve, only plot if is > 0
            num = sum([sum(~np.isnan(c.x())) for c in graph.iterCurves()])
            if num > 0:
                graph.update({'figsize': figSize, 'subplots_adjust': [1.1 / figSize[1], (nbCarriageReturnLabel+2)*0.06]})
                #graph.update({'title': titles[i]})
                graph.update({'ylabel': titles[i]})
                filesave = os.path.join(folder, exportPrefix + str(i))
                if mode == 'JV':
                    if JVupdates[i] is not {}:
                        graph.update(JVupdates[i])
                    tit = titles[i]
                    filesave = os.path.join(folder, exportPrefix + tit.split(' ')[0].replace('$','').replace('.','').replace('_',''))
                graph.plot(filesave=filesave, **plotkwargs)
                if pltClose:
                    plt.close()
    
    # print and save statistics
    if len(strStatistics) > 0:
        print(strStatistics)
        filesave = os.path.join(folder, exportPrefix + 'statistics.txt')
        print('filesave', filesave)
        f = open(filesave, 'w')
        f.write(strStatistics)
        f.close()

    # return last plot
    print('JVSummaryToBoxPlots completed.')
    return graphs[0]


if __name__ == "__main__":
    folder = './../examples/boxplot/'
#    folder = './../examples/boxplot/notJVspecific/'
    JVSummaryToBoxPlots(folder=folder, exportPrefix='boxplots_', pltClose=False, silent=True)
    plt.show()

