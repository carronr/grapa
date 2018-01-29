# -*- coding: utf-8 -*-
"""
@author: car
Copyright (c) 2018, Empa, Romain Carron
"""

import numpy as np
import os
import sys

import glob
from copy import deepcopy
from re import search as research
import matplotlib.pyplot as plt

path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.graphIO import GraphIO
from grapa.database import Database
from grapa.mathModule import is_number, roundSignificant, stringToVariable
from grapa.colorscale import Colorscale
from grapa.curve import Curve
from grapa.curve_subplot import Curve_Subplot

from grapa.datatypes.curveJV import GraphJV
from grapa.datatypes.graphJVDarkIllum import GraphJVDarkIllum


# prompt for folder
def promptForFolder ():
    # Start->Execute, cmd, type "pip install tkinter"
    from tkinter.filedialog import askdirectory
    path = askdirectory()
    return path



# auxiliary function
def dictToListSorted (d) :
    l = []
    for key in d :
        l.append(key)
    l.sort()
    return l

# auxiliary class - add some useful tools to the database class
class areaDB (Database) :
    def __init__ (self, folder, sample) :
        self.colIdx = 0
        self.folder = folder
        self.sample = sample
        # different possible names for area databases
        test = []
        test.append([os.path.join(folder, sample+'.txt'), ''])
        test.append([os.path.join(folder, sample+'_area.txt'), ''])
        test.append([os.path.join(folder, sample+'_areas.txt'), ''])
        test.append([os.path.join(folder, 'area.txt'), ''])
        test.append([os.path.join(folder, 'area.xlsx'), sample])
        test.append([os.path.join(folder, 'areas.txt'), ''])
        test.append([os.path.join(folder, 'areas.xlsx'), sample])
        self.flag = False
        self.flagCheat = False
        for t in test :
            print ('Area database: look in file ',t[0])
            graph = Graph(t[0], complement=t[1], silent=True, config=None)
            if graph.length() == 0 :
                continue
            else :
                try :
                    # try convert it in database
                    Database.__init__ (self, graph)
                    # identify suitable column
                    for col in self.colLabels :
                        if col.find ('area') > -1 :
                            self.colIdx = col
                    self.flag = True
                    print ('   database file parsed and area identified.')
                except Exception:
                    print ('   database file data could not be interpreted correctly.')
#                    print (e)
                    continue
                break # break if success
        if not self.flag :
            print ('areaDB: cannot find area database file.')
    
    def getArea (self, cell):
        if not self.flag : # if database could not be opened
            if self.flagCheat:
                if cell in self.data:
                    return self.data[cell]
            return np.nan
        out = self.value (self.colIdx, cell)
        if np.isnan (out) :
            out = self.value (self.colIdx, self.sample+' '+cell)
        if np.isnan (out) :
            print ('areaDB getArea: row "', cell, '"not found (', self.rowLabels,')')
#        print ('area cell',cell,':', out, '(',self.sample,')')
        return out
    
    def setArea (self, cell, value):
        if self.flag:
            self.setValue(self.colIdx, cell, value, silent=True)
        else:
            self.flagCheat = True
            if not hasattr(self, 'data'):
                self.data = {}
            self.data.update({cell: value})
        


# main function
def processJVfolder (folder, ylim=[-50,150], sampleName='', fitDiodeWeight=0, groupCell=True, figAx=None, pltClose=True, newGraphKwargs={}):
    print('Script processJV folder initiated. Data processing can last a few seconds.')
    newGraphKwargs = deepcopy(newGraphKwargs)
    newGraphKwargs.update({'silent': True})
    if figAx is not None:
        pltClose = False
    path = os.path.join(folder, '*.txt')
    
    cellDict = {}
    sampleAreaDict = {}
    outFlag = True
    out0 = ''
    
    graph = Graph('', **newGraphKwargs)
    for file in glob.glob(path):
        fileName, fileExt = os.path.splitext(file)
        fileExt = fileExt.lower()
        line1, line2, line3 = GraphIO.readDataFileLine123(filename=file)
        if GraphJV.isFileReadable(fileName, fileExt, line1=line1, line2=line2, line3=line3):
            try:
                graphTmp = Graph(file, **newGraphKwargs)
            except IndexError as e:
                continue # next file
        else:
            continue
        try:
            sample = graphTmp.curve(-1).sample()
            cell   = graphTmp.curve(-1).cell()
            dark   = graphTmp.curve(-1).darkOrIllum (ifText=True)
            measId = graphTmp.curve(-1).measId()
        except Exception:
            continue # go to next file
        
        print ('file', file)
        if sample == '' or cell == '':
            print('WARNING: cannot identify sample (', sample, ') or cell (', cell, ').')

        if sample != '' and cell != '' :
            if sample not in cellDict:
                cellDict.update ({sample: {}})
            if cell not in cellDict[sample]:
                cellDict[sample].update ({cell: {}})
            if dark not in cellDict[sample][cell]:
                cellDict[sample][cell].update ({dark: {}})
            if sample not in sampleAreaDict:
                sampleAreaDict.update ({sample: areaDB(folder, sample)})
            # open JV file again to correct for cell area
            area = sampleAreaDict[sample].getArea(cell)
            if np.isnan(area) and is_number(graphTmp.curve(0).getAttribute('Acquis soft Cell area')) and not np.isnan(graphTmp.curve(0).getAttribute('Acquis soft Cell area')):
                sampleAreaDict[sample].setArea(cell, graphTmp.curve(0).getAttribute('Acquis soft Cell area'))
                area = sampleAreaDict[sample].getArea(cell)
            cellDict[sample][cell][dark].update({measId: file})
            print (sample, 'cell', cell, 'area', area, 'cm2 (acquired as', graphTmp.curve(0).getAttribute('Acquis soft Cell area'), ')')

        if outFlag :
            out0 = out0 + graphTmp.curve(-1).printShort(header=True)
            outFlag = False


    # sweep through files, identify pairs
    listSample = dictToListSorted(cellDict)
    for s in listSample :
        out      = 'Sample\t' + s + '\n' + 'label\t' + s.replace('_','\\n') + '\n' + deepcopy(out0)
        outIllum = 'Sample\t' + s + '\n' + 'label\t' + s.replace('_','\\n') + '\n' + deepcopy(out0)
        outDark  = 'Sample\t' + s + '\n' + 'label\t' + s.replace('_','\\n') + '\n' + deepcopy(out0)
        listCell = dictToListSorted(cellDict[s])
        for c in listCell :
            listDarkIllum = dictToListSorted(cellDict[s][c])
            # if want to process each emasureemnt independently
            if not groupCell:
                for d in listDarkIllum:
                    listGraph = dictToListSorted(cellDict[s][c][d])
                    for m in listGraph:
                        filesave = 'export_' + s + '_' + c + '_' + m + ('_'+d).replace('_illum','')
                        print ('Graph saved as', filesave)
                        graph = GraphJVDarkIllum (cellDict[s][c][d][m], '', area=sampleAreaDict[s].getArea(c), complement={'ylim':ylim, 'saveSilent': True, '_fitDiodeWeight': fitDiodeWeight}, **newGraphKwargs)
                        out = out + graph.printShort()
                        filesave = os.path.join(folder, filesave)
                        graph.plot(filesave=filesave, figAx=figAx, pltClose=pltClose)
            else:
                # if want to restrict 1 msmt dark + 1 illum per cell
                if len(listDarkIllum) > 2 :
                    print ('test WARNING sorting JV files.')
                # only dark on only illum measurement
                if len(listDarkIllum) == 1 :
                    d = listDarkIllum[0]
                    listGraph = dictToListSorted (cellDict[s][c][d])
                    m = listGraph[0]
                    filesave = 'export_' + s + '_' + c + '_' + m + ('_'+d).replace('_illum','')
                    print ('Graph saved as', filesave)
                    fileDark = cellDict[s][c][d][m] if listDarkIllum[0] == 'dark'  else ''
                    fileIllum= cellDict[s][c][d][m] if listDarkIllum[0] == 'illum' else ''
                    # create Graph file
                    graph = GraphJVDarkIllum(fileDark, fileIllum, area=sampleAreaDict[s].getArea(c), complement={'ylim':ylim, 'saveSilent': True, '_fitDiodeWeight': fitDiodeWeight}, **newGraphKwargs)
                    filesave = os.path.join(folder, filesave)
                    graph.plot(filesave=filesave, figAx=figAx, pltClose=pltClose)
                    # prepare output summary files
                    out = out + graph.printShort()
                    if listDarkIllum[0] == 'dark' :
                        outDark  = outDark  + graph.printShort()
                    else :
                        outIllum = outIllum + graph.printShort()
                    #if len(listGraph) > 1 :
                    #    msg = '.'.join([cellDict[s][c][d][m2] for m2 in cellDict[s][c][d]][1:])
                    #    print ('test WARNING: other files ignored (,',msg,')')
    
                # can identify pair of dark-illum files
                if len(listDarkIllum) == 2 :
                    listGraph = dictToListSorted (cellDict[s][c][listDarkIllum[0]])
                    filesave = 'export_' + s + '_' + c + '_' + listGraph[0]
                    fileDark = cellDict[s][c][listDarkIllum[0]][listGraph[0]]

                    listGraph = dictToListSorted (cellDict[s][c][listDarkIllum[1]])
                    filesave = filesave + '-' + listGraph[0]
                    fileIllum= cellDict[s][c][listDarkIllum[1]][listGraph[0]]

                    print ('Graph saved as', filesave)
                    # create Graph file
                    graph = GraphJVDarkIllum (fileDark, fileIllum, area=sampleAreaDict[s].getArea(c), complement={'ylim':ylim, 'saveSilent': True, '_fitDiodeWeight': fitDiodeWeight}, **newGraphKwargs)
                    filesave = os.path.join(folder, filesave)
                    graph.plot(filesave=filesave, figAx=figAx, pltClose=pltClose)
                    # prepare output summary files
                    out      = out      + graph.printShort()
                    outIllum = outIllum + graph.printShort(onlyIllum=True)
                    outDark  = outDark  + graph.printShort(onlyDark =True)
        # print sample summary
        filesave = 'export_' + s + '_summary' + '.txt'
        filesave = os.path.join(folder, filesave)
        #print('End of JV curves processing, showing summary file...')
        #print(out)
        print('Summary saved in file', filesave, '.')
        f = open(filesave, 'w')
        f.write(out)
        f.close()
        if groupCell:
            filesave = 'export_' + s + '_summary_dark' + '.txt'
            filesave = os.path.join(folder, filesave)
            f = open(filesave, 'w')
            f.write(outDark)
            f.close()
            processSampleCellsMap(filesave, figAx=figAx, pltClose=pltClose)
            filesave = 'export_' + s + '_summary_illum' + '.txt'
            filesave = os.path.join(folder, filesave)
            f = open(filesave, 'w')
            f.write(outIllum)
            f.close()
            processSampleCellsMap(filesave, figAx=figAx, pltClose=pltClose)
            print(writeFileAvgMax(filesave))
    print('Script processJV folder done.')
    return graph



def writeFileAvgMax(fileOrContent, filesave=None, withHeader=True, colSample=True):
    colOfInterest = ['Voc', 'Jsc', 'FF', 'Eff']
    if isinstance(fileOrContent, Graph):
        content = fileOrContent
        filename = content.getAttribute('sample').replace('\n','')
        if filename == '':
            filename = content.getAttribute('label')
    else:
        content = Graph(fileOrContent)
        filename = fileOrContent
#    print(content)
    colLbl = content.getAttribute('collabels')
    cols = []
    idxs = []
    for c in colOfInterest:
        cols.append([np.nan])
        idxs.append(np.nan)
        for i in range(len(colLbl)):
            if c in colLbl[i]:
                cols[-1] = content.curve(i).y()
                idxs[-1] = i
    out = ''
    if withHeader:
        if not colSample:
            out += 'filename\t' + filename + '\n' 
            out += 'Sample\t' + content.getAttribute('sample') + '\n'
        if colSample:
            out += '\t'
        # column headers
        out += 'Parameter average' + '\t'*len(colOfInterest)
        out += 'Best cell (eff.)' + '\t'*len(colOfInterest)
        out += 'Parameter median' + '\t'*len(colOfInterest)
        out += '\n'
        # column name
        if colSample:
            out += 'Sample\t'
        for i in range(3):
            for c in idxs:
                out += (colLbl[c] if c is not np.isnan(c) else '') + '\t'
        out += '\n'
    # averages
    if colSample:
        out += (content.headers['sample'] if 'sample' in content.headers else 'SOMETHING') + '\t'
    for c in cols:
        out += str(np.average(c)) + '\t'
    # best cell
    eff = None
    for i in range(len(colLbl)):
        if 'Eff' in colLbl[i]:
            eff = content.curve(i).y()
    if eff is not None:
        idx = np.argmax(eff)
        for c in cols:
            out += str(c[idx]) + '\t'
    else:
        print('Could not find column Eff')
    # averages
    for c in cols:
        out += str(np.median(c)) + '\t'
    out += '\n'
    # maybe save result in a file
    if isinstance(filename, str) and filesave is True:
        filesave = filename.replace('.txt', '_avgmax.txt')
    if filesave is not None:
        f = open(filesave, 'w')
        f.write(out)
        f.close()
    return out
    
    
    
    
    
def processSampleCellsMap(file, colorscale=None, figAx=None, pltClose=True, newGraphKwargs={}):
    newGraphKwargs = deepcopy(newGraphKwargs)
    newGraphKwargs.update({'silent': True})
    
    content = Graph(file, **newGraphKwargs)
    colToPlot = ['Voc', 'Jsc', 'FF', 'Eff', 'Rp', 'Rs', 'n', 'J0']
    inveScale = [False, False, False, False, False, True, True, True] # inverted color scale
    cols = content.getAttribute('collabels')
    rows = content.getAttribute('rowlabels')
    if not isinstance(cols, list):
        print('Error processSampleCellsMap: cols is not a list (value', cols, ')')
    if colorscale is None:
        if content.getAttribute('colorscale', None) is not None:
            colorscale = content.getAttribute('colorscale')
        if content.getAttribute('cmap', None) is not None:
            colorscale = content.getAttribute('cmap')
    filelist = []
    # combined plot
    graphs = [Graph('', **newGraphKwargs), Graph('', **newGraphKwargs)]
    axisheights = [[],[]]
    # main loop
    for i in range(len(colToPlot)):
        look = colToPlot[i]
        for j in range(len(cols)):
            c = cols[j]
            if c[:len(look)] == look and (len(c) <= len(look) or c[len(look)] in [' ','_','-','.','[',']','(',')']):
                c = c.replace('_', ' [').replace('(','[')
                if '[' in c and ']' not in c:
                    c += ']'
                c = c.replace('[pc]', '[%]').replace('mApcm2', 'mA/cm2')
                # sort somehow identical cells?
                vals = content.curve(j).y()
                filesave = '.'.join(file.split('.')[:-1]) + '_' + look
                filelist.append(filesave + '.txt')

                res = plotSampleCellsMap(rows, vals, c, colorscale=colorscale, filesave=filesave, figAx=figAx, inverseScale=inveScale[i], pltClose=pltClose)

                if isinstance(res, Graph) and res.length() > 0:
                    AB = 0 if i < 4 else 1
                    graphs[AB].append(Curve_Subplot([[0],[0]], {'subplotfile': res.getAttribute('filesave')+'.txt'}))
                    figsize = res.getAttribute('figsize', [6,4])
                    sadjust = res.getAttribute('subplots_adjust', [0.1,0.1,0.9,0.9])
                    graphs[AB].update({'figsize': figsize, 'subplots_adjust': sadjust})
                    axisheights[AB].append(figsize[1] * (sadjust[3]-sadjust[1]))
                break
    # set correct graph size for compiled graph
    bottom, top, hspace = 0.5, 0.5, 0.5
    totalh = [(sum(tmp) + (len(tmp)-1)*hspace + bottom + top) for tmp in axisheights]
    for i in range(2):
        figsize = graphs[i].getAttribute('figsize')
        sadjust = graphs[i].getAttribute('subplots_adjust')
        attr = {'figsize': [figsize[0], totalh[i]]}
        attr.update({'subplots_adjust': [sadjust[0], bottom/totalh[i], sadjust[2], 1-top/totalh[i], 0, hspace/axisheights[i][0]]})
        attr.update({'subplotsncols': 1, 'subplotsheight_ratios': axisheights[i]})
        graphs[i].update(attr)
        filesave = '.'.join(file.split('.')[:-1]) + '_' + ['basic','diode'][i]
        graphs[i].filename = filesave
        graphs[i].plot(filesave=filesave, figAx=figAx)
    return filelist
            
    

def plotSampleCellsMap(cells, values, title, colorscale=None, filesave='', figAx=None, inverseScale=False, pltClose=True, newGraphKwargs={}):
    sizeCell = np.array([0.6, 0.6])
    margin   = np.array([0.4, 0.4])

    newGraphKwargs = deepcopy(newGraphKwargs)
    newGraphKwargs.update({'silent': True})
    
    if len(values) == 0:
        return False

    # check cells are with correct form, i.e. 'a1'
    x, y, val = [], [], []
    split = [research(r'([a-zA-Z])([0-9]*)', c).groups() for c in cells]
    for i in range(len(split)):
        if len(split[i]) == 2:
            x.append(float(ord(split[i][0].lower())-96))
            y.append(float(split[i][1]))
            val.append(values[i]) # prefer to work on a copy and not modifying the list values
    x, y, val = np.array(x), np.array(y), np.array(val)
    if title == 'Voc [V]':
        title = 'Voc [mV]'
        val *= 1000
    title = title.replace('Voc_V','Voc [V]').replace('Jsc_mApcm2','Jsc [mA/cm2]').replace('FF_pc','FF [%]').replace('Eff_pc','Eff. [%]').replace('_Ohmcm2',' [Ohmcm2]')
    title = title.replace('A/cm2', 'A cm$^{-2}$').replace('J0 ','J$_0$ ').replace('Ohmcm2', '$\Omega$ cm$^2$')

    valNorm = val if not (val==0).all() and not len(val)==1 else [0.5]*len(val)
    if isinstance(colorscale, Colorscale):
        colorscale = colorscale.getColorScale()
    if isinstance(colorscale, str): # 'autumn' is a string, but a color list might not have been recognized as a list
        colorscale = stringToVariable(colorscale)
    if colorscale is None:
        colorscale = [[1,0,0], [1,1,0.5], [0,1,0]]
    if inverseScale and isinstance(colorscale, list):
        colorscale = colorscale[::-1]

    xticks = np.arange(0,max(x)+1,1) # if max(x) > 6 else np.arange(0,max(x)+1,1)
    yticks = np.arange(0,max(y)+1,1)
    axSize = np.array([sizeCell[0]*(max(xticks)-min(xticks)), sizeCell[1]*(max(yticks)-min(yticks))])
    figSize = axSize + 2 * margin[1]
    marg = margin / figSize
    txtCoords = np.transpose([(x - 0.5 - min(xticks)) / (max(xticks) - min(xticks)), (y - 0.5 - min(yticks)) / (max(yticks) - min(yticks))])
    toPrint = [roundSignificant(v, 3) for v in val]
    if np.average(val) > 1e2:
        toPrint = ['{:1.0f}'.format(v) for v in toPrint]
    if np.average(val) < 1e-3:
        toPrint = ['{:.1E}'.format(v).replace('E-0','E-') for v in toPrint]
    graph = Graph('', **newGraphKwargs)
    texttxt = []
    textarg = []
    for i in range(len(val)):
        texttxt.append(toPrint[i])
        textarg.append({'xytext': list(txtCoords[i]), 'xycoords': 'axes fraction', 'horizontalalignment':'center', 'verticalalignment':'center'})
    graph.append(Curve([x-0.5, y-0.5], {'type': 'scatter', 'marker': 's', 'markersize': (sizeCell[0]*72)**2, 'markeredgewidth': 0, 'cmap': colorscale}))#cmapParam
    graph.append(Curve([x-0.5, valNorm], {'type': 'scatter_c'}))
    
    graph.update({'subplots_adjust':[marg[0], marg[1], 1-marg[0], 1-marg[1]]})
    graph.update({'figSize': list(figSize)})
    graph.update({'text': texttxt, 'textargs': textarg})
    graph.update({'title': graph.formatAxisLabel(title)})
    graph.update({'xlim': [min(xticks), max(xticks)], 'ylim': [min(yticks), max(yticks)]})
    fct = []
    fct.append(['set_xticks', [list(xticks)], {}])
    fct.append(['set_yticks', [list(yticks)], {}])
    fct.append(['set_xticklabels', [[]], {}])
    fct.append(['set_yticklabels', [[]], {}])
    fct.append(['set_xticks', [list(xticks[1:]-0.5)], {'minor':True}])
    fct.append(['set_yticks', [list(yticks[1:]-0.5)], {'minor':True}])
    fct.append(['set_xticklabels', [[chr(int(i)-1+ord('a')) for i in xticks[1:]]], {'minor': True}])
    fct.append(['set_yticklabels', [[    int(i)             for i in yticks[1:]]], {'minor': True}])
    fct.append(['tick_params', [], {'axis': 'both', 'which': 'minor', 'length': 0}])
    fct.append(['grid', [True], {}])
    graph.update({'arbitraryfunctions': fct})

    if filesave is not None:
        graph.headers.update({'filesave': os.path.basename(filesave)})
        graph.plot(figAx=figAx, filesave=filesave)
    else:
        graph.plot(figAx=figAx, ifSave=False, ifExport=False, ifSubPlot=True)
    if pltClose and figAx is None:
        plt.close()
    return graph


	

if __name__ == "__main__":
    # go through files, store files content in order to later select pairs
    folder = './../examples/JV/SAMPLE_A/'
    processJVfolder(folder, fitDiodeWeight=5, pltClose=False)
#    processJVfolder(folder, groupCell=True, fitDiodeWeight=5, pltClose=False)


    file = r'./../examples/JV\SAMPLE_B_3layerMo\export_sample_b_3layermo_summary_illum.txt'
#    processSampleCellsMap(file, pltClose=False)
#    writeFileAvgMax(file)

