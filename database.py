# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:22:51 2016

@author: car
"""

import numpy as np


from grapa.graph import Graph
from grapa.mathModule import is_number


class Database:

    def __init__(self, data, colLabels=[], rowLabels=[], rowLabelNums=[]):
        """ can be instancied 2 ways:
        - by providing a Graph object
        - by provinding data: np.array, colLabels, rowLabels the labels of the columns, respetively the rows, and rowLabelNums a numeric index corresponding the row label.
        """
        self.rowLabel = 'Growth TL nb'
        self.attr = {}
        if isinstance(data, Graph):
            self.colLabels = data.getAttribute('collabels')
            self.rowLabels = data.getAttribute('rowlabels')
            self.rowLabelNums = data.getAttribute('rowlabelnums')
            if data.getAttribute('rowLabel') != '':
                self.rowLabel = data.getAttribute('rowLabel')
            if data.curve(0).shape(1) == 0:
                self.data = np.ndarray([])
            else:
                # assumes all xyCurves have identical x data
                self.data = np.ndarray((data.curve(0).shape(1), data.length()))
                for i in range(data.length()):
                    self.data[:, i] = data.curve(i).y()
                    if not np.array_equiv(data.curve(i).x(), data.curve(0).x()):
                        print('ERROR database init: columns with different rows x values! (col', i, ')')
#                        for j in range(len(data.curve(i).x())):
#                            if data.curve(i).x(j) != data.curve(0).x(j):
#                                print ('   ', j, data.curve(i).x(j), data.curve(0).x(j))
        else:
            self.data = np.array(data)
            self.colLabels = colLabels
            self.rowLabels = rowLabels
            if rowLabelNums == []:
                self.rowLabelNums = self.rowLabels[:]
            else:
                self.rowLabelNums = rowLabelNums
            for i in range(len(self.rowLabelNums)):
                try:
                    self.rowLabelNums[i] = float(self.rowLabelNums[i])
                except Exception:
                    self.rowLabelNums[i] = np.nan

    def update(self, attr):
        for key in attr :
            self.attr.update({key.lower(): attr[key]})
    def getAttr(self, key):
        if key.lower() in self.attr:
            return self.attr[key.lower()]
        return ''

    def shape(self, idx=''):
        """ Returns the shape of the np.ndarray data container. """
        if idx == '':
            return self.data.shape
        return self.data.shape[idx]

    def getColLabel(self, i=':'):
        """ Returns label of column i. """
        if i == ':':
            return self.colLabels
        return self.colLabels[i]

    def getRowLabel(self, i=':'):
        """ Return the row label. """
        if i == ':':
            return self.rowLabels
        return self.rowLabels[i]

    def getRowLabelNum(self, i=':'):
        """ Return the numeric equivalent for the row label. """
        if i == ':':
            return self.rowLabelNums
        return self.rowLabelNums[i]

    def renameColumn(self, oldname, newname):
        """ Change the label of a column. """
        if is_number(oldname):
            self.colLabels[oldname] = newname
        elif oldname in self.colLabels:
            self.colLabels[self.colLabels.index(oldname)] = newname
        else:
            print('WARNING database renameColumn: column', newname, ' not found.')
            print('   ', self.colLabels)

    def setRowLabel(self, newlabel, i=-1):
        """ Change the row label. If i=-1 self.rowLabel is modified (default 'Growth TL nb')  """
# change numeric rowLabelNum as well ?
        if i == -1:
            self.rowLabel = newlabel
        else:
            self.rowLabels[i] = newlabel



    def deleteRow(self, row_idx):
        self.rowLabels[row_idx] = ''
        for i in range(self.data.shape[1]):
            self.data[row_idx, i] = np.nan



    def merge(self, other):
        # merge rows
        le0 = self.data.shape[0]
        le1 = self.data.shape[1]
        le_ = other.data.shape
        if len(le_) == 1 and len(other.colLabels) == 1: # if only 1 column
            le_ = le_ + (1,)
        if len(le_) == 1 and len(other.colLabels) > 1: # if no entry
            le_ = le_ + (len(other.colLabels),)
        newdata = np.zeros((le0 + le_[0], le1 + le_[1]))
        newdata[0:le0, 0:le1] = self.data

        for i in range(other.data.shape[0]):
            try:
                idx = self.rowLabelNums.index(other.rowLabelNums[i])
                newdata = np.delete(newdata, -1, axis=0)
            except ValueError:
                idx = le0
                print('database merge: new line for index', i, '(', other.rowLabels[i], ')')
                self.rowLabels.append(other.rowLabels[i])
                self.rowLabelNums.append(other.rowLabelNums[i])
                le0 += 1
            if len(other.data.shape) == 1:
                newdata[idx, le1:] = other.data[i]
            else:
                newdata[idx, le1:] = other.data[i, :]
        self.data = newdata
        for col in other.colLabels:
            self.colLabels.append(col)





    def clean(self):
        for i in range(self.data.shape[0]):
            if (not self.data[i, :].any()):
                if self.rowLabels[i] == '':
                    print('Delete index ', i, self.rowLabels[i])
                    self.data = np.delete(self.data, i, axis=0)
                    del self.rowLabels[i]
                    del self.rowLabelNums[i]

    def rowIdFromLabel(self, y):
        return self.rowLabels.index(y)

    def colIdFromLabel(self, y):
        return self.colLabels.index(y)

    def value(self, col, row, silent=False):
        i = row if isinstance(row, int) else self.rowIdFromLabel(row)
        [val, collabel] = self.colValuesFromId(col)
        try:
            return val[i]
        except Exception:
            if not silent:
                print('Database value: cannot find col, row:', col, row, i, val)
        return np.nan

    def setValue(self, col, row, value, silent=False):
        i = row if isinstance(row, int) else self.rowIdFromLabel(row)
        if not isinstance(col, int):
            col = self.colIdFromLabel(col)
        try:
            self.data[i, col] = value
            return 0
        except Exception:
            if not silent:
                print('Database setValue: cannot find col, row:', col, row, i, value)
        return 1


    def colValuesFromId(self, x):
        xlabel = self.rowLabel
        if isinstance(x, int):
            xlabel = self.colLabels[x]
            x = self.data[:, x]
        else:
            if x.lower() in ['rowlabelnums', 'row']:
                x = self.rowLabelNums
            elif not is_number(x) and x in self.colLabels:
                xlabel = x
#                print('colValuesFromId', x, self.colLabels.index(x), self.data.shape, self.colLabels)
                x = self.data[:, self.colLabels.index(x)]
            else:
                if not is_number(x):
                    print('ERROR colValuesFromId', x, self.colLabels)
                xlabel = self.colLabels[x]
                x = self.data[:, x]
        return [np.array(x), xlabel]




    def maskFromConditions(self, tests):
        # tests can be [['Eff. [%]', '>=', 19], ['row', '<',  2700]]],
        # or [['VOC deficit', '<', 0.420]]
        # can handle several tests, but combination is always AND
        mask = list(np.ones(self.data.shape[0]))
        for test in tests:
            [vals, label] = self.colValuesFromId(test[0])
            for i in range(self.data.shape[0]):
                if test[1] == '==':
                    mask[i] = mask[i] * (vals[i] == test[2])
                if test[1] == '>':
                    mask[i] = mask[i] * (vals[i] >  test[2])
                if test[1] == '>=':
                    mask[i] = mask[i] * (vals[i] >= test[2])
                if test[1] == '<':
                    mask[i] = mask[i] * (vals[i] <  test[2])
                if test[1] == '<=':
                    mask[i] = mask[i] * (vals[i] <= test[2])
                if test[1] == '!=':
                    mask[i] = mask[i] * (vals[i] != test[2])
        mask2 = []
        for i in range(len(mask)):
            if mask[i] == 1:
                mask2.append(i)
        return mask2





    def plot2(self, x, y, linespec='x', xlim=[0, 0], ylim=[0, 0], complement={}, also=[], filesavesuffix=''):
        """ plot the data of
        Possible call:
            testEffId1 = [['Eff. [%]', '>=', 19], ['row', '<',  2700]]
            testEffId2 = [['Eff. [%]', '>=', 19], ['row', '>=', 2700], ['row', '<', 2884]]
            testEffId3 = [['Eff. [%]', '>=', 19], ['row', '>=', 2884]]
            def alsoEff(x, y, color):
                return [[x, y, 's'+color, testEffId1], [x, y, 'v'+color, testEffId2], [x, y, 'sr', testEffId3]]
            dbProcess.plot('row',          'Eff. [%]',     linespec='xk', xlim=[2350, np.ceil(max(dbProcess.rowLabelNums)/50)*50], ylim=[15, 21],  also=alsoEff('row',          'Eff. [%]', 'k'))
        """
        def cleanlabelfilename (label):
            return label.replace(' ', '').replace('[', '').replace(']', '').replace('%', '').replace('.', '')
            
        [x, xlabel] = self.colValuesFromId(x)
        [y, ylabel] = self.colValuesFromId(y)

        if 'linespec' not in complement:
            complement.update({'linespec': linespec})
        if 'xlabel' not in complement:
            complement.update({'xlabel': xlabel})
        if 'ylabel' not in complement:
            complement.update({'ylabel': ylabel})
        if xlim != [0, 0]:
            complement.update({'xlim': xlim})
        if ylim != [0, 0]:
            complement.update({'xlim': ylim})

        if filesavesuffix == '':
            filesavesuffix = 'dbPlot_'
        if filesavesuffix[-1] != '_':
            filesavesuffix += '_'
        filesave = './export/' + filesavesuffix + cleanlabelfilename(xlabel) + '_' + cleanlabelfilename(ylabel)
        graph = Graph([x, y], complement, silent=True)
        if also != []:
            #[ [x, y, linespec, [[colIdxCrit, crit, value], [colIdx, crit, value]] ] , second line, curve.]
            for curve in also:
                if curve[0] == '':
                    curve[0] = xlabel
                if curve[1] == '':
                    curve[1] = ylabel
                [x2, xlabel2] = self.colValuesFromId(curve[0])
                [y2, ylabel2] = self.colValuesFromId(curve[1])
                complement2 = {'linespec': curve[2]}
                mask = self.maskFromConditions(curve[3])
                if len(mask) > 0:
                    graph2 = Graph([x2[mask], y2[mask]], complement2, silent=True)
                    graph.merge(graph2)
                filesave += '_' + cleanlabelfilename(curve[3][0][0])

        if self.getAttr('savesilent') :
            graph.update({'savesilent': True})
        graph.plot(filesave=filesave)





    def plot(self, x, y, linespec='x', xlim=[0, 0], ylim=[0, 0], also=[]):
        """ deprecated plot function """
        self.plot2(x, y, complement={'linespec':linespec, 'xlim': xlim, 'ylim': ylim}, also=also)
#        [x, xLabel] = self.colValuesFromId(x)
#        [y, yLabel] = self.colValuesFromId(y)
#
#        fig,ax = plt.subplots()
#        fig.patch.set_alpha(0.0)
#
#        ax.plot(x, y, linespec)
#        ax.ticklabel_format(useOffset=False)
#        ax.set_ylabel(yLabel)
#        ax.set_xlabel(xLabel)
#        if ylim != [0,0]:
#            ax.set_ylim(bottom=ylim[0])
#            ax.set_ylim(top   =ylim[1])
#        if xlim != [0,0]:
#            ax.set_xlim(left= xlim[0])
#            ax.set_xlim(right=xlim[1])
#
#        if also != []:
#            #[ [x, y, linespec, [[colIdxCrit, crit, value], [colIdx, crit, value]] ] , second line, curve.]
#            for curve in also:
#                [x, xLabel] = self.colValuesFromId(curve[0])
#                [y, yLabel] = self.colValuesFromId(curve[1])
#                linespec = curve[2]
#                mask = self.maskFromConditions(curve[3])
#                if len(mask) > 0:
#                    ax.plot(x[mask], y[mask], linespec)
#                else:
#                    print('plot also: no data to plot')
#
#        filename = 'databasePlot_' + xLabel + '_' + yLabel
#        filesave = filename + '.png'
#        plt.savefig(filesave, transparent=True)
#
#        out = ''
#        out+= 'xlabel\t' + xLabel + '\n'
#        out+= 'ylabel\t' + yLabel + '\n'
#        out+= 'linespec\t' + linespec + '\n'
#        for i in range(len(x)):
#            out += str(x[i]) + '\t' + str(y[i]) + '\n'
#        f = open(filename + '.txt', 'w')
#        f.write(out)
#        f.close()
#
