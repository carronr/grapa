# -*- coding: utf-8 -*-
"""Database os not quite a Graph. Used to store data as table with labels.
NB: not well leveraged within grapa

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import logging

import numpy as np


from grapa.graph import Graph
from grapa.mathModule import is_number
from grapa.utils.error_management import issue_warning


logger = logging.getLogger(__name__)


class Database:
    """Database is not quite a Graph. Used to store data as table with labels.
    Database can be instancied 2 ways:

    - by providing a Graph object

    - by provinding data: np.array, colLabels, rowLabels the labels of the columns,
      respetively the rows, and rowLabelNums a numeric index corresponding the row label.
    """

    def __init__(self, data, colLabels=[], rowLabels=[], rowLabelNums=[]):
        self.rowLabel = "Growth TL nb"
        self.attr = {}
        if isinstance(data, Graph):
            self.colLabels = data.attr("collabels")
            self.rowLabels = data.attr("rowlabels")
            self.rowLabelNums = data.attr("rowlabelnums")
            if data.attr("rowLabel") != "":
                self.rowLabel = data.attr("rowLabel")
            if data[0].shape(1) == 0:
                self.data = np.ndarray([])
            else:
                # assumes all Curves have identical x data
                self.data = np.ndarray((data[0].shape(1), len(data)))
                for i, datai in enumerate(data):
                    self.data[:, i] = datai.y()
                    if not np.array_equiv(datai.x(), data[0].x()):
                        msg = (
                            "Database init: columns with different rows x "
                            "values! (col %s)"
                        )
                        logger.error(msg, i)
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
        for key in attr:
            self.attr.update({key.lower(): attr[key]})

    def getAttr(self, key):
        if key.lower() in self.attr:
            return self.attr[key.lower()]
        return ""

    def shape(self, idx=""):
        """Returns the shape of the np.ndarray data container."""
        if idx == "":
            return self.data.shape
        return self.data.shape[idx]

    def getColLabel(self, i=":"):
        """Returns label of column i."""
        if i == ":":
            return self.colLabels
        return self.colLabels[i]

    def getRowLabel(self, i=":"):
        """Return the row label."""
        if i == ":":
            return self.rowLabels
        return self.rowLabels[i]

    def getRowLabelNum(self, i=":"):
        """Return the numeric equivalent for the row label."""
        if i == ":":
            return self.rowLabelNums
        return self.rowLabelNums[i]

    def renameColumn(self, oldname, newname):
        """Change the label of a column."""
        if is_number(oldname):
            self.colLabels[oldname] = newname
        elif oldname in self.colLabels:
            self.colLabels[self.colLabels.index(oldname)] = newname
        else:
            msg = "database renameColumn: column {} not found, collabels: {}."
            issue_warning(logger, msg.format(newname, self.colLabels))

    def setRowLabel(self, newlabel, i=-1):
        """
        Change the row label. If i=-1 self.rowLabel is modified (default 'Growth TL nb')
        """
        # change numeric rowLabelNum as well ?
        if i == -1:
            self.rowLabel = newlabel
        else:
            self.rowLabels[i] = newlabel

    def deleteRow(self, row_idx):
        self.rowLabels[row_idx] = ""
        for i in range(self.data.shape[1]):
            self.data[row_idx, i] = np.nan

    def merge(self, other):
        # merge rows
        le0 = self.data.shape[0]
        le1 = self.data.shape[1]
        le_ = other.data.shape
        if len(le_) == 1 and len(other.colLabels) == 1:  # if only 1 column
            le_ = le_ + (1,)
        if len(le_) == 1 and len(other.colLabels) > 1:  # if no entry
            le_ = le_ + (len(other.colLabels),)
        newdata = np.zeros((le0 + le_[0], le1 + le_[1]))
        newdata[0:le0, 0:le1] = self.data

        for i in range(other.data.shape[0]):
            try:
                idx = self.rowLabelNums.index(other.rowLabelNums[i])
                newdata = np.delete(newdata, -1, axis=0)
            except ValueError:
                idx = le0
                print(
                    "database merge: new line for index",
                    i,
                    "(",
                    other.rowLabels[i],
                    ")",
                )
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
            if not self.data[i, :].any():
                if self.rowLabels[i] == "":
                    print("Delete index ", i, self.rowLabels[i])
                    self.data = np.delete(self.data, i, axis=0)
                    del self.rowLabels[i]
                    del self.rowLabelNums[i]

    def rowIdFromLabel(self, label):
        try:
            return self.rowLabels.index(label)
        except ValueError:
            logger.error("ValueError, ")

    def colIdFromLabel(self, y):
        return self.colLabels.index(y)

    def value(self, col, row, silent=False):
        i = row if isinstance(row, int) else self.rowIdFromLabel(row)
        [val, collabel] = self.colValuesFromId(col)
        try:
            return val[i]
        except Exception:
            if not silent:
                msg = "Database value: cannot find col {}, row {}, i {}, val {}."
                issue_warning(logger, msg.format(col, row, i, val))
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
                msg = "Database setValue: cannot find col {}, row {}, i {}, value {}"
                print(msg.format(col, row, i, value))
        return 1

    def colValuesFromId(self, x):
        xlabel = self.rowLabel
        if isinstance(x, int):
            xlabel = self.colLabels[x]
            x = self.data[:, x]
        else:
            if x.lower() in ["rowlabelnums", "row"]:
                x = self.rowLabelNums
            elif not is_number(x) and x in self.colLabels:
                xlabel = x
                # print('colValuesFromId', x, self.colLabels.index(x), self.data.shape, self.colLabels)
                x = self.data[:, self.colLabels.index(x)]
            else:
                if not is_number(x):
                    msg = "ERROR colValuesFromId x{}, colLabels {}."
                    logger.error(msg.format(x, self.colLabels))
                xlabel = self.colLabels[x]
                x = self.data[:, x]
        return [np.array(x), xlabel]

    def mask_from_conditions(self, tests):
        # tests can be [['Eff. [%]', '>=', 19], ['row', '<',  2700]]],
        # or [['VOC deficit', '<', 0.420]]
        # can handle several tests, but combination is always AND
        mask = list(np.ones(self.data.shape[0]))
        for test in tests:
            [vals, label] = self.colValuesFromId(test[0])
            for i in range(self.data.shape[0]):
                if test[1] == "==":
                    mask[i] = mask[i] * (vals[i] == test[2])
                if test[1] == ">":
                    mask[i] = mask[i] * (vals[i] > test[2])
                if test[1] == ">=":
                    mask[i] = mask[i] * (vals[i] >= test[2])
                if test[1] == "<":
                    mask[i] = mask[i] * (vals[i] < test[2])
                if test[1] == "<=":
                    mask[i] = mask[i] * (vals[i] <= test[2])
                if test[1] == "!=":
                    mask[i] = mask[i] * (vals[i] != test[2])
        mask2 = []
        for i in range(len(mask)):
            if mask[i] == 1:
                mask2.append(i)
        return mask2

    def plot2(
        self,
        x,
        y,
        linespec="x",
        xlim=[0, 0],
        ylim=[0, 0],
        complement={},
        also=[],
        filesavesuffix="",
    ):
        """plot the data of the Database
        Possible use case: ::

            testEffId1 = [['Eff. [%]', '>=', 19], ['row', '<',  2700]]
            testEffId2 = [['Eff. [%]', '>=', 19], ['row', '>=', 2700], ['row', '<', 2884]]
            testEffId3 = [['Eff. [%]', '>=', 19], ['row', '>=', 2884]]

            def alsoEff(x, y, color):
                return [[x, y, 's'+color, testEffId1],
                        [x, y, 'v'+color, testEffId2],
                        [x, y, 'sr', testEffId3]]

            dbProcess.plot('row',
                           'Eff. [%]',
                           linespec='xk',
                           xlim=[2350, np.ceil(max(dbProcess.rowLabelNums)/50)*50],
                           ylim=[15, 21],
                           also=alsoEff('row', 'Eff. [%]', 'k'))
        """

        def cleanlabelfilename(label):
            return (
                label.replace(" ", "")
                .replace("[", "")
                .replace("]", "")
                .replace("%", "")
                .replace(".", "")
            )

        [x, xlabel] = self.colValuesFromId(x)
        [y, ylabel] = self.colValuesFromId(y)

        if "linespec" not in complement:
            complement.update({"linespec": linespec})
        if "xlabel" not in complement:
            complement.update({"xlabel": xlabel})
        if "ylabel" not in complement:
            complement.update({"ylabel": ylabel})
        if xlim != [0, 0]:
            complement.update({"xlim": xlim})
        if ylim != [0, 0]:
            complement.update({"xlim": ylim})

        if filesavesuffix == "":
            filesavesuffix = "dbPlot_"
        if filesavesuffix[-1] != "_":
            filesavesuffix += "_"
        filesave = (
            "./export/"
            + filesavesuffix
            + cleanlabelfilename(xlabel)
            + "_"
            + cleanlabelfilename(ylabel)
        )
        graph = Graph([x, y], complement, silent=True)
        if also != []:
            # [ [x, y, linespec, [[colIdxCrit, crit, value], [colIdx, crit, value]] ] , second line, curve.]
            for curve in also:
                if curve[0] == "":
                    curve[0] = xlabel
                if curve[1] == "":
                    curve[1] = ylabel
                [x2, xlabel2] = self.colValuesFromId(curve[0])
                [y2, ylabel2] = self.colValuesFromId(curve[1])
                complement2 = {"linespec": curve[2]}
                mask = self.mask_from_conditions(curve[3])
                if len(mask) > 0:
                    graph2 = Graph([x2[mask], y2[mask]], complement2, silent=True)
                    graph.merge(graph2)
                filesave += "_" + cleanlabelfilename(curve[3][0][0])

        if self.getAttr("savesilent"):
            graph.update({"savesilent": True})
        graph.plot(filesave=filesave)

    def plot(self, x, y, linespec="x", xlim=[0, 0], ylim=[0, 0], also=[]):
        """deprecated plot function"""
        complement = {"linespec": linespec, "xlim": xlim, "ylim": ylim}
        self.plot2(x, y, complement=complement, also=also)
