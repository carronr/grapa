# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 19:47:25 2016

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import warnings
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.utils.export import export_filesave_default
from grapa.mathModule import is_number, roundgraphlim

from grapa.datatypes.curveJV import CurveJV


def _check_inverse(file_dark, file_illum):
    # intervert file_illum and file_dark if able to detect inversion
    swap = -1
    if isinstance(file_illum, str) and file_illum.find("dark") > -1:
        swap = 1
    elif isinstance(file_illum, Curve):
        if file_illum.attr("filename").find("dark") > -1:
            swap = 1
    if isinstance(file_dark, str) and file_dark.find("dark") > -1:
        swap = 0
    elif isinstance(file_dark, Curve):
        if file_dark.attr("filename").find("dark") > -1:
            swap = 0
    if swap == 1:
        swap = file_dark
        file_dark = file_illum
        file_illum = swap
    return file_dark, file_illum


class GraphJVDarkIllum(Graph):
    def __init__(
        self,
        file_dark,  # a filename, or a Curve
        file_illum,  # a filename, or a Curve
        area=np.nan,
        temperature=273.15 + 25,
        complement: Optional[dict] = None,
        silent=True,
        **newGraphKwargs
    ):
        if complement is None:
            complement = {}
        # prepare additional attributes
        complement.update({"temperature": temperature})
        complementArea = {}
        if is_number(area):
            complementArea = {"area": area}
        if "area" in complement:
            print(
                "WARNING GraphJVDarkIllum: area defined in argument complement, "
                "beware that area correction might not function properly!"
            )
        # intervert fileIllum and fileDark if able to detect inversion
        file_dark, file_illum = _check_inverse(file_dark, file_illum)

        # start opening files, supposedly the dark
        if isinstance(file_dark, Curve):
            super().__init__("", complement=complement, silent=silent, **newGraphKwargs)
            self.append(file_dark)
        else:
            super().__init__(
                file_dark, complement=complement, silent=silent, **newGraphKwargs
            )

        self.idx_dark = -1
        self.idx_illum = -1
        # fit first file
        if len(self) > 0:
            self[-1].update(complementArea)
            self.idx_dark = 0
            self._curve_append_fit(self[-1], "b", silent)

        # open illuminated file, supposedly the illum
        if isinstance(file_illum, Curve):
            graph = Graph()
            graph.append(file_illum)
        else:
            graph = Graph(
                file_illum, complement=complement, silent=silent, **newGraphKwargs
            )
        # fit second file
        if len(graph) > 0:
            graph[-1].update(complementArea)
            self.idx_illum = len(self)
            self.append(graph[-1])
            self._curve_append_fit(graph[-1], "r", silent)

        # apparent photocurrent: difference between dark and illum
        if len(self) > 2:
            # for this the opening of the 2 file must have been successful
            Jsc = self[self.idx_illum].attr("Jsc")
            idx = [self.idx_illum, self.idx_dark]
            c = [self[idx[0]], self[idx[1]]]
            V = np.sort(np.append(c[0].x(), c[1].x()))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                J = c[0].interpJ(V) - c[1].interpJ(V)
            self.append(
                CurveJV(
                    [V, J],
                    {
                        "color": "g",
                        "label": "Apparent photocurrent",
                        "Jsc": Jsc,
                        "area": self[self.idx_illum].area(),
                    },
                    ifCalc=False,
                    silent=True,
                )
            )
            self[-1].update({"linestyle": "none"})  # by default this curve is not shown
        else:
            pass

    def _curve_append_fit(self, curve, color, silent):
        curve.simpleLabel(forceCalc=True)
        # compute fit, create new curve for it
        curve.update({"color": color})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            curve_fit = curve.CurveJVFromFit(silent=silent)
        if not isinstance(curve_fit, Curve):  # if fit failed
            curve_fit = CurveJV(
                [[0], [np.nan]],
                {"diodeFit": [np.nan] * 5, "mpp": [np.nan] * 3},
                silent=True,
            )
        self.append(curve_fit)

    def printShort(self, header=False, onlyIllum=False, onlyDark=False):
        if header:
            return self[0].printShort(header=True)
        if onlyDark and self.idx_dark >= 0:
            if len(self[self.idx_dark].x()) > 0:
                return self[self.idx_dark].printShort()
        if onlyIllum and self.idx_illum >= 0:
            if len(self[self.idx_illum].x()) > 0:
                return self[self.idx_illum].printShort()
        out = ""
        if self.idx_dark >= 0:
            out = out + self[self.idx_dark].printShort()
        if self.idx_illum >= 0:
            out = out + self[self.idx_illum].printShort()
        return out

    def plot(
        self,
        filesave="",
        ylim=None,
        fig_ax=None,
        if_save=True,
        if_export="auto",
        alter=None,
        pltClose=False,
    ):
        """
        ifExport: Boolean. If 'auto': only exports the lin plot, not the log
        """
        if fig_ax is not None:
            pltClose = False
        # want to handle alter ourself. argument still placed to ensure call call be conducted
        d = self.idx_dark
        i = self.idx_illum
        if ylim is None:
            mM0 = np.array(
                [np.min(self[d].J()), np.max(self[d].J())]
                if d >= 0
                else [np.inf, -np.inf]
            )
            mM2 = np.array(
                [np.min(self[i].J()), np.max(self[i].J())]
                if i >= 0
                else [np.inf, -np.inf]
            )
            if self.attr("ylim", None) is None:
                ylim = [min(mM0[0], mM2[0]), max(mM0[1], mM2[1])]
                #               print ('GraphDarkIllum plot: ylim set to', ylim,'(0',mM0,mM2,')')
                ylim = list(roundgraphlim(ylim))
            #               print ('   ylim', ylim, type(ylim))
            mM0 = np.array(
                [
                    np.min(self[d].y(alter="log10abs")),
                    np.max(self[d].y(alter="log10abs")),
                ]
                if d >= 0
                else [np.inf, -np.inf]
            )
            mM2 = np.array(
                [
                    np.min(self[i].y(alter="log10abs")),
                    np.max(self[i].y(alter="log10abs")),
                ]
                if i >= 0
                else [np.inf, -np.inf]
            )
            ylimLog = [np.floor(min(mM0[0], mM2[0])), np.ceil(max(mM0[1], mM2[1]))]
            ylimLog = list(np.power(10, np.array(ylimLog)))
        # print ('    ylimlog', ylimLog,'(0',mM0,mM2,')')
        # ylimInit = self.attr('ylim')
        ex = True if if_export == "auto" else if_export
        self.plot_std(filesave, ifSave=if_save, ifExport=ex, figAx=fig_ax, ylim=ylim)
        if pltClose:
            plt.close()
        ex = False if if_export == "auto" else if_export
        if len(self) > 2:
            self.plot_diff(
                filesave, ifSave=if_save, ifExport=ex, figAx=fig_ax, ylim=ylim
            )
            if pltClose:
                plt.close()
        self.plot_logabs(
            filesave, ifSave=if_save, ifExport=ex, figAx=fig_ax, ylim=ylimLog
        )
        if pltClose:
            plt.close()
        # self.update({'ylim': ylimInit})

    def plot_std(self, filesave="", ylim=None, figAx=None, ifSave=True, ifExport=True):
        # normal plot
        restore = {}
        if ylim is not None:
            restore.update(self.attr_pop("ylim"))
            self.update({"ylim": ylim})
        self[1].update({"linestyle": "None"})
        if len(self) > 2:
            self[3].update({"linestyle": "None"})
        Graph.plot(
            self,
            filesave=export_filesave_default(self, filesave) + "lin",
            fig_ax=figAx,
            if_save=ifSave,
            if_export=ifExport,
        )
        self[1].update({"linestyle": ""})
        if len(self) > 2:
            self[3].update({"linestyle": ""})
        # restore initial attributes
        self.update(restore)

    def plot_logabs(
        self, filesave="", ylim=None, figAx=None, ifSave=True, ifExport=True
    ):
        # unset xlim, ylim indications
        # change ylabel of the graph
        keys = ["ylim", "xlim", "ylabel", "axhline", "axvline", "alter"]
        restore = {}
        for key in keys:
            restore.update(self.attr_pop(key))
        self.update({"ylabel": "Log current density", "alter": "log10abs"})
        if ylim is not None:
            self.update({"ylim": ylim})
        Graph.plot(
            self,
            filesave=export_filesave_default(self, filesave) + "log",
            fig_ax=figAx,
            if_save=ifSave,
            if_export=ifExport,
        )
        self.update(restore)

    def plot_diff(self, filesave="", ylim=None, figAx=None, ifSave=True, ifExport=True):
        keys = ["axhline", "alter"]
        restore = {}
        for key in keys:
            restore.update(self.attr_pop(key))
        if len(self) > 2:
            Jsc = self[self.idx_illum].attr("Jsc")
            if self.idx_dark >= 0 and self.idx_illum >= 0:
                self.update({"axhline": [Jsc, -Jsc]})
            else:
                self.update({"axhline": [-Jsc]})
        linestIdx = [i for i in [1, 3, 4] if i < len(self)]
        linestyle = [self[i].attr("linestyle") for i in linestIdx]
        if len(self) > 2:
            self[4].update({"linestyle": ""})
        if ylim is not None:
            restore.update(self.attr_pop("ylim"))
            self.update({"ylim": ylim})

        Graph.plot(
            self,
            filesave=export_filesave_default(self, filesave) + "diff",
            fig_ax=figAx,
            if_save=ifSave,
            if_export=ifExport,
        )

        # restore curve properties
        for i in range(len(linestIdx)):
            self[linestIdx[i]].update({"linestyle": linestyle[i]})
        self.update(restore)

    def get_datacurves(self):
        out = []
        if len(self) > 0:  # first curve
            out.append(self[0])
        if len(self) > 2:  # second curve, ignoring the fit which is in position 1
            out.append(self[2])
        return out
