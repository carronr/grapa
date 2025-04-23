# -*- coding: utf-8 -*-
"""Fitting fuctions. For thorough data analyis please use a dedicated XRD software.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os

from grapa.curve import Curve
from grapa.utils.curve_subclasses_utils import FitHandlerBasicFunc

# from grapa.gui.GUIFuncGUI import FuncGUI


class CurveXRD(Curve):
    """CurveXRD offers almost no capabilities for peak fitting of XRD diffractograms.

    The CurveXRD class was mostly to created to test whether it was easy to implement
    the generic fitting routine mechanisms. If you want to learn scientifically from
    your data, you should rather use a dedicated software.
    """

    CURVE = "Curve XRD"

    _PATHXRD = os.path.dirname(os.path.abspath(__file__))

    FITPARAMETERS_FILE = "XRD_fitparameters.txt"
    FITHANDLER = FitHandlerBasicFunc(os.path.join(_PATHXRD, FITPARAMETERS_FILE))

    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({"Curve": CurveXRD.CURVE})
        # could not provide information at instantiation of class variable as CurveXRD
        # was not yet defined
        self.FITHANDLER.set_typecurvefit(CurveXRD)

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # fit-related elements
        out += self.FITHANDLER.funcListGUI(self, **kwargs)
        # help
        out.append([self.print_help, "Help!", [], []])  # one line per function
        self._funclistgui_memorize(out)
        return out

    # Let's comment it out - but keep it in case of useful later
    # def alterListGUI(self):
    #    out = Curve.alterListGUI(self)
    #    return out

    def updateFitParam(self, *param):
        # override default behavior, additional things to do
        param, revert, func = self.FITHANDLER.updateFitParam_before(self, *param)
        # call base function, including parameter func
        out = super().updateFitParam(*param, func=func)
        # revert callable parameter to its initial string value
        self.FITHANDLER.updateFitParam_after(self, revert)
        return out

    def fit_explicit(
        self, roi, *args, funcname="", preset_label="", p0_raw=None, **kwargs
    ):
        """Fit fit_explicit: fit data, for example a peak of a CurveXRD data.
        Pre-set fit parameters are configured in file XRD_fitparameters.txt.

        :param roi: range of interest, in unit of channel
        :param funcname: name of the fit function. Mandatory (provided by GUI)
        :param preset_label: label of the fit preset in the configuration file, if
               relevant (provided by GUI)
        :param p0_raw: p0 values of the fit preset in configuration file, if relevant.
        :param kwargs: to provide custom initial guess p0. Weird syntax for grapa GUI.
               Syntax 1: keywords "p0" and "fixed" are provided. Instances of list.
               Syntax 2: to meet limitations of grapa GUI.
               Example: "0p0": 1.0, "0fixed": True (i.e. is fixed fit parameter),
               "1p0": 2.1, etc. These are later compiled  into lists p0 and fixed
        :return: a CurveXRD fit curve
        """
        return self.FITHANDLER.fit_explicit(
            self,
            roi,
            *args,
            funcname=funcname,
            preset_label=preset_label,
            p0_raw=p0_raw,
            **kwargs
        )
