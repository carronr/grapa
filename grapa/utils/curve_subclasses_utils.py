# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""


import copy
import inspect
import logging

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import numpy as np

from grapa.curve import Curve
from grapa.graph import Graph
from grapa.utils.funcgui import FuncGUI
from grapa.utils.error_management import issue_warning
from grapa.mathModule import is_number, roundSignificant

logger = logging.getLogger(__name__)


class FileLoaderOnDemand:
    """A class to handle loading data from external files (e.g. configuration
    information, reference datasets), that we do not want to parse systematically at
    run-time, only once, on demand when actually needed.
    The instance can be called (), accessed by index [], and .get("label abc").
    Also, .labels() provide list of possible labels (or other key)"""

    def __init__(self, filename, call_index=None):
        """
        filename: file where data are located
        call_index: call on object instance will return Object[call_index]
        """
        self.filename = filename
        self.content: Graph = None
        self.call_index = call_index

    def _load_if_not_already(self):
        if self.content is None:
            self.content = Graph(self.filename, complement={"readas": "generic"})
            # NB: force read the file parsing method to "generic", otherwise risk of
            # infinite loop at loading

    def __getitem__(self, index):
        """Returns Curve at given index"""
        self._load_if_not_already()
        try:
            return self.content.__getitem__(index)
        except IndexError as e:
            if len(self.content) == 0:
                logger.error("FileLoaderOnDemand: Hey, no data in the file indicated!!")
            # msg = "FileLoaderOnDemand: index {} not found in file {} (len {})"
            # print(msg.format(index, self.filename, len(self.content)))
            raise e

    def __call__(self):
        """Call on Instance: returns Curve at index call_index"""
        if self.call_index is not None:
            return self[self.call_index]
        self._load_if_not_already()
        if len(self.content) == 1:  # let's be helpful, return if only 1 Curve available
            return self.content[0]

        msg = (
            "FileLoaderOnDemand: cannot call Object, because call_index was not"
            "defined at instantiation and more than one possibility. (file %s)."
        )
        logger.error(msg, self.filename)
        raise RuntimeError(msg % self.filename)

    def get(self, value, key="label"):
        """Returns the first Curve with key==value"""
        self._load_if_not_already()
        for curve in self.content:
            if curve.attr(key) == value:
                return curve

        msg = "FileLoaderOnDemand: cannot find curve in file {} with {} = {}."
        issue_warning(logger, msg.format(self.filename, key, value))
        return None

    def labels(self, key="label"):
        """Returns the list of possible labels corresponding to each Curve in content"""
        self._load_if_not_already()
        return [curve.attr(key) for curve in self.content]


class FitterFixed:  # pylint: disable=too-few-public-methods
    """Wrap-up to be able to fit a function with fixed parameters
    (written in the old days, maybe python evolved, so it is not useful anymore?)"""

    @staticmethod
    def _func_caller(func, x, p0, fixed, *var):
        params = copy.deepcopy(p0)
        k = 0
        for i, fixed_i in enumerate(fixed):
            if not fixed_i:
                params[i] = var[k]
                k += 1
        return func(x, *params)

    @staticmethod
    def _popt(poptintern, p0, fixed):
        popt = list(p0)
        j = 0
        for i, fixed_i in enumerate(fixed):
            if not fixed_i:
                popt[i] = poptintern[j]
                j += 1
        return popt

    @classmethod
    def curve_fit_fixed(cls, func, datax, datay, p0, fixed):
        """wrap-up to be able to fit a function with fixed parameters"""
        p0intern = [p0[i] for i in range(len(fixed)) if not fixed[i]]
        # print("fit fixed", p0, fixed, p0intern)
        # f = lambda x, *var: cls._func_caller(func, x, p0, fixed, *var)
        poptintern, pcov = curve_fit(
            lambda x, *var: cls._func_caller(func, x, p0, fixed, *var),
            datax,
            datay,
            p0=p0intern,
        )
        popt = cls._popt(poptintern, p0, fixed)
        return popt, poptintern, pcov


class FitHandler:
    """
    This class should work as a minimum version, although it has no fit function
    implemented - see below shild class FitHandlerBasicFunc to see how to do that.

    The configuration file is a grapa-style text file, with following attributes (lines
    of parameters). See example grapa/datatypes/XRF_fitparameters.txt. Each column
    is a fit "preset", intended to be ready to be used. Required attributes:

    - label: used for referencing the preset

    - roi: list [valuemin, valuemax], range of interest, in units of the data

    - function: str, name of the fit function, that must be impolemented in a
      child class. Function names must start with `func_`

    - p0: a list, default initial guess for fit parameters. Presumably numeric.
      str are also possible. For example '1/350', see function p0_interpret_value; or
      "Background" that will return a interp1d interpolation function over the data
      stored in preset curve with label "Background".

    - fixed: a list of booleans, True if parameter is fixed by default

    - showingui if the preset is shown in the gui, by default

    - sum_amp_gaussian: not needed, specific to CurveMCA.

    Then, create a subclass containing the fit functions:

    - fit functions must be named starting with `func_`

    - also: overrides needed for additional behaviors (e.g. amplitude in CurveMCA)

    - also: overrides to add e.g. interpretation of text into numerical values

    - other needed behaviors

    For implementation into Curve subclass, take care of notably:

    - class variable - FitHandler, or rather subclass of if containing the fit functions
      of interest

      .. code-block:: python

         FITHANDLER = FitHandler(os.path.join(_PATHXRD, FITPARAMETERS_FILE))

    - __init__:

      .. code-block:: python

         # in __init__, as class itself not defined at instantiation of class variable
         self.FITHANDLER.set_typecurvefit(CurveXRD)  # to adapt with own Curve type

    - funcListGUI

      .. code-block:: python

         # fit-related elements
         out += self.FITHANDLER.funcListGUI(self, **kwargs)

    - updateFitParam, to provide fit function to Curve.updateFitParam, and
      in case textual keywords are used in fit parameters for e.g. "Background".

      .. code-block:: python

         def updateFitParam(self, *param):
             # override default behavior, additional things to do
             param, revert, func = self.FITHANDLER.updateFitParam_before(self, *param)
             # call base function, including parameter func
             out = super().updateFitParam(*param, func=func)
             # revert callable parameter to its initial string value
             self.FITHANDLER.updateFitParam_after(self, revert)
             return out

    - fit_explicit, so curve action can point curve.fit_explicit and not more complex
      dependency, and to host the docstring to report in printHelp()

      .. code-block:: python

         def fit_explicit(self, roi, *args, funcname="", preset_label="", p0_raw=None,
                          **kwargs):
             # docstring to write here between triple double-quotes
             return self.FITHANDLER.fit_explicit(self, roi, *args, funcname=funcname,
                                    preset_label=preset_label, p0_raw=p0_raw, **kwargs)

    - printHelp
    """

    def __init__(self, filename):
        # filename: where fit presets are stored
        self.database = FileLoaderOnDemand(filename, 0)
        self.gui_showselector = True
        self.fitcurve_attr_to_copy = ["offset", "muloffset"]
        self.typecurvefit = Curve

    # some getters and setters
    def set_guishowfitfunc(self, which, fitimmediately, curve: Curve = None):
        """To make GUI show specific fit functions

        :param which: label of the function to show
        :param fitimmediately: if True, fit immediately
        :param curve: must be provided
        """
        if isinstance(which, (list, tuple, str)):
            which = str(which)
        curve.update({"_guishowfunc": which})
        if bool(fitimmediately) and len(which) > 0:
            showlist = self.get_presets(which)
            if len(showlist) == 1:
                return self.fit_preset(curve, showlist[0][1].attr("label"))

            if len(showlist) > 1:
                msg = (
                    "FitHandler: cannot fit preset '{}' because several possible "
                    "matches ({})"
                )
                print(msg.format(which, [el[1].attr("label") for el in showlist]))
            elif len(showlist) == 0:
                msg = "No fit parameters defined with label '{}' (possibilities: {})"
                print(msg.format(which, self.database.labels()))
        return True

    def set_typecurvefit(self, typecurvefit):
        """set the type of Curve to be fitted"""
        if self.typecurvefit == typecurvefit:
            return  # no change
        if issubclass(typecurvefit, Curve):
            if self.typecurvefit != Curve:
                msg = (
                    "WARNING FitHandler.set_typecurvefit: type of output fit curves "
                    "changed from {} to {}."
                )
                print(msg.format(self.typecurvefit, typecurvefit))
            self.typecurvefit = typecurvefit
        else:
            msg = "FitHandler.set_typecurvefit: typecurvefit must be a subclass of Curve. Ignored."
            issue_warning(logger, msg)

    def get_fitfunction(self, funcname):
        """fit functions must have names starting with `func_`, and be implemented in a
        child class"""
        if funcname.startswith("func_") and hasattr(self, funcname):
            return getattr(self, funcname)

        msg = "Fit function (%s) does not seem implemented. Abort."
        logger.error(msg, funcname)
        raise NotImplementedError(msg % funcname)

    def get_preset(self, label):
        """Returns first preset which label matches label"""
        if label not in self.database.labels():
            msg = "No fit parameters defined with label: {} (possibilities: {})"
            issue_warning(logger, msg.format(label, self.database.labels()))
            return False

        preset = self.database.get(label)
        return preset

    def get_presets(self, label, startswith=True, fitdefined=True):
        """tolerant input, useful for GUI Curve Actions
        Returns a list of (idx, preset Curve)"""
        # if label is list: recursive call on each element
        if isinstance(label, list):
            out = []
            for lbl in label:
                news = self.get_presets(
                    lbl, startswith=startswith, fitdefined=fitdefined
                )
                for new in news:
                    if new not in out:
                        out.append(new)
            return out
        # solve for individual inputs
        label = str(label)
        idxs, presets = [], []
        labels = self.database.labels()
        if startswith:
            match = [f.startswith(label) for f in labels]
        else:
            match = [(f == label) for f in labels]
        for idx, value in enumerate(match):
            if not value:
                continue
            preset = self.database[idx]
            if fitdefined and len(str(preset.attr("function"))) == 0:
                continue  # no fit function defined. Possibly is Background
            if idx not in idxs:
                idxs.append(idx)
                presets.append(preset)
        return list(zip(idxs, presets))

    # helpers to Curve
    def funcListGUI_ifnewline(self, counter, _lbl, _func):
        """by dfault, new line every 4 parameters"""
        if counter > 4:
            return True
        return False

    def funcListGUI_notafit_choicepresets(self, curve, **_kwargs):
        """if not a fit, show fit options"""
        out = []
        # list of fit functions
        if self.gui_showselector:
            showfitfunc = curve.attr("_guishowfunc")
            presetslabels = self.database.labels()
            line = FuncGUI(self.set_guishowfitfunc, "Show fit preset")
            line.set_hiddenvars({"curve": curve})
            line.appendcbb("label", showfitfunc, presetslabels)
            line.append("and fit", True, options={"field": "Checkbutton"})
            out.append(line)
        return out

    def funcListGUI_notafit_whichones(self, curve, **_kwargs):
        """Chooses which preset to show"""
        showfitfunc = curve.attr("_guishowfunc")
        showlist = []
        if len(showfitfunc) > 0:
            showlist = self.get_presets(showfitfunc)
        # if no user choice, go for default as configured
        if len(showlist) == 0:
            showinguis = self.database.labels(key="showingui")
            mask = np.array(showinguis, bool)
            showidx = list(np.arange(len(showinguis))[mask])
            showpresets = [self.database[idx] for idx in showidx]
            showlist = zip(showidx, showpresets)
        return showlist

    def funcListGUI_notafit_preset(self, curve, preset, **_kwargs):
        """Fit using preset settings"""
        out = []
        label = preset.attr("label")
        roi = preset.attr("roi")
        p0, _ = self.p0_interpret(preset.attr("p0"), curve, interp=False)
        fixed = preset.attr("fixed")
        funcname = preset.attr("function")
        func = self.get_fitfunction(funcname)
        func_args = [str(a) for a in inspect.signature(func).parameters]
        docstring = func.__doc__.strip()
        docstring = "\n".join([line.strip() for line in docstring.split("\n")])
        line = FuncGUI(
            curve.fit_explicit,
            "Fit {}".format(label),
            hiddenvars={"funcname": funcname, "preset_label": label},
            tooltiptext=docstring,
        )
        line.append("ROI", roi, options={"width": 9})
        counterspace = 0
        for i in range(len(p0)):
            space = ""
            counterspace += 1
            lbl = func_args[i + 1]  # +1 because self
            if self.funcListGUI_ifnewline(counterspace, lbl, func):
                line.append("", "", "Frame")
                space = "      "
                counterspace = 0
            width = 10 if isinstance(p0[i], str) else 6
            optsp0 = {"width": width, "keyword": "{}p0".format(i)}
            if fixed[i]:
                optsp0.update({"state": "readonly"})
            optsfi = {
                "keyword": "{}fixed".format(i),
                "bind": "previouswidgettogglereadonly",
            }
            line.append(space + lbl, p0[i], options=optsp0)
            line.append("", fixed[i], "Checkbutton", options=optsfi)
        out.append(line)
        return out

    def funcListGUI_isafit(self, curve, **_kwargs):
        """If is a fit: updateFitParam, resamplex"""
        out = []
        label = curve.attr("_fit_presetlabel")
        preset = self.database.get(label)
        funcname = preset.attr("function")
        func = self.get_fitfunction(funcname)
        func_args = [str(a) for a in inspect.signature(func).parameters]
        popt = curve.attr("_popt")
        line = FuncGUI(curve.updateFitParam, "Update fit")
        counterspace = 0
        for i in range(len(popt)):
            space = ""
            counterspace += 1
            lbl = func_args[i + 1]
            if self.funcListGUI_ifnewline(counterspace, lbl, func) and i != 0:
                line.append("", "", "Frame")
                space = "      "
                counterspace = 0
            line.append(space + lbl, popt[i])
        out.append(line)
        # resample x
        x = curve.x()
        lst = roundSignificant([x[0], x[-1], (x[1] - x[0])], 5)
        line = FuncGUI(self.fit_resamplex, "Resample x", hiddenvars={"curve": curve})
        line.append("delta value, or [xmin, xmax, step]", lst)
        out.append(line)
        return out

    def funcListGUI(self, curve, **kwargs):
        """List of options for GUI"""
        out = []
        isfitfunc = curve.has_attr("_popt")
        # fit functions
        if not isfitfunc:
            # list of fit functions
            out += self.funcListGUI_notafit_choicepresets(curve, **kwargs)
            # display chosen fit functions
            showlist = self.funcListGUI_notafit_whichones(curve, **kwargs)
            for showelement in showlist:
                out += self.funcListGUI_notafit_preset(curve, showelement[1], **kwargs)
        else:
            out += self.funcListGUI_isafit(curve, **kwargs)
        return out

    def fit_resamplex(self, deltax, curve):
        """Change the base of datapoints a fit result is displayed on"""
        # let's not make that method static for clarity reason: it modifies object curve
        new = ""
        if isinstance(deltax, float):
            deltax = np.abs(deltax)
            new = np.arange(curve.x(0), curve.x(-1) + deltax, deltax)
        if isinstance(deltax, list):
            if len(deltax) == 3:
                new = np.arange(deltax[0], deltax[1], deltax[2])
            else:
                new = np.array(deltax)
        # if valid input, do something
        if len(new) > 0 or new != "":
            # recreate data container, with new x values both in x and y
            # positions. next step it to compute new y values
            curve.set_data(np.array([new, new]))
            curve.updateFitParam(*curve.attr("_popt"))
            return 1
        return "Invalid input."

    def updateFitParam_before(self, curve, *popt):
        """Small bit of code performed before updateFitParam"""
        param, revert = self.p0_interpret(popt, curve)
        funcname = curve.attr("_fitfunc")
        func = self.get_fitfunction(funcname)
        return param, revert, func

    def updateFitParam_after(self, curve, revert):
        """Small bit of code performed after updateFitParam"""
        if len(revert) > 0:
            popt = curve.attr("_popt")
            for key in revert:
                popt[key] = revert[key]
            curve.update({"_popt": popt})

    # mechanics of the fitting
    def _p0_interpret_value(self, value, _curve, interp):
        if "/" in value:
            num, denom = value.split("/")
            return float(num) / float(denom)

        if "*" in value:
            q1, q2 = value.split("*")
            return float(q1) * float(q2)

        if "+" in value:
            q1, q2 = value.split("+")
            return float(q1) + float(q2)

        if value in self.database.labels():
            if interp:
                # note: then, fixed[i] has to be True
                bgcurve = self.database.get(value)
                return interp1d(bgcurve.x(), bgcurve.y())

        if interp:
            msg = "FitHandler._p0_interpret_value Cannot find data for input '%s' %s"
            msa = (value, interp)
            logger.error(msg, *msa)
            raise NotImplementedError(msg % msa)
        return value

    def p0_interpret(self, p0, curve=None, interp=True):
        """Returns a list of parameters, after interpreting textual values e.g.
        contaiing +, * or /, or background curves"""
        p0 = list(p0)
        revert = {}
        for i in range(len(p0)):
            value = p0[i]
            if isinstance(value, str):
                revert.update({i: value})
                p0[i] = self._p0_interpret_value(value, curve, interp)
        return p0, revert

    @staticmethod
    def _kwargs_to_p0_fixed(kwargs):
        if "p0" in kwargs and "fixed" in kwargs:
            p0, fixed = kwargs["p0"], kwargs["fixed"]
            if isinstance(p0, list) and isinstance(fixed, list):
                return kwargs["p0"], kwargs["fixed"]

            issue_warning(logger, "fit retrieve inputs: p0 or fixed not lists. Ignore.")
        i = 0
        p0_raw, fixed = [], []
        while True:
            ip0, ifi = "{}p0".format(i), "{}fixed".format(i)
            if ip0 in kwargs and ifi in kwargs:
                p0_raw.append(kwargs[ip0])
                fixed.append(kwargs[ifi])
                i += 1
            else:
                break
        return p0_raw, fixed

    # functions performing the fit, and wrapper interface functions
    def fit(self, curve, func, roi, p0, fixed):
        x, y = curve.x(), curve.y()
        mask = (x >= np.min(roi[0])) * (x <= np.max(roi))
        datax = x[mask]
        datay = y[mask]
        try:
            popt, _poptintern, _pcov = FitterFixed.curve_fit_fixed(
                func, datax, datay, p0=p0, fixed=fixed
            )
        except RuntimeError:
            msg = "Exception RuntimeError fit, curve {}"
            issue_warning(None, msg.format(curve.attr("label")), exc_info=True)
            curvefit = self.typecurvefit([[0], [0]], {})
            for key in self.fitcurve_attr_to_copy:
                curvefit.update({key: curve.attr(key)})
            return [np.nan] * len(p0), curvefit

        attrs = {
            "_fitfunc": func.__name__,
            "_popt": popt,
            "_fit_fixed": fixed,
            "_fit_roi": roi,
        }
        curvefit = self.typecurvefit([datax, func(datax, *popt)], attrs)
        for key in self.fitcurve_attr_to_copy:
            curvefit.update({key: curve.attr(key)})
        return popt, curvefit

    def fit_preset(self, curve, preset_label, roi="auto", p0="auto", fixed="auto"):
        """
        Fit fit_preset: fit a Curve, for example a peak of a CurveMCA data.
        Pre-set fit parameters are configured in file XRF_fitparameters.txt.

        :param curve: a Curve object contaiing the data to fit
        :param preset_label: see row "label" in config file. e.g. "Cu Ka1,2", "Ag Ka1,2"
        :param roi: range of interest, in unit of channel. Overwrite preset.
        :param p0: list, to provide custom initial guess p0. Overwrite preset.
        :param fixed: list of boolean, to provide custom initial guess p0. Overwrite
               reset.
        :return: a CurveMCA fit curve
        """
        preset = self.get_preset(preset_label)
        if not isinstance(preset, Curve):
            msg = "CurveMCA.fit_preset cannot find desired preset in config file ({})."
            raise RuntimeError(msg.format(preset_label))
        funcname = preset.attr("function")
        if roi == "auto":
            roi = preset.attr("roi")
        p0_raw = preset.attr("p0") if p0 == "auto" else list(p0)
        p0, _ = self.p0_interpret(p0_raw, curve)
        if fixed == "auto":
            fixed = preset.attr("fixed")
        sum_amp_gaussian = preset.attr("sum_amp_gaussian")
        kwargs = {
            "funcname": funcname,
            "p0_raw": p0_raw,
            "preset_label": preset_label,
            "sum_amp_gaussian": sum_amp_gaussian,
        }
        for i in range(len(p0)):
            kwargs.update({"{}p0".format(i): p0[i], "{}fixed".format(i): fixed[i]})
        return self.fit_explicit(curve, roi, **kwargs)

    def fit_explicit(
        self, curve, roi, *_args, funcname="", preset_label="", p0_raw=None, **kwargs
    ):
        """
        Fit fit_explicit: fit data, for example a peak of a CurveMCA data.
        Pre-set fit parameters are configured in file indicated at instantiation.

        :param curve: a Curve object containing the data to fit.
        :param roi: range of interest, in unit of channel.
        :param funcname: name of the fit function. Mandatory.
        :param preset_label: label of the fit preset in the configuration file, if
               relevant.
        :param p0_raw: p0 values of the fit preset in configuration file, if relevant.
        :param kwargs: to provide custom initial guess p0. Weird syntax for grapa GUI.

               Syntax 1: keywords "p0" and "fixed" are provided. Instances of list.

               Syntax 2: to meet limitations of grapa GUI. "0p0": 1.0, "0fi": True
               (i.e. is fixed fit parameter), "1p0": 2.1, etc. These are later compiled
               into lists: `p0` and `fixed`
        :return: a CurveMCA fit curve
        """
        # print("roi", roi, "ARGS", args)
        roi = list(roi)
        if len(roi) != 2 or not is_number(roi[0]) or not is_number(roi[1]):
            msg = "fit_explicit: parameter roi must be a len=2 list of numerals."
            raise RuntimeError(msg)
        funcname = str(funcname)
        func = self.get_fitfunction(funcname)
        p0_raw = [] if p0_raw is None else list(p0_raw)
        p0_input, fixed = self._kwargs_to_p0_fixed(kwargs)
        p0, _ = self.p0_interpret(p0_input, curve)

        # actual fitting
        popt, curvefit = self.fit(curve, func, roi, p0, fixed)

        # result output
        label = "{} fit {}".format(curve.attr("label"), preset_label)
        # cleanup - e.g. background information etc.
        for i in range(len(popt)):
            if not is_number(popt[i]):
                popt[i] = p0_input[i]
        # output
        attrs = {
            "color": "k",
            "linewidth": 0.5,
            "label": label,
            "labelhide": 1,
            "_popt": popt,
            "_fit_p0": p0_input,
            "_fit_p0_raw": p0_raw,
            "_fit_presetlabel": preset_label,
        }
        curvefit.update(attrs)
        return curvefit


class FitHandlerBasicFunc(FitHandler):
    """A class providing a few starting fitting functions"""

    def funcListGUI_ifnewline(self, counter, lbl, func):
        # to help grapa GUI look nice: add new lines when too many fit parameters
        if func in [self.func_gaussa, self.func_cst_gaussa] and lbl == "a":
            return True
        if func in [self.func_gaussi, self.func_cst_gaussi] and lbl == "i":
            return True
        return super().funcListGUI_ifnewline(counter, lbl, func)

    # basic set of fit functions that can readily be used by other subclasses
    @classmethod
    def func_gaussa(cls, x, a, x0, sigma):
        """Gaussian peak, characterized by peak area"""
        return (
            a
            / (np.abs(sigma) * np.sqrt(2 * np.pi))
            * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
        )

    @classmethod
    def func_gaussi(cls, x, i, x0, sigma):
        """Gaussian peak, characterized by peak value"""
        return i * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    @classmethod
    def func_cst_gaussa(cls, x, cst, a, x0, sigma):
        """constant value + Gaussian (area)"""
        return cst + cls.func_gaussa(x, a, x0, sigma)

    @classmethod
    def func_cst_gaussi(cls, x, cst, i, x0, sigma):
        """constant value + Gaussian(peak value)"""
        return cst + cls.func_gaussi(x, i, x0, sigma)

    # END OF FIT FUNCTIONS
