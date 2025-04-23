# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:44:58 2016

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os

import numpy as np
import copy

from grapa.mathModule import is_number, roundSignificant
from grapa.curve import Curve
from grapa.utils.curve_subclasses_utils import FileLoaderOnDemand, FitHandlerBasicFunc
from grapa.utils.funcgui import FuncListGUIHelper, FuncGUI

KEY_CTOKEV_OFFSET = "_MCA_CtokeV_offset"
KEY_CTOKEV_MULT = "_MCA_CtokeV_mult"


class FitHandlerXRF(FitHandlerBasicFunc):
    """Added on base class
    - fit fucntions
    - several overrides to handle peak amplitude
    - added text interpreters for input database file"""

    def __init__(self, filename):
        super().__init__(filename)
        self.fitcurve_attr_to_copy += [KEY_CTOKEV_OFFSET, KEY_CTOKEV_MULT]

    def funcListGUI_ifnewline(self, counter, lbl, func):
        if lbl.startswith("bgvalueatx0"):
            return True
        if lbl.startswith("interp"):
            return True
        if len(lbl) in [2, 3, 3] and lbl[0] == "g" and lbl[-1] == "a":
            return True
        if len(lbl) in [7, 8] and lbl[0] == "g" and lbl.endswith("ratio"):
            return True
        return super().funcListGUI_ifnewline(counter, lbl, func)

    def funcListGUI_notafit_preset(self, curve, preset, **kwargs):
        out = super().funcListGUI_notafit_preset(curve, preset, **kwargs)
        # add fields to calculate amplitude, and insert it where it is needed
        sum_amp_gaussian = preset.attr("sum_amp_gaussian")
        out[-1].append(
            "calculation amplitude", sum_amp_gaussian, keyword="sum_amp_gaussian"
        )
        out[-1].move_item(-1, 1)
        return out

    def updateFitParam_after(self, curve, revert):
        # super() behavior
        super().updateFitParam_after(curve, revert)
        # compute and display peak area
        self.compute_amplitude(curve)

    # handling of Amplitude
    def compute_amplitude(self, curve):
        # if popt is None:
        sum_amp_gaussian = curve.attr("_fit_sum_amp_gaussian")
        if isinstance(sum_amp_gaussian, list):
            popt = curve.attr("_popt")
            funcname = curve.attr("_fitfunc")
            func = self.get_fitfunction(funcname)
            amp = self.compute_amplitude_actual(func, popt, sum_amp_gaussian)
            curve.update({"_fit_amp": amp})
            msg = "CurveMCA.compute_amplitude: peak area {} (gaussian {})"
            print(msg.format(amp, sum_amp_gaussian))
            return amp
        raise NotImplementedError

    def compute_amplitude_actual(self, func, popt, sum_amp_gaussian):
        """
        Returns peak amplitude, based on the fitfunc, popt sum_amp_gaussian. e.g. sum
        of one or several gaussian peaks sum_amp_gaussian: e.g [0, 1], aka first 2
        gaussians must be considered
        """
        amp = 0
        if 0 in sum_amp_gaussian:
            if func in [
                self.func_linbg_gaussa,
                self.func_linbg_2gaussa,
                self.func_linbg_2gaussaratiodelta,
                self.func_linbg_3gaussaratiodelta,
                self.func_linbg_4gaussaratiodelta,
                self.func_linbg_2gaussaratiodelta_3gaussaratiodelta,
            ]:
                amp += popt[3]
            elif func in [self.func_interp_2gaussaratiodeltadx]:
                amp += popt[4]
            else:
                msg = "ERROR fit_amp calculation amp 0 {}, {}"
                print(msg.format(sum_amp_gaussian, func.__name__))
                raise NotImplementedError(msg.format(sum_amp_gaussian, func.__name__))
        if 1 in sum_amp_gaussian:
            if func == self.func_linbg_2gaussa:
                amp += popt[6]
            elif func in [
                self.func_linbg_2gaussaratiodelta,
                self.func_linbg_3gaussaratiodelta,
                self.func_linbg_2gaussaratiodelta_3gaussaratiodelta,
            ]:
                amp += popt[3] * popt[6]
            elif func in [self.func_interp_2gaussaratiodeltadx]:
                amp += popt[4] * popt[7]
            else:
                msg = "ERROR fit_amp calculation amp 1 {}, {}"
                print(msg.format(sum_amp_gaussian, func.__name__))
                raise NotImplementedError(msg.format(sum_amp_gaussian, func.__name__))
        if 2 in sum_amp_gaussian:
            msg = "ERROR fit_amp calculation amp 2 {}, {}"
            print(msg.format(sum_amp_gaussian, func.__name__))
            raise NotImplementedError(msg.format(sum_amp_gaussian, func.__name__))
        return amp

    # START OF FIT FUNCTIONS
    # A bunch of fit functions
    @classmethod
    def func_gaussa(cls, x, a, x0, sigma):
        """Gaussian peak"""
        return (
            a
            / (np.abs(sigma) * np.sqrt(2 * np.pi))
            * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
        )

    @classmethod
    def func_linbg(cls, x, valueatx0, slope, x0):
        """Linear function, with slope scaling with cst multiplying value.
        e.g. background signal proportional to time"""
        return valueatx0 * (1 + slope * (x - x0))

    @classmethod
    def func_interp(cls, x, interp, mult, cst, dx):
        """Interpolates on an existing dataset, multiplication, add constant, x shift.
        The fit parameter corresponding to interp must be fixed."""
        return interp(x + dx) * mult + cst

    @classmethod
    def func_linbg_gaussa(cls, x, bgvalueatx0, bgslope, bgx0, ga, gx0, gsigma):
        """Linear background (value at x0 and slope) and 1 gaussian peak"""
        return cls.func_linbg(x, bgvalueatx0, bgslope, bgx0) + cls.func_gaussa(
            x, ga, gx0, gsigma
        )

    @classmethod
    def func_linbg_2gaussa(
        cls, x, bgvalueatx0, bgslope, bgx0, g1a, g1x0, g1sigma, g2a, g2x0, g2sigma
    ):
        """Linear background (value at x0 and slope) and 2 gaussian peaks"""
        return (
            cls.func_linbg(x, bgvalueatx0, bgslope, bgx0)
            + cls.func_gaussa(x, g1a, g1x0, g1sigma)
            + cls.func_gaussa(x, g2a, g2x0, g2sigma)
        )

    @classmethod
    def func_linbg_2gaussaratiodelta(
        cls, x, bgvalueatx0, bgslope, bgx0, g1a, g1x0, g1sigma, g2ratio, g2deltax
    ):
        """Linear background (value at x0 and slope) and two gaussian peaks, the second
        parametrized as area ratio and x shift from the first one."""
        return (
            cls.func_linbg(x, bgvalueatx0, bgslope, bgx0)
            + cls.func_gaussa(x, g1a, g1x0, g1sigma)
            + cls.func_gaussa(x, g1a * g2ratio, g1x0 + g2deltax, g1sigma)
        )

    @classmethod
    def func_linbg_3gaussaratiodelta(
        cls,
        x,
        bgvalueatx0,
        bgslope,
        bgx0,
        g1a,
        g1x0,
        g1sigma,
        g2ratio,
        g2deltax,
        g3ratio,
        g3deltax,
    ):
        """Linear background (value at x0 and slope) and 3 gaussian peaks, the latter
        parametrized as area ratios and x shifts from the first one."""
        return (
            cls.func_linbg(x, bgvalueatx0, bgslope, bgx0)
            + cls.func_gaussa(x, g1a, g1x0, g1sigma)
            + cls.func_gaussa(x, g1a * g2ratio, g1x0 + g2deltax, g1sigma)
            + cls.func_gaussa(x, g1a * g3ratio, g1x0 + g3deltax, g1sigma)
        )

    @classmethod
    def func_linbg_4gaussaratiodelta(
        cls,
        x,
        bgvalueatx0,
        bgslope,
        bgx0,
        g1a,
        g1x0,
        g1sigma,
        g2ratio,
        g2deltax,
        g3ratio,
        g3deltax,
        g4ratio,
        g4deltax,
    ):
        """Linear background (value at x0 and slope) and 4 gaussian peaks, the latter
        parametrized as area ratios and x shifts from the first one."""
        return (
            cls.func_linbg(x, bgvalueatx0, bgslope, bgx0)
            + cls.func_gaussa(x, g1a, g1x0, g1sigma)
            + cls.func_gaussa(x, g1a * g2ratio, g1x0 + g2deltax, g1sigma)
            + cls.func_gaussa(x, g1a * g3ratio, g1x0 + g3deltax, g1sigma)
            + cls.func_gaussa(x, g1a * g4ratio, g1x0 + g4deltax, g1sigma)
        )

    @classmethod
    def func_linbg_2gaussaratiodelta_3gaussaratiodelta(
        cls,
        x,
        bgvalueatx0,
        bgslope,
        bgx0,
        g1a,
        g1x0,
        g1sigma,
        g2ratio,
        g2deltax,
        g3a,
        g3x0,
        g3sigma,
        g4ratio,
        g4deltax,
        g5ratio,
        g5deltax,
    ):
        """Linear background (value at x0 and slope) and 3 gaussian peaks, the latter
        parametrized as area ratios and x shifts from the first one."""
        return (
            cls.func_linbg(x, bgvalueatx0, bgslope, bgx0)
            + cls.func_gaussa(x, g1a, g1x0, g1sigma)
            + cls.func_gaussa(x, g1a * g2ratio, g1x0 + g2deltax, g1sigma)
            + cls.func_gaussa(x, g3a, g3x0, g3sigma)
            + cls.func_gaussa(x, g3a * g4ratio, g3x0 + g4deltax, g3sigma)
            + cls.func_gaussa(x, g3a * g5ratio, g3x0 + g5deltax, g3sigma)
        )

    @classmethod
    def func_interp_gaussa(cls, x, interp, mult, cst, dx, ga, gx0, gsigma):
        """Background interpolated from dataset (multiplier, offset, x shift), and a
        gaussian peak"""
        return cls.func_interp(x, interp, mult, cst, dx) + cls.func_gaussa(
            x, ga, gx0, gsigma
        )

    @classmethod
    def func_interp_2gaussaratiodeltadx(
        cls, x, interp, mult, cst, dx, g1a, g1x0, g1sigma, g2ratio, g2deltax
    ):
        """Background interpolated from dataset, and 2 gaussianpeaks, the second
        parametrized as area ratio and x shift from the first one."""
        return (
            cls.func_interp(x, interp, mult, cst, dx)
            + cls.func_gaussa(x, g1a, g1x0 + dx, g1sigma)
            + cls.func_gaussa(x, g1a * g2ratio, g1x0 + g2deltax + dx, g1sigma)
        )

    # END OF FIT FUNCTIONS
    # Fit-related functions. Override for additional behaviors
    def _p0_interpret_value(self, value, curve, interp):
        """Override, additional text interpretation into values"""
        if value.startswith("channel(") and value.endswith(")"):
            val = value.strip("channel() ")
            return curve.kev_to_channel(float(val))
        elif value.startswith("channeldelta(") and value.endswith(")"):
            val = value.strip("channeldelta() ")
            vals = np.array([float(v) for v in val.split(",")])
            channels = curve.kev_to_channel(vals)
            return channels[1] - channels[0]
        # if nothing specific to this implementation, fall back to default
        return super()._p0_interpret_value(value, curve, interp)

    def fit_explicit(self, curve, *args, sum_amp_gaussian=None, **kwargs):
        """Override, intercept variable sum_amp_gaussian and compute amplitude

        :param curve: a grapa Curve object containing the data to fit
        :param sum_amp_gaussian: e.g. [0, 1], indicate output "amplitude" as sum of
               gaussian peaks 0 and 1
        """
        # intercept variable sum_amp_gaussian

        # fitting -> call super()
        curvefit = super().fit_explicit(curve, *args, **kwargs)

        # some post-processing
        sum_amp_gaussian = [] if sum_amp_gaussian is None else list(sum_amp_gaussian)
        curvefit.update({"_fit_sum_amp_gaussian": sum_amp_gaussian})
        amp = self.compute_amplitude(curvefit)
        label = curvefit.attr("label") + " fit area {}".format(roundSignificant(amp, 5))
        curvefit.update({"label": label})
        return curvefit


class CurveMCA(Curve):
    CURVE = "Curve MCA"

    AXISLABELS_X = {
        "": ["XRF detector channel", "", ""],
        "CurveMCA.x_kev": ["X-ray energy", "", "keV"],
    }
    AXISLABELS_Y = {"": ["Intensity", "", "counts"]}

    _PATHMCA = os.path.dirname(os.path.abspath(__file__))

    FITPARAMETERS_FILE = "XRF_fitparameters.txt"
    FITHANDLER = FitHandlerXRF(os.path.join(_PATHMCA, FITPARAMETERS_FILE))

    PEAKS_FILENAME = "XRF_photonenergiesintensities.txt"
    PEAKS_DATA = None
    PEAKS_ELEMENTS = None
    PEAKS_LINES = None

    DEFAULTVALUES_FILE = "MCA_default.txt"
    DEFAULTVALUES = FileLoaderOnDemand(os.path.join(_PATHMCA, DEFAULTVALUES_FILE), 0)

    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        self.update({"Curve": CurveMCA.CURVE})
        # in __init__ as class itself not yet defined at instantiation of class variable
        self.FITHANDLER.set_typecurvefit(CurveMCA)
        # default channel to keV scaling
        if not self.has_attr(KEY_CTOKEV_OFFSET):
            offset = CurveMCA.DEFAULTVALUES().attr(KEY_CTOKEV_OFFSET)
            self.update({KEY_CTOKEV_OFFSET: offset})  # -3.35
        if not self.has_attr(KEY_CTOKEV_MULT):
            mult = CurveMCA.DEFAULTVALUES().attr(KEY_CTOKEV_MULT)
            self.update({KEY_CTOKEV_MULT: mult})  # 0.031490109515627
        if CurveMCA.PEAKS_DATA is None:
            CurveMCA.peaks_data()

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # energy calibration settings
        at = [KEY_CTOKEV_OFFSET, KEY_CTOKEV_MULT]
        line = FuncGUI(self.updateValuesDictkeys, "Save", hiddenvars={"keys": at})
        line.append("keV = (channel +", self.attr(at[0]))
        line.append(") * ", self.attr(at[1]))
        out.append(line)
        # peaks labelling
        self.peaks_data()
        if CurveMCA.PEAKS_DATA is not None:
            elements = self.PEAKS_ELEMENTS
            lines = ["all"] + self.PEAKS_LINES
            addrem = ["add", "remove"]
            line = FuncGUI(self.addElementLabels, "Save")
            line.set_hiddenvars({"graph": kwargs["graph"]})
            line.appendcbb("", addrem[0], addrem, options={"width": 5})
            line.appendcbb("element", elements[0], elements, options={"width": 5})
            line.appendcbb("series", lines[0], lines, options={"width": 4})
            line.append("mult.", 0.1, options={"width": 6})
            line.append("vline bottom", 0)
            out.append(line)
        # fit-related elements
        out += self.FITHANDLER.funcListGUI(self, **kwargs)
        # help
        out.append([self.print_help, "Help!", [], []])  # one line per function

        muloffsety = self.get_muloffset()[1]
        kw = dict(kwargs)
        kw.update({"lookup_y": {"": "counts s$^{-1}$"}} if muloffsety != 1 else {})
        out += FuncListGUIHelper.graph_axislabels(self, **kw)
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out += [["Channel <-> keV", ["CurveMCA.x_kev", ""], ""]]
        return out

    def updateFitParam(self, *param):
        # override default behavior, additional things to do
        # convert string reference to e.g. background into callable function
        param, revert, func = self.FITHANDLER.updateFitParam_before(self, *param)
        # call base function
        out = super().updateFitParam(*param, func=func)
        # revert callable parameter to its initial string value
        # also, compute and display amplitude - child class
        self.FITHANDLER.updateFitParam_after(self, revert)
        return out

    def x_kev(self, **kwargs):
        # do not call method channel_to_kev. Should be called on other Curve subclasses
        offset = self.attr(KEY_CTOKEV_OFFSET, default=0)
        mult = self.attr(KEY_CTOKEV_MULT, default=1)
        return (self.x(**kwargs) + offset) * mult

    def channel_to_kev(self, channel):
        offset = self.attr(KEY_CTOKEV_OFFSET, default=0)
        mult = self.attr(KEY_CTOKEV_MULT, default=1)
        return (channel + offset) * mult

    def kev_to_channel(self, kev):
        offset = self.attr(KEY_CTOKEV_OFFSET, default=0)
        mult = self.attr(KEY_CTOKEV_MULT, default=1)
        return kev / mult - offset

    @classmethod
    def peaks_data(cls):
        if cls.PEAKS_DATA is None:
            path = os.path.dirname(os.path.abspath(__file__))
            kwargs = {"dtype": None, "skip_header": 3, "delimiter": "\t"}

            try:
                from packaging import version

                if version.parse(np.__version__) >= version.parse("1.1.14"):
                    kwargs.update({"encoding": None})
            except ImportError:
                pass

            data_ = np.genfromtxt(os.path.join(path, cls.PEAKS_FILENAME), **kwargs)
            # make sure we don't have annoying byte lying around, only str
            data = []
            for row in data_:
                data.append(
                    [r.decode("UTF-8") if isinstance(r, np.bytes_) else r for r in row]
                )
                # print(data[-1])
            elements, eldict = [], {}
            lines = []
            for row in data:
                if int(row[1]) not in eldict:
                    eldict[int(row[1])] = str(row[2]) + " " + str(row[1])
                tmp = row[3][0]  # e.g. 'K', 'L', etc.
                if tmp not in lines:
                    lines.append(tmp)
            m = np.max(list(eldict.keys())) + 1
            for key in range(m):
                if key in eldict:
                    elements.append(eldict[key])
            lines.sort()
            cls.PEAKS_DATA = data
            cls.PEAKS_ELEMENTS = elements
            cls.PEAKS_LINES = lines
        return cls.PEAKS_DATA

    def addElementLabels(
        self, addrem: str, element: str, line, multiplier=1, vlinebottom=0, **kwargs
    ) -> bool:
        """Add labels for transition for a given element, according to the database
        https://xdb.lbl.gov/Section1/Table_1-3.pdf

        :param addrem: 'add', or 'remove' sets of lines.
        :param element: str starting with element 2-letter, or numeric (Z number).
               Possible values: 'Cu', 'Zn 30', 25, etc.
        :param line: 'K', 'M, 'L', 'all'
        :param multiplier: multiplier to the tabulated peak intensity. Default 1.
        :param vlinebottom: value at which the vertical line ends. Default 0.
        :return: True if success, False otherwise
        """
        to_keV = 0.001
        if "graph" not in kwargs:
            print('CurveMCA addElementLabels expect "graph" in kwargs')
            return False
        graph = kwargs["graph"]
        data = self.peaks_data()

        def textformat(element, line):
            line = (
                line.replace("a", r"$\alpha$")
                .replace("b", r"$\beta$")
                .replace("g", r"$\gamma$")
            )
            if "$" in line and line[-1] != "$":  # indices
                tmp = line.split("$")
                tmp[-1] = "$_{" + tmp[-1] + "}$"
                line = "$".join(tmp)
                line = line.replace("$$", "")
            return element + " " + line

        def removeTextStartswith(start):
            if start == "":
                return
            graph.text_check_valid()
            texts = graph.attr("text", None)
            idxdel = []
            if texts is not None:
                for i in range(len(texts)):
                    if texts[i].startswith(start):
                        idxdel.append(i)
            for i in idxdel[::-1]:
                # print('remove text i', i, texts[i])
                graph.text_remove(i)
            return True

        if is_number(element):
            element = int(element)
            for row in data:
                if int(row[1]) == element:
                    element = row[2].decode("UTF-8")
                    break
        if is_number(element):
            print("CurveMCA addElementLabels cannot find data for element", element)
            return False
        element = element.split(" ")[0]
        # if user wants to remove some text
        if addrem == "remove":
            start = element
            if line != "all":
                start += " " + line  # also safe vs $
            removeTextStartswith(start)
        elif self.PEAKS_DATA is not None:
            # if add text
            lines = self.PEAKS_LINES if line == "all" else [line]
            adds = []
            argdef = {
                "verticalalignment": "bottom",
                "annotation_clip": False,
                "horizontalalignment": "center",
                "textcoords": "data",
                "arrowprops": {
                    "headwidth": 0,
                    "facecolor": "k",
                    "width": 0,
                    "shrink": 0,
                    "set_clip_box": True,
                },
            }
            for row in data:
                el = row[2]
                li = row[3]
                if el == element and li[0] in lines:
                    text = textformat(row[2], li)
                    textxy = [row[0] * to_keV, row[4] * multiplier]
                    args = copy.deepcopy(argdef)
                    args["xy"] = [row[0] * to_keV, vlinebottom]
                    adds.append([text, textxy, args])
            for add in adds:
                # print('add text', add[0])
                removeTextStartswith(add[0])
                graph.text_add(add[0], add[1], textargs=add[2])
        return True

    # set of functions to handle peak fitting
    def fit(self, func, roi, p0, fixed):
        return self.FITHANDLER.fit(self, func, roi, p0, fixed)

    def fit_preset(self, preset_label, roi="auto", p0="auto", fixed="auto"):
        return self.FITHANDLER.fit_preset(
            self, preset_label, roi=roi, p0=p0, fixed=fixed
        )

    def fit_explicit(
        self,
        roi,
        *args,
        funcname="",
        preset_label="",
        p0_raw=None,
        sum_amp_gaussian=None,
        **kwargs
    ):
        """
        Fit fit_explicit: fit data, for example a peak of a CurveMCA data.
        Pre-set fit parameters are configured in file XRF_fitparameters.txt.

        :param roi: range of interest, in unit of channel
        :param funcname: name of the fit function. Mandatory (provided by GUI)
        :param preset_label: label of the fit preset in the configuration file, if
               relevant (provided by GUI)
        :param p0_raw: p0 values of the fit preset in configuration file, if relevant.
        :param kwargs: to provide custom initial guess p0. Weird syntax for grapa GUI.

               - Syntax 1: keywords "p0" and "fixed" are provided. Instances of list.

               - Syntax 2: to meet limitations of grapa GUI.
                 Example: "0p0": 1.0,  "0fixed": True (i.e. is fixed fit parameter),
                 "1p0": 2.1, etc. These are later compiled into lists: p0 and fixed
        :param sum_amp_gaussian: how to compute to return "amplitude" value.
               Eg [0, 1] would compute as the sum of peak areas of gaussian peaks 0 and
               1, accounting for definition of gaussians with fixed respective peak
               areas e.g. func_linbg_2gaussaratiodelta
        :return: a CurveMCA fit curve
        """
        return self.FITHANDLER.fit_explicit(
            self,
            roi,
            *args,
            funcname=funcname,
            preset_label=preset_label,
            p0_raw=p0_raw,
            sum_amp_gaussian=sum_amp_gaussian,
            **kwargs
        )
        # return CurveMCA(curvefit.data, curvefit.get_attributes())

    def print_help(self):
        print("*** *** ***")
        print("CurveMCA offers capabilities to display raw XRF data.")
        print(
            "Data are automatically given a multiplicative offset",
            "1/acquisition time.",
        )
        print("Curve transforms:")
        print(
            "- Channel <-> eV: switch [channel] data into keV representation",
            "based on properties _MCA_CtokeV_offset and _MCA_CtokeV_mult.",
        )
        print("Analysis functions:")
        self.print_help_func(CurveMCA.addElementLabels)
        self.print_help_func(CurveMCA.fit_explicit)
        return True
