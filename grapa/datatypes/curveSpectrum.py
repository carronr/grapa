# -*- coding: utf-8 -*-
"""Curve subclass to provide analysis function for spectra. For example spectrometers,
UV-VIS, PL spectra.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import numpy as np

from grapa.graph import Graph
from grapa.curve import Curve, math_on_curves
from grapa.mathModule import is_number, roundSignificantRange, MathOperator
from grapa.constants import CST
from grapa.datatypes.curveTRPL import integrate

# TODO: better parse perkin elmer Reflectnce/transmittance
# TODO: substract absorption in substrate


class CurveSpectrum(Curve):
    """
    Class handling optical spectra, with notably nm to eV conversion and
    background substraction.
    """

    CURVE = "Curve Spectrum"
    FILE_INSTR_RESP = None
    FILE_SUBSTRATES = None
    LIST_SUBSTRATES = None

    def __init__(self, data, attributes, silent=False):
        # main constructor
        Curve.__init__(self, data, attributes, silent=silent)
        # initialize
        self.update({"Curve": CurveSpectrum.CURVE})
        self.update({"_spectrumGUIoffset": True, "_spectrumGUIdark": True})
        subclass = self.getSubclass()
        if subclass is not None:
            subclass.initialize(self)

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        out = Curve.funcListGUI(self, **kwargs)
        # format: [func, 'func label', ['input 1', 'input 2', 'input 3', ...]]
        # choice between subclass
        subclass = self.getSubclass()
        # nm <> eV conversion
        out.append([self.dataModifySwapNmEv, "change data nm<->eV", [], []])
        # subtype
        choices = ["undefined"]
        choices += [s.__name__.replace("CurveSpectrum", "") for s in self.subclasses]
        currentsubclass = (
            subclass.__name__.replace("CurveSpectrum", "")
            if subclass is not None
            else "undefined"
        )
        unit = str(self.attr("_spectrumunit"))
        out.append(
            [
                self.setSubclass,
                "Save",
                ["Spectrum type", "unit"],
                [currentsubclass, unit],
                {},
                [
                    {"field": "Combobox", "values": choices},
                    {"field": "Combobox", "values": ["(no unit)", "%"]},
                ],
            ]
        )
        # offset
        if self.attr("_spectrumGUIoffset"):
            out.append(
                [self.addOffset, "Add offset", ["new offset"], [self.getOffset()]]
            )  # one line per function
        # dark
        if self.attr("_spectrumGUIdark"):
            iddark = []
            if "graph" in kwargs:
                for c in range(len(kwargs["graph"])):
                    iddark.append(str(c) + " " + kwargs["graph"][c].attr("label")[:6])
            else:
                print("graph must be provided to funcListGUI")
                kwargs["graph"] = None
            out.append(
                [
                    self.substractBG,
                    "Subtract dark",
                    ["id dark", "", "", ""],
                    ["0", "1 interpolate both", "1 new curve", "0 ignore offset"],
                    {"graph": kwargs["graph"], "operator": MathOperator.SUB},
                    [
                        {
                            "field": "Combobox",
                            "bind": "beforespace",
                            "width": 4,
                            "values": iddark,
                        },
                        {
                            "field": "Combobox",
                            "width": 13,
                            "values": [
                                "1 interpolate both",
                                "2 interpolate dark",
                                "0 element-wise",
                            ],
                        },
                        {
                            "field": "Combobox",
                            "width": 8,
                            "values": ["1 new curve", "0 replace"],
                        },
                        {
                            "field": "Combobox",
                            "width": 13,
                            "values": ["0 ignore offsets", "1 with offsets"],
                        },
                    ],
                ]
            )
        # integration
        try:
            from grapa.datatypes.curveTRPL import CurveTRPL

            alter = (
                str(kwargs["graph"].attr("alter")) if "graph" in kwargs else "['', '']"
            )
            ROI = roundSignificantRange([min(self.x()), max(self.x())], 2)
            out.append(
                [
                    self.integrate,
                    "Integrate",
                    ["ROI", "data transform"],
                    [ROI, alter],
                    [{}, {"field": "Combobox", "values": ["raw", alter]}],
                ]
            )
        except ImportError:
            pass
        # other, specific
        if subclass is not None and hasattr(subclass, "funcListGUISpecific"):
            out += subclass.funcListGUISpecific(self, **kwargs)
        # help
        out.append([self.print_help, "Help!", [], []])  # one line per function
        return out

    def alterListGUI(self):
        out = Curve.alterListGUI(self)
        out += [["nm <-> eV", [self.ALTER_NM_EV, ""], ""]]
        out += [["nm <-> cm-1", [self.ALTER_NM_CM, ""], ""]]
        return out

    def getSubclass(self):
        self.subclasses = CurveSpectrum.__subclasses__()
        val = self.attr("_spectrumSubclass")
        for s in self.subclasses:
            if val == s.__name__:
                return s
        return None

    def setSubclass(self, newtype, unit) -> bool:
        """Spectrum type: set the Spectrum spectrum, affecting the GUI actions.

        :param newtype: subclass of CurveSpectrum. Examples:
               'CurveSpectrumReflectance', 'CurveSpectrumTransmittance', etc.
        :param unit: empty, or '%' if data are in percent units.
        :return: True for success, False otherwise.
        """
        # set unit
        if unit != "(no unit)":
            self.update({"_spectrumunit": unit})
        # set new type
        if newtype in ["", "undefined"]:
            self.update({"_spectrumSubclass": ""})
        else:
            if not newtype.startswith("CurveSpectrum"):
                newtype = "CurveSpectrum" + newtype
            for s in CurveSpectrum.__subclasses__():
                if newtype == s.__name__:
                    self.update({"_spectrumSubclass": newtype})
                    return True
            print("CurveSpectrum.setSubclass: cannot find class", newtype)
            return False
        return True

    @classmethod
    def fileInstrResp(cls):
        if CurveSpectrum.FILE_INSTR_RESP is None:
            path = os.path.dirname(os.path.abspath(__file__))
            ref = Graph(os.path.join(path, "spectrumInstrumentalResponses.txt"))
            CurveSpectrum.FILE_INSTR_RESP = ref
        return CurveSpectrum.FILE_INSTR_RESP

    @classmethod
    def fileSubstrates(cls):
        if CurveSpectrum.FILE_SUBSTRATES is None:
            path = os.path.dirname(os.path.abspath(__file__))
            ref = Graph(os.path.join(path, "spectrumSubstrates.txt"))
            CurveSpectrum.FILE_SUBSTRATES = ref
        return CurveSpectrum.FILE_SUBSTRATES

    @classmethod
    def getListSubstrates(cls):
        """
        returns the substrates that have <substrate>+' R' and <substrate>+' T'
        in spectrumSubstrates.txt
        """
        if CurveSpectrum.LIST_SUBSTRATES is None:
            graph = cls.fileSubstrates()
            lbls = [curve.attr("label") for curve in graph]
            subsdict = {}
            for lbl in lbls:
                key = lbl[:-2]
                if key not in subsdict:
                    subsdict.update({key: ""})
                if lbl.endswith(" T"):
                    subsdict[key] = subsdict[key] + "T"
                if lbl.endswith(" R"):
                    subsdict[key] = subsdict[key] + "R"
            out = ["nope"]
            for key in subsdict:
                if "R" in subsdict[key] and "T" in subsdict[key]:
                    out.append(key)
            CurveSpectrum.LIST_SUBSTRATES = out
        return CurveSpectrum.LIST_SUBSTRATES

    def getOffset(self):
        return self.attr("_spectrumOffset", 0)

    def addOffset(self, value):
        """
        Add offset: adds a vertical offset to the data. The data are modified.
        The cumulated data correction is displayed, such that setting it to 0
        retrieves the original data (with some rounding errors)
        """
        if is_number(value):
            self.setY(self.y() + value - self.getOffset())
            self.update({"_spectrumOffset": value})
            return True
        return False

    # more "usual" methods
    def dataModifySwapNmEv(self):
        self.setX(CST.nm_eV / self.x())

    def substractBG(self, id_dark, interpolate, if_new, offsets, graph=None, **kwargs):
        """Substract dark: substract a curve to the data.

        :param id_dark: index of the Curve containing the dark spectrum.
        :param interpolate:

               - 0: performs element-wise substraction,

               - 1: output on x points of both data and dark, interpolate both,

               - 2: output on selected Curve x points, interpolate the dark Curve.

        :param if_new: 1: create a new Curve, 0: modify existing data.
        :param offsets: 0: ignore offset and muloffset information.
               1: substract data after offset and muloffset operation.
        :param graph: Graph object containing the dark Curve referred by idDark
        """
        # clean input from GUI: interpolate
        if interpolate in [0, 1, 2, "0", "1", "2"]:
            interpolate = int(interpolate)
        elif (
            isinstance(interpolate, str)
            and len(interpolate) > 0
            and interpolate[0] in ["0", "1", "2"]
        ):
            interpolate = int(interpolate[0])
        else:
            interpolate = 1

        # clean input from GUI: ifNew
        def cleanInputBool(in_, default=False):
            if in_ in [0, 1, True, False, "0", "1", "True", "False"]:
                if in_ in ["0", "False"]:
                    in_ = False
                else:
                    in_ = bool(in_)
            elif isinstance(in_, str) and len(in_) > 0 and in_[0] in ["0", "1"]:
                in_ = True if in_[0] == "1" else False
            else:
                in_ = default
            return in_

        # clean input from GUI: ifNew, offsets
        if_new = cleanInputBool(if_new, default=True)
        offsets = cleanInputBool(offsets, default=False)
        id_dark = min(int(id_dark), len(graph) - 1)
        id_self = None  # index of 'curve' in graph
        for c in range(len(graph)):
            if graph[c] == self:
                id_self = c
                break

        if "operator" not in kwargs:
            kwargs["operator"] = MathOperator.SUB
        out = math_on_curves(
            self, graph[id_dark], interpolate=interpolate, offsets=offsets, **kwargs
        )
        key = "_sub"
        msg = "{{Curve {}: {}}} - {{Curve {}: {}}}"
        msgfmt = msg.format(
            (str(id_self) if id_self is not None else ""),
            self.attr("label"),
            str(id_dark),
            graph[id_dark].attr("label"),
        )
        out.update({key: msgfmt})
        if not if_new:
            if id_self is not None:
                graph.curve_replace(out, id_self)
                print("CurveSpectrum.substractBG: modified Curve data.")
                return True
            print(
                "CurveSpectrum.substractBG: cannot identify Curve!?! Created",
                "a new one instead.",
            )
        print("CurveSpectrum.substractBG: created new Curve.")
        return out

    def correctInstrumentalResponse(self, instrresp, *_args, **_kwargs):
        """
        Correct for the instrumental response, using an instrumental response
        provided in file grapa/datatypes/spectrumInstrumentalResponses.txt
        !! Feel free to adapt the file with curves matching your system !!
        Parameters:
        - instrresp: label of the desired instr. resp. Curve in the file
        """
        file = self.fileInstrResp()
        for curve in file:
            if curve.attr("label") == instrresp:
                out = self / curve
                val = self.attr("_spectrumCorrected")
                out.update(
                    {
                        "_spectrumCorrected": val + instrresp + ";",
                        "label": self.attr("label") + " instr. resp. corr.",
                    }
                )
                self.update({"linestyle": "none"})
                return out
        return (
            "Curve Spectrum instrumental response not found (required",
            instrresp,
            ", available",
            ", ".join([curve.attr("label") for curve in file]),
            ")",
        )

    def computeAbsorptance(self, auxcurve, graph=None, **kwargs) -> Curve:
        """
        Compute absorptance: computes an absorptance curve, defined as A% = 1 - R% - T%.

        :param auxcurve: index to a transmittance/reflectance curve,
        :param graph: a graph containing the curve refered as auxcurve
        :return: a Curve, or False if failure
        """
        out = self.computeAlpha(
            auxcurve, None, None, alpha=False, graph=graph, **kwargs
        )
        if out is False:
            return out
        out.update(
            {
                "_spectrumalpha": "",
                "_spectrumabsorptance": "args auxcurve" + str(auxcurve),
            }
        )
        return out

    def computeAlpha(
        self, auxcurve, thickness, substrate, alpha=True, graph=None, **_kwargs
    ) -> Curve:
        """
        Estimate alpha: estimates the optical absorption spectrum of a layer,
        alpha, or absrptance,
        alpha uses the formula ``alpha[cm-1] = - 1 / d[cm] * ln(T / (1-R)``
        This formula is a reasonable approximation, assuming a few assumptions
        detailed in the manual. Notably, the substrate should be transparent,
        and most of the reflections occurs at the air-layer interface.
        A substrate with low refractive index is to be preferred.

        :param auxcurve: index to a transmittance/reflectance curve,
        :param thickness: thickness of the layer, in [nm],
        :param substrate: a simple model can be used to account for the absorption
               in the substrate. The formula becomes:
               ``alpha[cm-1] = - 1 / d * ln(T / (1-R) * (1-Rsub) / Tsub)``
        :param alpha: if True, computes alpha. If False, compute absorptance
        :param graph: a graph containing the curve refered as auxcurve
        :return: a CurveSpectrum
        """
        if not isinstance(graph, Graph):
            print('Curve Spectrum computeAbsorptance, keyword "graph" not', "provided")
            return False

        if alpha:
            Rsub, Tsub = None, None
            if substrate not in ["", "none"]:
                filesubs = self.fileSubstrates()
                Rsub, Tsub = None, None
                for curve in filesubs:
                    lbl = curve.attr("label")
                    if lbl == substrate + " R":
                        Rsub = curve
                    if lbl == substrate + " T":
                        Tsub = curve
                if Rsub is None or Tsub is None:
                    msg = (
                        "Curve Spectrum computeAbsorptance (alpha), cannot find "
                        "substrate curves ({}), substrate ignored."
                    )
                    print(msg.format(substrate))
                    Rsub, Tsub = None, None
        aux = None
        if isinstance(auxcurve, float):
            auxcurve = int(auxcurve)
            if 0 <= auxcurve < len(graph):
                aux = graph[auxcurve]
        else:
            for curve in range(len(graph)):
                if auxcurve == str(curve) + " " + graph[curve].attr("label"):
                    aux = graph[curve]
                    break
        if aux is None:
            msg = "Curve Spectrum computeAbsorptance, auxiliary curve not found ({})"
            print(msg.format(auxcurve))
            return False

        # determination R or T
        if (
            self.attr("_spectrumSubclass") == "CurveSpectrumTransmittance"
            or aux.attr("_spectrumSubclass") == "CurveSpectrumReflectance"
        ):
            R, T = aux, self
        else:
            R, T = self, aux
        multR = 100 if R.attr("_spectrumunit") == "%" else 1
        multT = 100 if T.attr("_spectrumunit") == "%" else 1
        if alpha:  # alpha
            out = T / multT / (1 - R / multR)
            if Rsub is not None and Tsub is not None:
                out *= (1 - Rsub) / Tsub
            out.setY(-1 / (1e-7 * thickness) * np.log(out.y()))
            out.update({"_spectrumSubclass": ""})
        else:  # absorptance
            out = 1 - R / multR - T / multT
            if multR == 100 and multT == 100:
                out.setY(out.y() * 100)
            out.update({"_spectrumSubclass": "CurveSpectrumAbsorptance"})
        lbl = self.attr("label").replace(" instr. resp. corr.", "")
        replacelist = [
            "Reflectance",
            "reflectance",
            "R%",
            "Transmittance",
            "transmittance",
            "T%",
        ]
        replacement = "Absorptance" if not alpha else "alpha [cm$^{-1}$]"
        flag = False
        for r in replacelist:
            if r in lbl:
                lbl = lbl.replace(r, replacement)
                flag = True
        if not flag:
            lbl = lbl + " " + replacement
        out.update(
            {
                "label": lbl,
                "_spectrumalpha": "args auxcurve"
                + str(auxcurve)
                + ", thickness"
                + str(thickness)
                + ", substrate"
                + str(substrate),
            }
        )
        return out

    def integrate(self, ROI=None, alter=None):
        """Integrate: returns the integral of the curve, within ROI.

        :param ROI: example [xmin, xmax]
        :param alter: 'raw', or any Graph 'alter' value including (mul-)offsets.
        """
        return integrate(self, roi=ROI, alter=alter)

    def print_help(self):
        print("*** *** ***")
        print(
            "CurveSpectrum offers some capabilities to process optical spectrum data."
        )
        print("Curve transforms:")
        print(
            " - nm <-> eV: switch [eV] or [nm] data into the other",
            "representation (eV =",
            CST.nm_eV,
            "/ nm).",
        )
        print("Analysis functions")
        print(
            " - Change data nm<->eV: changes data from nm to eV or",
            "inversely (eV =",
            CST.nm_eV,
            "/ nm),",
        )
        self.print_help_func(CurveSpectrum.setSubclass)
        if self.attr("_spectrumGUIoffset", True):
            self.print_help_func(CurveSpectrum.addOffset)
        if self.attr("_spectrumGUIdark", True):
            self.print_help_func(CurveSpectrum.substractBG)
        sub = self.getSubclass()
        if sub is not None and hasattr(sub, "printHelp_"):
            sub.printHelp_(self)
        return True


class CurveSpectrumReflectance(CurveSpectrum):
    def initialize(self):
        self.update({"_spectrumGUIoffset": False, "_spectrumGUIdark": False})

    def funcListGUISpecific(self, **kwargs):
        out = []
        graph_instrresp = self.fileInstrResp()
        graph = kwargs["graph"] if "graph" in kwargs else None
        # instrumental response
        corr = "(already done) " if self.attr("_spectrumCorrected") else ""
        lbls = [curve.attr("label") for curve in graph_instrresp]
        out.append(
            [
                self.correctInstrumentalResponse,
                "Correct",
                [corr + "instrumental response"],
                [lbls[0]],
                {},
                [{"field": "Combobox", "width": 25, "values": lbls}],
            ]
        )
        # compute absorptance, alpha
        if graph is not None:
            poss = []
            for c in range(len(graph)):
                if isinstance(graph[c], CurveSpectrum):
                    if (
                        graph[c].attr("_spectrumSubclass")
                        == "CurveSpectrumTransmittance"
                    ):
                        poss.append(str(c) + " " + graph[c].attr("label"))
            poss_ = poss[0] if len(poss) > 0 else ""
            out.append(
                [
                    self.computeAbsorptance,
                    "Compute absorptance",
                    ["select transmittance"],
                    [poss_],
                    {"graph": graph},
                    [{"field": "Combobox", "width": 20, "values": poss}],
                ]
            )
            listsubs = self.getListSubstrates()
            out.append(
                [
                    self.computeAlpha,
                    "Estimate \u03B1",
                    ["transm.", "thickness [nm]", "substrate"],
                    [poss_, 50, "none"],
                    {"graph": graph},
                    [
                        {"field": "Combobox", "width": 12, "values": poss},
                        {"width": 7},
                        {"field": "Combobox", "width": 6, "values": listsubs},
                    ],
                ]
            )
        return out

    def printHelp_(self):
        print("CurveSpectrumReflectance offers 3 additional processing", "functions.")
        self.print_help_func(CurveSpectrum.correctInstrumentalResponse)
        self.print_help_func(CurveSpectrum.computeAbsorptance)
        self.print_help_func(CurveSpectrum.computeAlpha)


class CurveSpectrumTransmittance(CurveSpectrum):
    def initialize(self):
        self.update({"_spectrumGUIoffset": False, "_spectrumGUIdark": False})

    def funcListGUISpecific(self, **kwargs):
        out = []
        graph_instrresp = self.fileInstrResp()
        graph = kwargs["graph"] if "graph" in kwargs else None
        # instrumental response
        corr = "(already) " if self.attr("_spectrumCorrected") else ""
        lbls = [curve.attr("label") for curve in graph_instrresp]
        value = "Transmittance (specular)"
        if value not in lbls:
            if "Transmittance (diffuse)" in lbls:
                value = "Transmittance (diffuse)"
            if corr == "":
                corr = "(are you sure?) "
        out.append(
            [
                self.correctInstrumentalResponse,
                "Correct",
                [corr + "instrumental response"],
                [value],
                {},
                [{"field": "Combobox", "width": 22, "values": lbls}],
            ]
        )
        # compute absorptance, alpha
        if graph is not None:
            poss = []
            for c in range(len(graph)):
                if isinstance(graph[c], CurveSpectrum):
                    if graph[c].attr("_spectrumSubclass") == "CurveSpectrumReflectance":
                        poss.append(str(c) + " " + graph[c].attr("label"))
            poss_ = poss[0] if len(poss) > 0 else ""
            out.append(
                [
                    self.computeAbsorptance,
                    "Compute absorptance",
                    ["select reflectance"],
                    [poss_ if len(poss) > 0 else ""],
                    {"graph": graph},
                    [{"field": "Combobox", "width": 20, "values": poss}],
                ]
            )
            listsubs = self.getListSubstrates()
            out.append(
                [
                    self.computeAlpha,
                    "Estimate \u03B1",
                    ["refl.", "thickness [nm]", "substrate"],
                    [poss_, 50, "none"],
                    {"graph": graph},
                    [
                        {"field": "Combobox", "width": 12, "values": poss},
                        {"width": 7},
                        {"field": "Combobox", "width": 6, "values": listsubs},
                    ],
                ]
            )
        return out

    def printHelp_(self):
        print("CurveSpectrumTransmittance offers 3 additional processing", "functions.")
        self.print_help_func(CurveSpectrum.correctInstrumentalResponse)
        self.print_help_func(CurveSpectrum.computeAbsorptance)
        self.print_help_func(CurveSpectrum.computeAlpha)


class CurveSpectrumAbsorptance(CurveSpectrum):
    def initialize(self):
        self.update({"_spectrumGUIoffset": False, "_spectrumGUIdark": False})
