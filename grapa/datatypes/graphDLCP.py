# -*- coding: utf-8 -*-
"""
To parse files containing Drive-Level Capacitance Profiling DLCP data,
according to file format of Abt207 Empa
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

from copy import deepcopy

import numpy as np

from grapa.constants import CST
from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import is_number
from grapa.utils.string_manipulations import strToVar
from grapa.datatypes.curveCV import CurveCV


# TODO: understand this factor 2 (maybe V_AC rms vs max value or similar, derivative numerically wrong??)
# https://www.diva-portal.org/smash/get/diva2%3A1504355/FULLTEXT01.pdf?utm_source=chatgpt.com


def curvecv_to_doping_vs_depth(curve, lblsuffix=" N vs depth"):
    if curve is None:
        return None

    depth = curve.x_CVdepth_nm()
    doping = curve.y_CV_Napparent()
    mask = ~np.isnan(doping)
    curve2 = Curve([depth[mask], doping[mask]], curve.get_attributes())
    curve2.update(
        {
            "label": str(curve.attr("label")) + lblsuffix,
            "linespec": "",
            Curve.KEY_AXISLABEL_X: CurveCV.AXISLABELS_X["CurveCV.x_CVdepth_nm"],
            Curve.KEY_AXISLABEL_Y: CurveCV.AXISLABELS_Y["CurveCV.y_CV_Napparent"],
        }
    )
    curve2.data_units(unit_x="nm", unit_y="cm-3")
    return curve2


class GraphDLCP(Graph):
    """To parse files contaiing capacitance versus voltage C-V data."""

    KEY_VAC = "V_AC [V]"

    FILEIO_GRAPHTYPE = "DLCP curve"

    AXISLABELS = [CurveCV.AXISLABELS_X[""], CurveCV.AXISLABELS_Y[""]]

    AXISLABEL_X_DEPTH = CurveCV.AXISLABELS_X["CurveCV.x_CVdepth_nm"]
    AXISLABEL_Y_NDLCP = list(CurveCV.AXISLABELS_Y["CurveCV.y_CV_Napparent"])
    AXISLABEL_Y_NDLCP[1] = "N_{DLCP}"

    @classmethod
    def isFileReadable(cls, filename, fileext, line1="", **_kwargs):
        """Decides whther the file can be read by this class or not"""
        line1filtered = line1.encode("ascii", errors="ignore").decode()
        if (
            fileext == ".txt"
            and line1filtered.strip().startswith("Sample name:")
            and filename.startswith("DLCP_")
        ):
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        """Parse the data file"""
        # read file content
        data, attrs = _parse_file(self.filename)

        # set [nF] units
        axislabels = deepcopy(GraphDLCP.AXISLABELS)
        units = ["V", "nF"]
        data[:, 2] *= 1e9
        # normalize with areal capacitance
        for key in ["cell area (cm2)", "cell area", "area"]:
            if key in attrs:
                area = attrs[key]
                data[:, 2] /= area
                attrs["cell area (cm2)"] = area
                axislabels[1][2] = axislabels[1][2].replace("F", "F cm$^{-2}$")
                units[1] = "nF cm-2"
                if not self.silent:
                    msg = "Capacitance normalized to area {} cm2."
                    print(msg.format(area))
                break

        # make summary Curve: all datapoints
        curvemain = _extract_curvemain(data, attributes, attrs)
        curvemain.data_units(*units)
        # additional curves: C-V at different V_AC bias amplitudes
        attrs = curvemain.get_attributes()
        label = curvemain.attr("label")
        curves = _extract_cv_at_vac(data, attrs, label)
        # proxy for DLCP at V_AC = 0 V: take the closest values
        curve_dlcplowvac = _extract_dlcp_at_low_vac(curves, attrs, label)
        suffix = " N_DLCP vs depth"
        curve_dlcplowvac_nd = curvecv_to_doping_vs_depth(curve_dlcplowvac, suffix)
        # second calculation: N_DLCP vs depth, according to C0, C1 formula
        curve_dlcpc0c1 = _extract_dlcp_c0_c1(curves, attrs, label)
        # last for comparison: the C-V depth profile with lowest V_AC amplitude
        curve_cvvacmin_nd = curvecv_to_doping_vs_depth(curves[0], " N_CV vs depth")

        # add Curves into Graph
        if curve_dlcplowvac is not None:
            curves.insert(0, curve_dlcplowvac)  # will be added and hidden later
        if curve_dlcpc0c1 is not None:
            self.append(curve_dlcpc0c1)  # not through curves because loops below
            axislabels[0] = list(curve_dlcpc0c1.attr(Curve.KEY_AXISLABEL_X))
            axislabels[1] = list(curve_dlcpc0c1.attr(Curve.KEY_AXISLABEL_Y))
            self.update({"typeplot": "semilogy"})
        if curve_dlcplowvac_nd is not None:
            self.append(curve_dlcplowvac_nd)
            curve_dlcplowvac_nd.update({Curve.KEY_AXISLABEL_Y: list(axislabels[1])})
        if curve_cvvacmin_nd is not None:
            self.append(curve_cvvacmin_nd)
        self.append(curvemain)
        self.append(curves)

        # cosmetics
        curvemain.visible(False)
        for curve in curves:
            curve.visible(False)
        self.update({"xlabel": axislabels[0], "ylabel": axislabels[1]})


def _parse_file(filename):
    """Reads and interpreet the content of the input file"""
    # read file content
    with open(filename, "r", encoding="ascii", errors="ignore") as file:
        lines = [line.strip() for line in file.readlines()]
        # lines = [line.encode("ascii", "ignore").decode("ascii") for line in lines]
    # interpret content: metadata, data
    data = []
    attrs = {}
    collabels = []
    reacheddata = False
    for line in lines:
        if not reacheddata:
            if len(line) == 0:
                continue  # NB: want to keep blank lines once reached data

            split = line.split("\t")
            if len(split) == 2:
                key = split[0].strip(" :")
                if key.lower() == "cell area (cm)":
                    key = "cell area (cm2)"
                attrs.update({key: strToVar(split[1].strip())})
                continue

            if len(split) == 3:
                collabels = split
                reacheddata = True
                continue

        if len(line) == 0 and len(data) == 0:
            continue

        split = [float(s) if is_number(s) else np.nan for s in line.split("\t")]
        while len(split) < 3:
            split.append(np.nan)
        data.append(split)
    data = np.array(data, dtype=float)

    # some checks
    expected = ["Vac [V]", "Vdc [V]", "C [F]"]
    if collabels != expected:
        msg = "Beware, unexpected file format! Expected collabels {}, got {}."
        print(msg.format(expected, collabels))
    return data, attrs


def _extract_dlcp_c0_c1(curves, attrs, label):
    """
    NB: does not try to extrapolate to V_AC = 0 V, just takes the first datapoint
    """
    if len(curves) < 2:
        return None

    epsilon_r = curves[0].getEpsR()
    vacs = [curves[0].attr(GraphDLCP.KEY_VAC), curves[1].attr(GraphDLCP.KEY_VAC)]

    c0 = curves[0].y() * 1e-9 * 1e4  # c0 units [F m-2] instead of [nF cm-2]
    # c1 in units of [F m-2 V-1]
    c1 = (curves[0].y() - curves[1].y()) / (vacs[0] - vacs[1]) * 1e-9 * 1e4
    n_dlcp = -(c0**3) / (2 * CST.q * CST.epsilon0 * epsilon_r * c1) * 1e-6  # [cm-3]

    depth = curves[0].x_CVdepth_nm()
    curve = Curve([depth, n_dlcp], attrs)
    curve.update(
        {
            "label": "{} C0,C1 N_DLCP vs depth".format(label),
            "linespec": "",
            Curve.KEY_AXISLABEL_X: GraphDLCP.AXISLABEL_X_DEPTH,
            Curve.KEY_AXISLABEL_Y: GraphDLCP.AXISLABEL_Y_NDLCP,
        }
    )
    curve.data_units(unit_x="nm", unit_y="cm-3")
    return curve


def _extract_dlcp_at_low_vac(curves, attrs, label):
    """Extracts the DLCP profile: cannot guess the value at V_AC = 0V modulation.
    As a proxy, takes the first 3 lowest V_AC values - grapa apparent doping would show
    that as only 1 point (derivative kills side points if NaNs, we have plenty of NaNs).
    """
    # if at least 3 different V_AC biases: extract extrema datapoint
    if len(curves) < 3:
        return None, None

    v_ac = curves[1].attr(GraphDLCP.KEY_VAC)
    le = len(curves[0].x())
    xs = np.array([curves[i].x() for i in range(3)] + [[np.nan] * le])
    ys = np.array([curves[i].y() for i in range(3)] + [[np.nan] * le])
    xs = [0] + list(np.transpose(xs).flatten()) + [0]
    ys = [np.nan] + list(np.transpose(ys).flatten()) + [np.nan]
    curve = CurveCV([xs, ys], attrs)
    curve.update(
        {
            "label": "{} DLCP V_AC={}V".format(label, v_ac),
            "linespec": "o",
            GraphDLCP.KEY_VAC: v_ac,
        }
    )
    return curve


def _extract_curvemain(data, attributes, attrs):
    """Create curvemain that contains all data, columns 1 and 2 of the input file"""
    curvemain = CurveCV([[], []], attributes)
    curvemain.appendPoints(data[:, 1], data[:, 2])  # this way keep the NaN values
    curvemain.update(attrs)

    # label based on file name, maybe want to base it on file content
    label = curvemain.attr("label")
    if label.startswith("DLCP "):
        label = label[5:]
    curvemain.update({"label": label, "label_initial": label})
    # cell name, sample name
    cellname = curvemain.attr("cell name")
    if not curvemain.has_attr("cell") and cellname != "":
        curvemain.update({"cell": cellname})
    samplename = curvemain.attr("sample name")
    if curvemain.attr("sample") == "" and samplename != "":
        curvemain.update({"sample": samplename})
    return curvemain


def _extract_cv_at_vac(data, attrs, label):
    """Generates the list of C-V sweeps for different V_AC values"""
    to_add = {}
    for line in data:
        if np.isnan(line[0]):
            continue
        key = line[0]
        if key not in to_add:
            to_add[key] = []
        to_add[key].append([line[1], line[2]])
    keys_vac_sorted = sorted(list(to_add.keys()))
    curves = []
    for key in keys_vac_sorted:
        arr = np.array(to_add[key])
        curve = CurveCV([arr[:, 0], arr[:, 1]], attrs)
        curve.update({"label": "{} C-V V_AC={}V".format(label, key)})
        curve.update({GraphDLCP.KEY_VAC: key})
        curves.append(curve)
    return curves
