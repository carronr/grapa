# -*- coding: utf-8 -*-

"""
Created on Fri Jul 15 15:46:13 2016

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
import warnings
import inspect
import importlib

from string import Template

from grapa.constants import CST
from grapa.utils.funcgui import FuncListGUIHelper, FuncGUI


class TemplateCustom(Template):
    # braceidpattern = "(?a:[_a-z][_a-z0-9 \[\]]*)"  # added: whitespace, [, ]
    # added: whitespace, [, ]; also, formatter indication as ${key, :.0f}
    braceidpattern = r"(?a:[_a-z][_a-z0-9 \[\]]*(,[ ]*[:.0-9\-\+eEfFgG%]+)?)"


class Curve:
    CURVE = "Curve"

    LINESTYLEHIDE = ["none", "None", None]

    def __init__(self, data, attributes, silent=True):
        self.silent = silent
        self.data = np.array([])
        self.attributes = {}
        # add data
        if isinstance(data, (np.ndarray, np.generic)):
            self.data = data
        else:
            self.data = np.array(data)
        if self.data.shape[0] < 2:
            self.data = self.data.reshape((2, len(self.data)))
            msg = (
                "WARNING class Curve: data Curve need to contain at least 2 "
                "columns. Shape of data: {}"
            )
            print(msg.format(self.data.shape))
        # clean nan pairs at end of data
        iMax = self.data.shape[1]
        # loop down to 1 only, not 0
        for i in range(self.data.shape[1] - 1, 0, -1):
            if np.isnan(self.data[:, i]).all():
                iMax -= 1
            else:
                break
        if iMax < self.data.shape[1]:
            self.data = self.data[:, :iMax]
        # add attributes
        if isinstance(attributes, dict):
            self.update(attributes)
        else:
            print("ERROR class Curve: attributes need to be of class dict.")

    # methods related to GUI
    @classmethod
    def classNameGUI(cls):  # can be overridden, see CurveArrhenius
        return cls.CURVE

    def funcListGUI(self, **kwargs):
        """
        Fills in the Curve actions specific to the Curve type. Syntax:
        [func,
         'Button text',
         ['label 1', 'label 2', ...],
         ['value 1', 'value 2', ...],
         {'hiddenvar1': 'value1', ...}, (optional)
         [dictFieldAttributes, {}, ...]] (optional)
        By default returns quick modifs for offset and muloffset (if already
        set), and a help for some plot types (errorbar, scatter)
        In principle the current Graph and the position of self in the Graph
        can be accessed through kwargs['graph'] and kwargs['graph_i']
        """
        out = []
        # Curve typeplot eg scatter, fill, boxplot etc.
        out += FuncListGUIHelper.typeplot(self, **kwargs)
        # offset and muloffset can be directly accessed if already set
        if (
            self.attr("offset", None) is not None
            or self.attr("muloffset", None) is not None
        ):
            from grapa.graph import Graph

            at = ["offset", "muloffset"]
            line = FuncGUI(
                self.updateValuesDictkeys,
                "Modify screen offsets",
                hiddenvars={"keys": at},
            )
            for key in at:
                vals = []
                if key in Graph.KEYWORDS_CURVE["keys"]:
                    i = Graph.KEYWORDS_CURVE["keys"].index(key)
                    vals = [str(v) for v in Graph.KEYWORDS_CURVE["guiexamples"][i]]
                line.append(
                    key,
                    self.attr(key),
                    widgetclass="Combobox",
                    options={"values": vals},
                )
            out.append(line)
        return out

    def plotterBoxplotAddonUpdateCurve(self, fun, kw):
        # cannot be placed in Plotter, in case user want to apply function on several
        # curves at once as selected from the GUI
        if fun == 0:
            self.update({"boxplot_addon": ""})
        else:
            if not isinstance(kw, dict):
                try:
                    kw = dict(kw)
                except ValueError:
                    print("Invalid input value, failed: {}".format(kw))
                    return False
            self.update({"boxplot_addon": [fun, kw]})
        return True

    def updateNextCurvesScatter(self, *values, **kwargs):
        try:
            graph = kwargs["graph"]
            c = kwargs["graph_i"]
        except KeyError:
            print(
                'KeyError in Curve.updateNextCurvesScatter: "graph" or',
                '"graph_i" not provided in kwargs',
            )
            return False
        for i in range(len(values)):
            if c + i + 1 < len(graph):
                graph[c + 1 + i].update({"type": values[i]})
        return True

    # alterListGUI
    def alterListGUI(self):
        """
        Determines the possible curve visualisations. Syntax:
        ['GUI label', ['alter_x', 'alter_y'], 'semilogx']
        By default only Linear (ie. raw data) is provided
        """
        out = []
        out.append(["no transform", ["", ""], ""])
        return out

    # Function related to Fits
    def updateFitParam(self, *param, func=None):
        # func: will call func() instead of getattr(self, self.attr("fitfunc"))()
        # used (at least) in curveMCA for fit
        f = self.attr("_fitFunc", "")
        # print ('Curve update', param, 'f', f)
        if f != "" and not self.attributeEqual("_popt"):
            if func is None:
                if hasattr(self, f):
                    func = getattr(self, f)
            if func is not None:
                self.setY(func(self.x(), *param))
                self.update({"_popt": list(self.updateFitParamFormatPopt(f, param))})
                return True
            return "ERROR Update fit parameter: No such fit function (", f, ")."
        return (
            "ERROR Update fit parameter: Empty parameter (_fitFunc:",
            f,
            ", _popt:",
            self.attr("_popt"),
            ").",
        )

    def updateFitParamFormatPopt(self, f, param):
        # possibility to override in subclasses, esp. when handling of test
        # input is required.
        # by default, assumes all numeric -> best stored in a np.array
        return np.array(param)

    # more classical class methods
    def getData(self):
        return self.data

    def shape(self, idx=":"):
        if idx == ":":
            return self.data.shape
        return self.data.shape[idx]

    def _fractionToFloat(self, frac_str):
        try:
            return float(frac_str)
        except ValueError:
            num, denom = frac_str.split("/")
            return float(num) / float(denom)

    def x_offsets(self, **kwargs):
        """
        Same as x(), including the effect of offset and muloffset on output.
        """
        reserved = ["minmax", "0max"]
        special = None
        x = self.x(**kwargs)
        offset = self.attr("offset", None)
        if offset is not None:
            o = offset[0] if isinstance(offset, list) else 0
            if isinstance(o, str):
                if o in reserved:
                    special = o
                    o = 0
                else:
                    o = self._fractionToFloat(o)
            x = x + o
        muloffset = self.attr("muloffset", None)
        if muloffset is not None:
            o = muloffset[0] if isinstance(muloffset, list) else 1
            if isinstance(o, str):
                if o.replace(" ", "") in reserved:
                    special = o
                    o = 1
                else:
                    o = self._fractionToFloat(o)
            x = x * o
        if special is not None:
            m, M = np.min(x), np.max(x)
            if special == "minmax":
                x = (x - m) / (M - m)
            elif special == "0max":
                x = x / M
        return x

    def x(
        self, index=np.nan, alter="", xyValue=None, errorIfxyMix=False, neutral=False
    ):
        """
        Returns the x data.
        index: range to the data.
        alter: possible alteration to the data (i.e. '', 'nmeV')
        xyValue: provide [x, y] value pair to be alter-ed. x, y can be np.array
        errorIfxyMix: throw ValueError exception if alter value calculation
            requires both x and y components. Useful for transforming xlim and
            ylim, where x do not know y in advance.
        neutral: no use yet, introduced to keep same call arguments as self.y()
        """
        if alter != "":
            if alter == "nmeV":
                # invert order of xyValue, to help for graph xlim and ylim
                if xyValue is not None:
                    xyValue = np.array(xyValue)
                    if len(xyValue.shape) > 1:
                        xyValue = [xyValue[0, ::-1], xyValue[1, ::-1]]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return CST.nm_eV / self.x(index, xyValue=xyValue)
            elif alter == "nmcm-1":
                # invert order of xyValue, to help for graph xlim and ylim
                if xyValue is not None:
                    xyValue = np.array(xyValue)
                    if len(xyValue.shape) > 1:
                        xyValue = [xyValue[0, ::-1], xyValue[1, ::-1]]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return 1e7 / self.x(index, xyValue=xyValue)
            elif alter == "MCAkeV":  # deprecated since 0.6.4.1
                offset = self.attr("_MCA_CtokeV_offset", default=0)
                mult = self.attr("_MCA_CtokeV_mult", default=1)
                return mult * (self.x(index, xyValue=xyValue) + offset)
            elif alter == "SIMSdepth":
                offset = self.attr("_SIMSdepth_offset", default=0)
                mult = self.attr("_SIMSdepth_mult", default=1)
                return mult * (self.x(index, xyValue=xyValue) + offset)
            elif alter == "y":
                try:
                    xyValue = xyValue[::-1]
                except TypeError:
                    pass
                # print('x xyValue', xyValue, self.y(index, xyValue=xyValue))
                return self.y(index, xyValue=xyValue)
            elif alter != "":
                split = alter.split(".")
                if len(split) == 2:
                    module_name = (
                        "grapa.datatypes." + split[0][0].lower() + split[0][1:]
                    )
                    try:
                        mod = importlib.import_module(module_name)
                        met = getattr(getattr(mod, split[0]), split[1])
                        return met(self, index=index, xyValue=xyValue)
                    except ImportError as e:
                        msg = "ERROR Curve.x Exception raised during module import {}:"
                        print(msg.format(module_name))
                        print(e)
                else:
                    msg = "Error Curve.x: cannot identify alter keyword ({})."
                    print(msg.format(alter))

        # alter might be used by subclasses
        val = self.data if xyValue is None else np.array(xyValue)
        if len(val.shape) > 1:
            if np.isnan(index).any():
                return val[0, :]
            return val[0, index]
        return val[0]

    def y_offsets(self, **kwargs):
        """
        Same as y(), including the effect of offset and muloffset on output.
        """
        reserved = ["minmax", "0max"]
        special = None
        y = self.y(**kwargs)
        offset = self.attr("offset", None)
        if offset is not None:
            o = offset[1] if isinstance(offset, list) else offset
            if isinstance(o, str):
                if o in reserved:
                    special = o
                    o = 0
                else:
                    o = self._fractionToFloat(o)
            y = y + o
        muloffset = self.attr("muloffset", None)
        if muloffset is not None:
            o = muloffset[1] if isinstance(muloffset, list) else muloffset
            if isinstance(o, str):
                if o.replace(" ", "") in reserved:
                    special = o
                    o = 1
                else:
                    o = self._fractionToFloat(o)
            y = y * o
        if special is not None:
            m, M = np.min(y), np.max(y)
            if special == "minmax":
                y = (y - m) / (M - m)
            elif special == "0max":
                y = y / M
        return y

    def y(
        self, index=np.nan, alter="", xyValue=None, errorIfxyMix=False, neutral=False
    ):
        """
        Returns the y data.
        index: range to the data.
        alter: possible alteration to the data (i.e. '', 'log10abs', 'tauc')
        xyValue: provide [x, y] value pair to be alter-ed. x, y can be np.array
        errorIfxyMix: throw ValueError exception if alter value calculation
            requires both x and y components. Useful for transforming xlim and
            ylim, where x do not know y in advance.
        neutral: additional parameter. If True prevent Jsc substraction for
            CurveJV.
        """

        # if some alteration required: make operation on data requested by
        # itself, without operation
        if alter != "":
            if alter == "log10abs":
                jsc = self.attr("Jsc") if self.attr("Jsc") != "" and not neutral else 0
                return np.log10(np.abs(self.y(index, xyValue=xyValue) + jsc))
            elif alter == "abs":
                jsc = self.attr("Jsc") if self.attr("Jsc") != "" and not neutral else 0
                return np.abs(self.y(index, xyValue=xyValue) + jsc)
            elif alter == "abs0":
                return np.abs(self.y(index, xyValue=xyValue))
            elif alter == "tauc":
                if errorIfxyMix:
                    return ValueError
                return np.power(
                    self.y(index, xyValue=xyValue)
                    * self.x(index, alter="nmeV", xyValue=xyValue),
                    2,
                )
            elif alter == "taucln1-eqe":
                if errorIfxyMix:
                    return ValueError
                return np.power(
                    np.log(1 - self.y(index, xyValue=xyValue))
                    * self.x(index, alter="nmeV", xyValue=xyValue),
                    2,
                )
            elif alter == "normalized":
                out = self.y(index, xyValue=xyValue)
                return out / np.max(out)
            elif alter == "x":
                try:
                    xyValue = xyValue[::-1]
                except TypeError:
                    pass
                return self.x(index, xyValue=xyValue)
            elif alter != "":
                split = alter.split(".")
                if len(split) == 2:
                    module_name = (
                        "grapa.datatypes." + split[0][0].lower() + split[0][1:]
                    )
                    try:
                        mod = importlib.import_module(module_name)
                        met = getattr(getattr(mod, split[0]), split[1])
                        return met(self, index=index, xyValue=xyValue)
                    except ImportError as e:
                        msg = "ERROR Curve.y Exception raised during module import {}:"
                        print(msg.format(module_name))
                        print(e)
                else:
                    if not alter.startswith("idle"):
                        msg = "Error Curve.y: cannot identify alter keyword ({})."
                        print(msg.format(alter))
        # return (subset of) data
        val = self.data if xyValue is None else np.array(xyValue)
        if len(val.shape) > 1:
            if np.isnan(index).any():
                return val[1, :]
            return val[1, index]
        return val[1]

    def setX(self, x, index=np.nan):
        """
        Set new value for the x data.
        Index can be provided (self.data[0,index] = x).
        """
        if np.isnan(index):
            self.data[0, :] = x
        else:
            self.data[0, index] = x

    def setY(self, y, index=np.nan):
        """
        Set new value for the y data.
        Index can be provided (self.data[1,index] = y).
        """
        if len(self.shape()) > 1:
            if np.isnan(index):
                self.data[1, :] = y
            else:
                self.data[1, index] = y
        else:
            self.data[1] = y

    def appendPoints(self, xSeries, ySeries):
        xSeries, ySeries = np.array(xSeries), np.array(ySeries)
        if len(xSeries) == 0:
            print("Curve appendPoints: empty xSeries provided.")
            return False
        if len(ySeries) != len(xSeries):
            print(
                "Curve appendPoints: cannot handle data series with",
                "different lengths.",
            )
            return False
        self.data = np.append(self.data, np.array([xSeries, ySeries]), axis=1)
        return True

    def attr(self, key, default=""):
        """Getter of attribute"""
        if key.lower() in self.attributes:
            return self.attributes[key.lower()]
        return default

    def getAttribute(self, key, default=""):
        """Legacy getter of attribute"""
        return self.attr(key, default=default)

    def getAttributes(self, keysList=None):
        """Returns all attributes, or a subset with keys given in keyList"""
        if isinstance(keysList, list):
            out = {}
            for key in keysList:
                out.update({key: self.attr(key)})
            return out
        return self.attributes

    def attributeEqual(self, key, value=""):
        """
        Check if attribute 'key' has a certain value
        The value can be left empty: check if attr(key) returns the
        default value.
        """
        # value='' must be same as the default parameter of method getAttribute
        val = self.attr(key)
        if isinstance(val, type(value)) and val == value:
            return True
        return False

    def update(self, attributes):
        """Updates attributes. a dict must be provided"""
        for key in attributes:
            k = key.lower()
            if not isinstance(attributes[key], str) or attributes[key] != "":
                k_ = k.strip(" =:\t\n").replace("ï»¿", "")
                self.attributes.update({k_: attributes[key]})
            elif k in self.attributes:
                del self.attributes[k]

    def updateValuesDictkeys(self, *args, **kwargs):
        """
        Performs update({key1: value1, key2: value2, ...}) with
        arguments value1, value2, ... , (*args) and
        kwargument keys=['key1', 'key2', ...]
        """
        if "keys" not in kwargs:
            print(
                'Error Curve updateValuesDictkeys: "keys" key must be',
                "provided, must be a list of keys corresponding to the",
                "values provided in *args.",
            )
            return False
        if len(kwargs["keys"]) != len(args):
            print(
                'WARNING Curve updateValuesDictkeys: len of list "keys"',
                "argument must match the number of provided values (",
                len(args),
                "args provided,",
                len(kwargs["keys"]),
                "keys).",
            )
        lenmax = min(len(kwargs["keys"]), len(args))
        for i in range(lenmax):
            self.update({kwargs["keys"][i]: args[i]})
        return True

    def updateValuesDictkeysGraph(
        self, *args, keys=None, graph=None, alsoAttr=None, alsoVals=None, **kwargs
    ):
        """
        similar as updateValuesDictkeys, for all curves inside a given graph
        """
        from grapa.graph import Graph

        if not isinstance(keys, list):
            print(
                'Error Curve updateValuesDictkeysGraph: "keys" key must be',
                "provided, must be a list of keys corresponding to the",
                "values provided in *args.",
            )
            return False
        if not isinstance(graph, Graph):
            print(
                'Error Curve updateValuesDictkeysGraph: "graph" key must be',
                "provided, must be a Graph object.",
            )
            return False
        if not isinstance(alsoAttr, list):
            print(
                'Error Curve updateValuesDictkeysGraph: "alsoAttr" key must',
                "be provided, must be a list of attributes.",
            )
            return False
        if not isinstance(alsoVals, list):
            print(
                'Error Curve updateValuesDictkeysGraph: "alsoVals" key must',
                "be provided, must be a list of values matching alsoAttr.",
            )
            return False
        while len(alsoVals) < len(alsoAttr):
            alsoVals.append("")
        for curve in graph.iterCurves():
            flag = True
            for i in range(len(alsoAttr)):
                if not curve.attr(alsoAttr[i]) == alsoVals[i]:
                    flag = False
            if flag:
                curve.updateValuesDictkeys(*args, keys=keys)
        return True

    def autoLabel(self, formatter):
        """
        Update the curve label according to formatting using python string template,
        with curve attributes as variables
        e.g. "$sample $_simselement"
        """
        # if modify implementation: beware GraphSIMS, autoLabel(...)
        t = TemplateCustom(formatter)
        try:
            identifiers = t.get_identifiers()
        except AttributeError:  # python < 3.11
            # identifiers must be surrounded by {}
            from string import Formatter

            # identifiers = [ele[1] for ele in Formatter().parse(formatter) if ele[1]]
            identifiers = []
            for ele in Formatter().parse(formatter):
                if ele[1]:
                    identifiers.append(ele[1])
                    if ele[2]:
                        identifiers[-1] = identifiers[-1] + ":" + ele[2]
        attrs = {}
        for key in identifiers:
            attrs[key] = str(self.attr(key))
            if "," in key:  # to supper e.g. "${temperature, :.0f}"
                split = key.split(",")
                ke = split[0].strip()
                fm = split[1].strip()
                if len(fm) > 0:
                    attrs[key] = ("{" + fm + "}").format(self.attr(ke))
                else:
                    attrs[key] = str(self.attr(ke))
        label = t.safe_substitute(attrs)
        self.update({"label": label.replace("  ", " ").strip()})
        return True

    def delete(self, key):
        out = {}
        if key.lower() in self.attributes:
            out.update({key: self.attr("key")})
            del self.attributes[key.lower()]
        return out

    def swapShowHide(self):
        self.update({"linestyle": "" if self.isHidden() else "none"})
        return True

    def isHidden(self):
        return True if self.attr("linestyle") in Curve.LINESTYLEHIDE else False

    def castCurveListGUI(self, onlyDifferent=True):
        curveTypes = [] if onlyDifferent else [Curve]
        curveTypes += Curve.__subclasses__()
        out = []
        for c in curveTypes:
            if not onlyDifferent or not isinstance(self, c):
                name = c.classNameGUI()
                out.append([name, c.__name__, c])
        key = [x[0] for x in out]
        for i in range(len(key) - 2, -1, -1):
            if key[i] in key[:i]:
                del key[i]
                del out[i]
        # sort list by name
        out = [x for (y, x) in sorted(zip(key, out), key=lambda pair: pair[0])]
        return out

    def castCurve(self, newTypeGUI):
        if newTypeGUI == "Curve":
            newTypeGUI = "Curve"
        if type(self).__name__ == newTypeGUI:
            return self
        curveTypes = self.castCurveListGUI()
        for typ_ in curveTypes:
            if newTypeGUI == "Curve":
                return Curve(self.data, attributes=self.getAttributes(), silent=True)
            if (
                newTypeGUI == typ_[0]
                or newTypeGUI == typ_[1]
                or newTypeGUI.replace(" ", "") == typ_[1]
            ):
                if isinstance(self, typ_[2]):
                    print("Curve.castCurve: already that type!")
                else:
                    return typ_[2](
                        self.data, attributes=self.getAttributes(), silent=True
                    )
                break
        print(
            "Curve.castCurve: Cannot understand new type:",
            newTypeGUI,
            "(possible types:",
            [key for key in curveTypes],
            ")",
        )
        return False

    def selectData(self, xlim=None, ylim=None, alter="", offsets=False, data=None):
        """
        Returns slices of the data self.x(), self.y() which satisfy
        xlim[0] <= x <= xlim[1], and ylim[0] <= y <= ylim[1]
        alter: '', or ['', 'abs']. Affect the test AND the output.
        offset: if True, calls x and y are affected by offset and muloffset
        data [xSeries, ySeries] can be provided. Helps the code of calling
            methods being cleaner. If provided, alter and offsets are ignored.
        """
        if isinstance(alter, str):
            alter = ["", alter]
        # retrieves entry data
        if data is None:
            if offsets:
                x, y = self.x_offsets(alter=alter[0]), self.y_offsets(alter=alter[1])
            else:
                x, y = self.x(alter=alter[0]), self.y(alter=alter[1])
        else:
            x, y = np.array(data[0]), np.array(data[1])
        # identify boundaries
        if xlim is None:
            xlim = [min(x), max(x)]
        if ylim is None:
            ylim = [min(y), max(y)]
        # construct a data mask
        mask = np.ones(len(x), dtype=bool)
        for i in range(len(mask)):
            if x[i] < xlim[0] or x[i] > xlim[1]:
                mask[i] = False
            if y[i] < ylim[0] or y[i] > ylim[1]:
                mask[i] = False
        return x[mask], y[mask]

    def getPointClosestToXY(self, x, y, alter="", offsets=False):
        """
        Return the data point closest to the x,y values.
        Priority on x, compares y only if equal distance on x
        """
        if isinstance(alter, str):
            alter = ["", alter]
        # select most suitable point based on x
        datax = self.x_offsets(alter=alter[0])
        absm = np.abs(datax - x)
        idx = np.where(absm == np.min(absm))
        if len(idx) == 0:
            idx = np.argmin(absm)
        elif len(idx) == 1:
            idx = idx[0]
        else:
            # len(idx) > 1: select most suitable point based on y
            datay = self.y_offsets(index=idx, alter=alter[1])
            absM = np.abs(datay - y)
            idX = np.where(absM == np.min(absM))
            if len(idX) == 0:
                idx = idx[0]
            elif len(idX) == 1:
                idx = idx[idX[0]]
            else:  # equally close in x and y -> returns first datapoint found
                idx = idx[idX[0]]
        idxOut = idx if len(idx) <= 1 else idx[0]
        if offsets:
            # no alter, but offset for the return value
            return self.x_offsets(index=idx)[0], self.y_offsets(index=idx)[0], idxOut
        # no alter, no offsets for the return value
        return self.x(index=idx)[0], self.y(index=idx)[0], idxOut

    def getDataCustomPickerXY(self, idx, alter="", strDescription=False):
        """
        Given an index in the Curve data, returns a modified data which has
        some sense to the user. Is overloaded in some child Curve classes, see
        for example CurveCf
        If strDescription is True, then returns a string which describes what
        this method is doing.
        """
        if strDescription:
            return "(x,y) data with offset and transform"
        if isinstance(alter, str):
            alter = ["", alter]
        attr = {}
        return (
            self.x_offsets(index=idx, alter=alter[0]),
            self.y_offsets(index=idx, alter=alter[1]),
            attr,
        )

    @staticmethod
    def printHelpFunc(func, leadingstrings=None):
        """prints the docstring of a function"""
        if leadingstrings is None:
            leadingstrings = ["- ", "  "]
        a, idx = 0, None
        for line in func.__doc__.split("\n"):
            if len(line) == 0:
                continue
            if idx is None:
                idx = len(line) - len(line.lstrip(" "))
            lin_ = line[idx:]
            if len(lin_) == 0:
                continue
            # do not want to keep the reStructuredText formatting in printout
            if lin_.startswith(":param "):
                lin_ = "-" + lin_[6:]
            if lin_.startswith(":return:"):
                lin_ = "Returns:" + lin_[8:]
            print(leadingstrings[a] + lin_)
            a = 1

    # some arithmetics
    def __add__(self, other, sub=False, interpolate=-1, offsets=False, operator="add"):
        """
        Addition operation, element-wise or interpolated along x data
        interpolate: interpolate, or not the data on the x axis
            0: no interpolation, only consider y() values
            1: interpolation, x are the points contained in self.x() or in
               other.x()
            2: interpolation, x are the points of self.x(), restricted to
               min&max of other.x()
            -1: 0 if same y values (gain time), otherwise 1
        With sub=True, performs substraction
        offsets: if True, adds Curves after computing offsets and muloffsets.
            If not adds on the raw data.
        operator: 'add', 'sub', 'mul', 'div'. sub=True is a shortcut for
            operatore='sub'. Overriden by sub=True argument.
        """

        def op(x, y, operator):
            if operator == "sub":
                return x - y
            if operator == "mul":
                return x * y
            if operator == "div":
                return x / y
            if operator == "pow":
                return x**y
            if operator != "add":
                print(
                    "WARNING Curve.__add__: unexpected operator argument("
                    + operator
                    + ")."
                )
            return x + y

        if sub == True:
            operator = "sub"
        selfx = self.x_offsets if offsets else self.x
        selfy = self.y_offsets if offsets else self.y
        if not isinstance(other, Curve):  # add someting/number to a Curve
            out = Curve([selfx(), op(selfy(), other, operator)], self.getAttributes())
            # remove offset information if use it during calculation
            if offsets:
                out.update({"offset": "", "muloffset": ""})
            out = out.castCurve(self.classNameGUI())
            return out  # cast type
        # curve1 is a Curve
        otherx = other.x_offsets if offsets else other.x
        othery = other.y_offsets if offsets else other.y
        # default mode -1: check if can gain time (avoid interpolating)
        r = range(0, min(len(selfy()), len(othery())))
        if interpolate == -1:
            interpolate = 0 if np.array_equal(selfx(index=r), otherx(index=r)) else 1
        # avoiding interpolation
        if not interpolate:
            le = min(len(selfy()), len(othery()))
            r = range(0, le)
            if le < len(selfy()):
                print(
                    "WARNING Curve __add__: Curves not same lengths, clipped",
                    "result to shortest (",
                    len(selfy()),
                    ",",
                    len(othery()),
                    ")",
                )
            if not np.array_equal(selfx(index=r), otherx(index=r)):
                print(
                    "WARNING Curve __add__ (" + operator + "): Curves not same x",
                    "axis values. Consider interpolation (interpolate=1).",
                )
            out = Curve(
                [selfx(index=r), op(selfy(index=r), othery(index=r), operator)],
                other.getAttributes(),
            )
            out.update(self.getAttributes())
            if offsets:  # remove offset information if use during calculation
                out.update({"offset": "", "muloffset": ""})
            out = out.castCurve(self.classNameGUI())
            return out
        else:  # not elementwise : interpolate
            from scipy.interpolate import interp1d

            # construct new x -> all x which are in the range of the other curv
            datax = list(selfx())
            if interpolate == 1:  # x from both self and other
                xmin = max(min(selfx()), min(otherx()))
                xmax = min(max(selfx()), max(otherx()))
                # no duplicates
                datax += [x for x in otherx() if x not in datax]
            else:
                # interpolate 2: copy x from self, restrict to min&max of other
                xmin, xmax = min(otherx()), max(otherx())
            datax = [x for x in datax if x <= xmax and x >= xmin]
            reverse = (selfx(index=0) > selfx(index=1)) if len(selfx()) > 1 else False
            datax.sort(reverse=reverse)
            f0 = interp1d(selfx(), selfy(), kind=1)
            f1 = interp1d(otherx(), othery(), kind=1)
            datay = [op(f0(x), f1(x), operator) for x in datax]
            out = Curve([datax, datay], other.getAttributes())
            out.update(self.getAttributes())
            if offsets:  # remove offset information if use during calculation
                out.update({"offset": "", "muloffset": ""})
            out = out.castCurve(self.classNameGUI())
            return out

    def __radd__(self, other, **kwargs):
        return self.__add__(other, **kwargs)

    def __sub__(self, other, **kwargs):
        """substract operation, element-wise or interpolated"""
        kwargs.update({"sub": True})
        return self.__add__(other, **kwargs)

    def __rsub__(self, other, **kwargs):
        """reversed substract operation, element-wise or interpolated"""
        kwargs.update({"sub": False, "operator": "add"})
        return Curve.__add__(self.__neg__(), other, **kwargs)

    def __mul__(self, other, **kwargs):
        """multiplication operation, element-wise or interpolated"""
        kwargs.update({"operator": "mul"})
        return self.__add__(other, **kwargs)

    def __rmul__(self, other, **kwargs):
        kwargs.update({"operator": "mul"})
        return self.__add__(other, **kwargs)

    def __div__(self, other, **kwargs):
        """division operation, element-wise or interpolated"""
        kwargs.update({"operator": "div"})
        return self.__add__(other, **kwargs)

    def __rdiv__(self, other, **kwargs):
        """division operation, element-wise or interpolated"""
        kwargs.update({"operator": "mul"})
        return Curve.__add__(self.__invertArithmetic__(), other, **kwargs)

    def __truediv__(self, other, **kwargs):
        """division operation, element-wise or interpolated"""
        kwargs.update({"operator": "div"})
        return self.__add__(other, **kwargs)

    def __rtruediv__(self, other, **kwargs):
        """division operation, element-wise or interpolated"""
        kwargs.update({"operator": "mul"})
        return Curve.__add__(self.__invertArithmetic__(), other, **kwargs)

    def __pow__(self, other, **kwargs):
        """power operation, element-wise or interpolated"""
        kwargs.update({"operator": "pow"})
        return self.__add__(other, **kwargs)

    def __neg__(self, **kwargs):
        out = Curve([self.x(), -self.y()], self.getAttributes())
        out = out.castCurve(self.classNameGUI())
        return out

    def __invertArithmetic__(self, **_kwargs):
        out = Curve([self.x(), 1 / self.y()], self.getAttributes())
        out = out.castCurve(self.classNameGUI())
        return out

    # plot function
    def plot(
        self, ax, groupedplotters, graph=None, graph_i=None, type_plot="", ignoreNext=0
    ):
        """
        plot a Curve on some axis
        groupedplotters: to handle boxplots, violinplots and such (see plot_graph_aux.py)
        graph, graph_i: a graph instance, such that self==graph.curve(graph_i)
            required to properly plot scatter with scatter_c, etc.
        alter: '', or ['nmeV', 'abs']
        type_plot: 'semilogy'
        ignoreNext: int, counter to decide whether the next curves shall not be
            plotted (multi-Curve plotting such as scatter)
        """
        from grapa.graph import Graph

        handle = None

        ifplotdata = True
        if self.attr("curve") == "subplot":
            ifplotdata = False
        if not ifplotdata:
            return None, ignoreNext

        # check default arguments
        if graph is None:
            graph_i = None
        else:
            if graph[graph_i] != self:
                graph_i = None
            if graph_i is None:
                for c in range(len(graph)):
                    if graph[c] == self:
                        graph_i = c
                        break
            if graph_i is None:
                graph = None  # self was not found in graph
                print("Warning Curve.plot: Curve not found in provided Graph")

        # retrieve basic information
        alter = graph.get_alter() if graph is not None else ["", ""]
        attr = self.getAttributes()
        linespec = self.attr("linespec")
        # construct dict of keywords based on curves attributes, in a very
        # restrictive way
        # some attributes are commands for plotting, some are just related to
        # the sample, and no obvious way to discriminate between the 2
        fmt = {}
        for key in attr:
            if not isinstance(key, str):
                print(type(key), key, attr[key])
            if (
                (not isinstance(attr[key], str) or attr[key] != "")
                and key in Graph.KEYWORDS_CURVE["keys"]
                and key
                not in [
                    "plot",
                    "linespec",
                    "type",
                    "ax_twinx",
                    "ax_twiny",
                    "offset",
                    "muloffset",
                    "labelhide",
                    "colorbar",
                    "xerr",
                    "yerr",
                ]
            ):
                fmt[key] = attr[key]
        # do not plot curve if was asked not to display it.
        if "linestyle" in fmt and fmt["linestyle"] in Curve.LINESTYLEHIDE:
            return None, ignoreNext
        # some renaming of kewords, etc
        if "legend" in fmt:
            fmt["label"] = fmt["legend"]
            del fmt["legend"]
        if "cmap" in fmt and not isinstance(fmt["cmap"], str):
            # convert Colorscale into matplotlib cmap
            from grapa.colorscale import Colorscale

            fmt["cmap"] = Colorscale(fmt["cmap"]).cmap()
        if "vminmax" in fmt:
            if isinstance(fmt["vminmax"], list) and len(fmt["vminmax"]) > 1:
                if (
                    fmt["vminmax"][0] != ""
                    and not np.isnan(fmt["vminmax"][0])
                    and not np.isinf(fmt["vminmax"][0])
                ):
                    fmt.update({"vmin": fmt["vminmax"][0]})
                if (
                    fmt["vminmax"][1] != ""
                    and not np.isnan(fmt["vminmax"][1])
                    and not np.isinf(fmt["vminmax"][1])
                ):
                    fmt.update({"vmax": fmt["vminmax"][1]})
            del fmt["vminmax"]

        # start plotting
        # retrieve data after transform, including of offset and muloffset
        x = self.x_offsets(alter=alter[0])
        y = self.y_offsets(alter=alter[1])
        type_graph = self.attr("type", "plot")
        if type_plot.endswith(" norm."):
            type_graph = type_plot[:-6]
            y = y / max(y)

        # add keyword arguments which are in the plot method prototypes
        try:
            sig = inspect.signature(getattr(ax, type_graph))
            for key in sig.parameters:
                if key in attr and key not in fmt:
                    fmt.update({key: attr[key]})
        except AttributeError:
            print(
                "Curve.plot: desired plotting method not found ("
                + type_graph
                + "). Going for default."
            )
            # for xample 'errorbar_yerr' after suppression of previous Curve
            # 'errorbar'. Will be 'plot' anyway.
            pass
        except Exception as e:
            print("Exception in Curve.plot while identifying keyword", "arguments:")
            print(type(e), e)

        if "labelhide" in attr and attr["labelhide"]:
            if "label" in fmt:
                del fmt["label"]

        # No support for the following methods (either 2D data, or complicated
        # to implement):
        #    hlines, vlines, broken_barh, polar,
        #    pcolor, pcolormesh, streamplot, tricontour, tricontourf,
        #    tripcolor
        # Partial support for:
        #    imgshow, contour, contourf (part of Curve_Image)
        attrIgnore = [
            "label",
            "plot",
            "linespec",
            "type",
            "ax_twinx",
            "ax_twiny",
            "offset",
            "muloffset",
            "labelhide",
            "colorbar",
        ]

        def _curvedata_same_x(x_, range_, graph_, alter_, where_):
            # function used for bar, bah. Certainly others make good use of it
            if graph is None:
                return None, None, None
            for j_ in range_:
                if graph_[j_].visible():
                    flag = True
                    for key_, value_ in where_.items():
                        if graph_[j_].attr(key_) != value_:
                            flag = False
                    if not flag:
                        continue
                    x2_ = graph_[j_].x_offsets(alter=alter_[0])
                    if np.array_equal(x_, x2_):
                        y2_ = graph_[j_].y_offsets(alter=alter_[1])
                        return j_, x2_, y2_
            return None, None, None

        # "simple" plotting methods, with prototype similar to plot()
        if type_graph in [
            "semilogx",
            "semilogy",
            "loglog",
            "plot_date",
            "stem",
            "step",
            "triplot",
        ]:
            handle = getattr(ax, type_graph)(x, y, linespec, **fmt)
        elif type_graph in ["fill"]:
            if self.attr("fill_padto0", False):
                handle = ax.fill(
                    [x[0]] + list(x) + [x[-1]], [0] + list(y) + [0], linespec, **fmt
                )
            else:
                handle = ax.fill(x, y, linespec, **fmt)
        # plotting methods not accepting formatting string as 3rd argument
        elif type_graph in [
            # "bar",  "barh",  "fill_between", "fill_betweenx"
            "barbs",
            "cohere",
            "csd",
            "hexbin",
            "hist2d",
            "quiver",
            "xcorr",
        ]:
            handle = getattr(ax, type_graph)(x, y, **fmt)
        elif type_graph in ["bar", "barh"]:
            if graph is not None:
                key = "bottom" if type_graph == "bar" else "left"
                value = self.attr(key)
                range_ = None
                if value == "bar_first":
                    range_ = range(graph_i)
                elif value == "bar_previous":
                    range_ = range(graph_i - 1, -1, -1)
                elif value == "bar_next":
                    range_ = range(graph_i + 1, len(graph))
                elif value == "bar_last":
                    range_ = range(len(graph) - 1, graph_i, -1)
                if range_ is not None:
                    argssamex = [graph, alter, {"type": type_graph}]
                    j, x2, y2 = _curvedata_same_x(x, range_, *argssamex)
                    if j is not None:
                        fmt.update({key: y2})
                    else:
                        msg = "Curve.plot {}: no suitable Curve found ({}, {}, {})"
                        print(msg.format(type_graph, graph_i, key, value))
            handle = getattr(ax, type_graph)(x, y, **fmt)
        elif type_graph in ["fill_between", "fill_betweenx"]:
            success = False
            if graph is not None and len(graph) > graph_i + 1:
                x2 = graph[graph_i + 1].x_offsets(alter=alter[0])
                y2 = graph[graph_i + 1].y_offsets(alter=alter[1])
                if not np.array_equal(x, x2):
                    msg = (
                        "WARNING Curve {} and {}: fill_between, fill_betweenx: x "
                        "series must be equal. Fill to 0."
                    )
                    print(msg.format(graph_i, graph_i + 1))
                else:
                    ignoreNext += 1
                    success = True
                    handle = getattr(ax, type_graph)(x, y, y2, **fmt)
            if not success:
                handle = getattr(ax, type_graph)(x, y, **fmt)
        #  plotting of single vector data
        elif type_graph in [
            "acorr",
            "angle_spectrum",
            "eventplot",
            "hist",
            "magnitude_spectrum",
            "phase_spectrum",
            "pie",
            "psd",
            "specgram",
        ]:
            # careful with eventplot, the Curve data are modified
            handle = getattr(ax, type_graph)(y, **fmt)
        # a more peculiar plotting
        elif type_graph in ["spy"]:
            handle = getattr(ax, type_graph)([x, y], **fmt)
        elif type_graph == "stackplot":
            # look for next Curves with type == 'stackplot', and same x
            nexty = []
            fmt["labels"], fmt["colors"] = [""], [""]
            if "label" in fmt:
                fmt["labels"] = ["" if self.attr("labelhide") else fmt["label"]]
                del fmt["label"]
            if "color" in fmt:
                fmt["colors"] = [fmt["color"]]
                del fmt["color"]
            attrIgnore.append("color")
            if graph is not None:
                for j in range(graph_i + 1, len(graph)):
                    if graph[j].attr("type") == type_graph and np.array_equal(
                        x, graph[j].x_offsets(alter=alter[0])
                    ):
                        ignoreNext += 1
                        if graph[j].visible():
                            nexty.append(graph[j].y_offsets(alter=alter[1]))
                            lbl = graph[j].attr("label")
                            fmt["labels"].append(
                                "" if graph[j].attr("labelhide") else lbl
                            )
                            fmt["colors"].append(graph[j].attr("color"))
                            continue
                    else:
                        break
            if np.all([(c == "") for c in fmt["colors"]]):
                del fmt["colors"]
            handle = getattr(ax, type_graph)(x, y, *nexty, **fmt)
        elif type_graph == "errorbar":
            # look for next Curves, maybe xerr/yerr was provided
            if "xerr" in attr:
                fmt.update({"yerr": attr["xerr"]})
            if "yerr" in attr:
                fmt.update({"yerr": attr["yerr"]})
            if graph is not None:
                for j in range(graph_i + 1, min(graph_i + 3, len(graph))):
                    if len(graph[j].y()) == len(y):
                        typenext = graph[j].attr("type")
                        if typenext not in ["errorbar_xerr", "errorbar_yerr"]:
                            break
                        if typenext == "errorbar_xerr":
                            fmt.update({"xerr": graph[j].y_offsets()})
                            ignoreNext += 1
                            continue
                        if typenext == "errorbar_yerr":
                            fmt.update({"yerr": graph[j].y_offsets()})
                            ignoreNext += 1
                            continue
                    break
            handle = ax.errorbar(x, y, fmt=linespec, **fmt)
        elif type_graph == "scatter":
            convert = {"markersize": "s", "markeredgewidth": "linewidths"}
            for key in convert:
                if key in fmt:
                    fmt.update({convert[key]: fmt[key]})
                    del fmt[key]
            try:
                if graph is not None:
                    for j in range(graph_i + 1, min(graph_i + 3, len(graph))):
                        typenext = graph[j].attr("type")
                        if typenext not in ["scatter_c", "scatter_s"]:
                            break
                        if "s" not in fmt and typenext == "scatter_s":
                            fmt.update({"s": graph[j].y_offsets(alter=alter[1])})
                            ignoreNext += 1
                            continue
                        elif "c" not in fmt and (
                            typenext == "scatter_c"
                            or np.array_equal(x, graph[j].x_offsets(alter=alter[0]))
                        ):
                            fmt.update({"c": graph[j].y_offsets(alter=alter[1])})
                            ignoreNext += 1
                            if "color" in fmt:
                                # there cannot be both c and color keywords
                                del fmt["color"]
                            continue
                        else:
                            break
                handle = ax.scatter(x, y, **fmt)
            except Exception as e:
                msg = "ERROR! Exception occured in Curve.plot function during scatter."
                print(msg)
                print(type(e), e)
        elif type_graph in ["boxplot", "violinplot"]:
            handle = groupedplotters.add_curve(type_graph, self, y, fmt, ax)
        elif type_graph in ["imshow", "contour", "contourf"]:
            from grapa.curve_image import Curve_Image

            img, ignoreNext, X, Y = Curve_Image.get_image_data(
                self, graph, graph_i, alter, ignoreNext
            )
            if "label" in fmt:
                del fmt["label"]
            if type_graph in ["contour", "contourf"]:
                for key in [
                    "corner_mask",
                    "colors",
                    "alpha",
                    "cmap",
                    "norm",
                    "vmin",
                    "vmax",
                    "levels",
                    "origin",
                    "extent",
                    "locator",
                    "extend",
                    "xunits",
                    "yunits",
                    "antialiased",
                    "nchunk",
                    "linewidths",
                    "linestyles",
                    "hatches",
                ]:
                    if key in attr and key not in fmt:
                        fmt.update({key: attr[key]})
                # TODO: remove linewidths, linestyles for contourf, hatches for
                # contour
            args = [img]
            if (
                X is not None
                and Y is not None
                and type_graph in ["contour", "contourf"]
            ):
                args = [X, Y] + args
            try:
                handle = getattr(ax, type_graph)(*args, **fmt)
            except Exception as e:
                print("Curve plot", type_graph, "Exception")
                print(type(e), e)
        else:
            # default is plot (lin-lin) # also valid if no information is
            # stored, aka returned ''
            handle = ax.plot(x, y, linespec, **fmt)

        handles = handle if isinstance(handle, list) else [handle]
        for key in attr:
            if key not in fmt and key not in attrIgnore:
                for h in handles:
                    if hasattr(h, "set_" + key):
                        try:
                            getattr(h, "set_" + key)(attr[key])
                        except Exception as e:
                            print(
                                "GraphIO Exception during plot kwargs",
                                "adjustment for key",
                                key,
                                ":",
                                type(e),
                            )
                            print(e)

        return handle, ignoreNext
