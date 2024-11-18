# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:19:52 2017

@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
from copy import deepcopy
import numpy as np
import matplotlib

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.gui.GUIFuncGUI import FuncGUI


class Curve_Image(Curve):
    """
    The purpose is this class is to provid GUI support to create images, contour plots
    etc.
    """

    CURVE = "image"

    def __init__(self, *args, **kwargs):
        Curve.__init__(self, *args, **kwargs)
        # define default values for important parameters
        if self.attr("type") not in ["imshow", "contour", "contourf"]:
            self.update({"type": "imshow"})
        # legacy keyword
        imagefile = deepcopy(self.attr("imagefile", None))
        if imagefile is not None:
            self.update({"datafile": imagefile, "imagefile": ""})
        self.update({"Curve": Curve_Image.CURVE})

    # GUI RELATED FUNCTIONS
    def funcListGUI(self, **kwargs):
        graph = None
        if "graph" in kwargs:
            graph = kwargs["graph"]

        out = Curve.funcListGUI(self, **kwargs)
        # curve image type
        texttype, default = "keyword 'type'", self.attr("type")
        type_values = ["imshow", "contour", "contourf"]
        if default not in type_values:
            texttype = "issue: keyword 'type' should be:"
            default = "imshow"
        out.append(FuncGUI(self.updateValuesDictkeys, "Set", {"keys": ["type"]}))
        out[-1].append(
            texttype, default, options={"field": "Combobox", "values": type_values}
        )

        interpolation = [
            "",
            "none",
            "nearest",
            "bilinear",
            "bicubic",
            "spline16",
            "spline36",
            "hanning",
            "hamming",
            "hermite",
            "kaiser",
            "quadric",
            "catrom",
            "gaussian",
            "bessel",
            "mitchell",
            "sinc",
            "lanczos",
        ]
        aspect = ["", "1 scalar", "auto", "equal"]
        datafile_xy1rowcol = bool(self.attr("datafile_XY1rowcol"))
        datafile_dataasxyz = bool(self.attr("datafile_dataasxyz"))
        # file
        out.append(FuncGUI(self.updateValuesDictkeys, "Set", {"keys": ["datafile"]}))
        out[-1].append("data file", self.attr("datafile"), options={"width": 40})
        # X,Y 1st row column; data as XYZ
        out.append(
            [
                self.updateValuesDictkeys,
                "Set",
                ["first row, column as coordinates", "data as 3-colum XYZ"],
                [datafile_xy1rowcol, datafile_dataasxyz],
                {"keys": ["datafile_XY1rowcol", "datafile_dataasxyz"]},
                [{"field": "Checkbutton"}, {"field": "Checkbutton"}],
            ]
        )
        # transpose, rotate
        at = ["transpose", "rotate"]
        out.append(
            [
                self.updateValuesDictkeys,
                "Set",
                at,
                [self.attr(a) for a in at],
                {"keys": at},
                [{"field": "Combobox", "values": ["", "True", "False"]}, {}],
            ]
        )
        # extent
        if not datafile_xy1rowcol:
            extent = list(self.attr("extent"))
            while len(extent) < 4:
                extent.append("")
            out.append(
                [
                    self.updateExtent,
                    "Set",
                    ["extent left", "right", "bottom", "top"],
                    extent,
                ]
            )
        else:
            msg = "extent: not active if first row, column as coordinates"
            out.append([None, msg, [], []])
        # colorscale
        at = ["colorbar"]
        out.append(FuncGUI(self.updateValuesDictkeys, "Set", {"keys": at}))
        out[-1].append(at[0], self.attr(at[0]), options={"width": 50})
        # colormap
        attxt = ["cmap (ignored if color image)", "vmin", "vmax"]
        at = [a.split(" ")[0] for a in attxt]
        out.append(
            [
                self.updateValuesDictkeys,
                "Set",
                attxt,
                [self.attr(a) for a in at],
                {"keys": at},
            ]
        )
        # aspect ratio, interpolation, or levels
        if self.attr("type") == "imshow":
            at = ["aspect", "interpolation"]
            out.append(FuncGUI(self.updateValuesDictkeys, "Set", {"keys": at}))
            out[-1].append(
                "aspect ratio",
                self.attr(at[0]),
                options={"field": "Combobox", "values": aspect, "bind": "beforespace"},
            )
            out[-1].append(
                "interpolation",
                self.attr(at[1]),
                options={"field": "Combobox", "values": interpolation},
            )
        else:  # contour, contourf
            at = ["levels", "extend", "norm"]
            optse = {"field": "Combobox", "values": ["neither", "both", "min", "max"]}
            optsn = {
                "field": "Combobox",
                "values": list(matplotlib.scale.get_scale_names()),
            }
            out.append(FuncGUI(self.updateValuesDictkeys, "Set", {"keys": at}))
            out[-1].append("levels (list of values)", self.attr(at[0]))
            out[-1].append("extend", self.attr(at[1]), options=optse)
            out[-1].append(", norm", self.attr(at[2]), options=optsn)

            levels = self.attr("levels")
            if len(levels) == 0:
                levels = [np.min(self.x()), np.max(self.y())]
            out.append(FuncGUI(self.updateLevels, "Set", {}))
            choice = ["arange", "linspace", "logspace", "geomspace"]
            opts = {"field": "Combobox", "values": choice}
            out[-1].append("levels:", choice[1], options=opts)
            out[-1].append("start", np.min(levels))
            out[-1].append("stop", np.max(levels))
            out[-1].append("num or step", len(levels))

        # convert data matrix to 3 column xyz and vice-versa
        if graph is not None:
            if datafile_dataasxyz:
                out.append(
                    FuncGUI(self.convert_xyz_matrix, "Convert", {"graph": graph})
                )
                out[-1].append(
                    "xyz 3-column format into matrix", "", options={"field": "Label"}
                )
            else:
                out.append(
                    FuncGUI(self.convert_matrix_xyz, "Convert", {"graph": graph})
                )
                out[-1].append(
                    "matrix into xyz 3-column format", "", options={"field": "Label"}
                )

        # help
        out.append(FuncGUI(self.printHelp, "Help!"))
        return out

    def updateExtent(self, *args):
        flag = False
        for a in args:
            if a != "":
                flag = True
        self.update({"extent": (list(args) if flag else "")})

    def updateLevels(self, typ, start, stop, step):
        try:
            if typ == "arange":
                levels = np.arange(start, stop, step)
            elif typ == "linspace":
                levels = np.linspace(start, stop, int(step))
            elif typ == "logspace":
                levels = np.logspace(start, stop, int(step))
            elif typ == "geomspace":
                levels = np.geomspace(start, stop, int(step))
            else:
                msg = "Curve image updateLevels, input value not recognized '{}'."
                return msg.format(typ)
        except ValueError as e:
            print("Exception", type(e), e)
            return False
        if len(levels) == 0:
            levels = ""
        if len(levels) > 1000:
            msg = (
                "Curve image updateLevels, grapa won't generate that many points ({})."
            )
            return msg.format(len(levels))
        self.update({"levels": list(levels)})
        return True

    def convert_matrix_xyz(self, *args, graph=None):
        """Converts a 2D data meshgrid Z[x, y] into 3 columns X, Y, Z"""
        # args: to catch fake argument for GUI purpose
        data, num = self.aggregate_into_matrix(graph=graph)
        data = data.transpose()
        x, y = range(data.shape[0]), range(data.shape[1])
        xyz = self.x_y_z_to_xyz(x, y, data)
        attrs = self.getAttributes()
        out = Graph()
        out.append(Curve([xyz[:, 0], xyz[:, 1]], attrs))
        out.append(Curve([xyz[:, 0], xyz[:, 2]], {}))
        curvetype = out[0].attr("curve")
        if curvetype != "":
            out.castCurve(curvetype, 0)
        out[0].update({"datafile_dataasxyz": True})
        return out  # [c for c in out]

    def convert_xyz_matrix(self, *args, graph=None):
        print("Curve Image convert_xyz_matrix: About to loose x, y information.")
        data, num = self.aggregate_into_matrix(graph=graph)
        if data.shape[0] < 3:
            print("ERROR Curve Image convert_xyz_matrix: not enough data columns.")
            return False
        vectorx, vectory, matrix = self.xyz_to_x_y_z(data)
        out = Graph()
        for x in range(1, matrix.shape[0]):
            out.append(Curve([matrix[0, :], matrix[x, :]], {}))
        out[0].update(self.getAttributes())
        curvetype = out[0].attr("curve")
        if curvetype != "":
            out.castCurve(curvetype, 0)
        out[0].update({"datafile_dataasxyz": False})
        return out  # [c for c in out]

    @staticmethod
    def xyz_to_x_y_z(data):
        """
        Interpret a 3-column XYZ array into X, Y (both 1D vectors of x, y values) and
        Z (meshgrid format)
        """

        def get_z(data, x, y):
            ind = (data[0, :] == x) * (data[1, :] == y)
            if sum(ind) == 0:
                msg = (
                    "Warning xyz_to_x_y_z: Missing datapoint, must be regular mesh "
                    "e.g. meshgrid. Coordinates {}, {}."
                )
                print(msg.format(x, y))
                return np.nan
            if sum(ind) > 1:
                msg = (
                    "Warning xyz_to_x_y_z: Duplicate datapoint, must be regular "
                    "mesh e.g. meshgrid. Coordinates {}, {}."
                )
                print(msg.format(x, y))
            return data[2, ind][0]

        # probably not the fastest algorithm
        X1, Y1 = np.unique(data[0, :]), np.unique(data[1, :])
        X, Y = np.meshgrid(X1, Y1)
        z = np.array([get_z(data, x, y) for (x, y) in zip(np.ravel(X), np.ravel(Y))])
        Z = z.reshape(X.shape)
        return X1, Y1, Z
        # return X, Y, Z

    @staticmethod
    def x_y_z_to_xyz(vectorx, vectory, matrixz):
        """
        From vectorx, vectory and Z (meshgrid format),
        Returns a 3-column XYZ array
        """
        if len(np.array(vectorx).shape) != 1 or len(np.array(vectory).shape) != 1:
            msg = "ERROR x_y_z_to_xyz, X and Y must be 1D vectors. Shape: {}, {}."
            print(msg.format(np.array(vectorx).shape, np.array(vectory).shape))
            return False
        data = []
        for x in range(len(vectorx)):
            for y in range(len(vectory)):
                data.append([x, y, matrixz[x, y]])
        return np.array(data)

    def aggregate_into_matrix(self, graph: Graph, alter: list = None):
        """
        see aggregate_samex_into_matrix_i
        starts at current curve index
        """
        # identify curve index within Graph
        start_i = None
        for i in range(len(graph)):
            if graph[i] == self:
                start_i = i
                break
        if start_i is None:
            msg = (
                "ERROR Curve Image aggregate_samex_into_matrix: cannot find curve "
                "in provided graph."
            )
            print(msg)
            raise RuntimeError
        # aggregate data
        return Curve_Image.aggregate_into_matrix_i(graph, start_i, alter=alter)

    @staticmethod
    def aggregate_into_matrix_i(graph: Graph, start_i: int, alter: list = None):
        """
        aggregates content of several curves into a data matrix.
        starts at given curve index
        """
        if alter is None:
            alter = ["", ""]
        # aggregate data
        x = graph[start_i].x_offsets(alter=alter[0])
        y = graph[start_i].y_offsets(alter=alter[1])
        data = [x, y]
        numcurves = 1
        for j in range(start_i + 1, len(graph)):
            if graph[j].isHidden():
                numcurves += 1
                continue
            # test equal accepting nan, w/o using equal_nan=True for retro compatibility
            test = graph[j].x_offsets(alter=alter[0])
            if len(x) != len(test):
                break
            mask = ~(np.isnan(x) * np.isnan(test))
            if np.array_equal(x[mask], test[mask]):
                data.append(graph[j].y_offsets(alter=alter[1]))
                numcurves += 1
            else:
                break
        return np.array(data), numcurves

    @staticmethod
    def process_matrix(data, x, y, xy1rowcol=False, transpose=False, rotate=False):
        """
        Perform header label extraction, transposition and rotation on data matrix
        """
        if xy1rowcol:
            try:
                x = data[0, 1:]
                y = data[1:, 0]
                data = data[1:, 1:]
            except:
                pass
        if transpose:
            data = np.transpose(data)
            if x is not None and y is not None:
                swap = x
                x = y
                y = swap
        if rotate:
            data = np.rot90(data, k=int(rotate))  # only by 90°
            if x is not None and y is not None:
                nrot = int(rotate) % 4
                while nrot > 0:
                    swap = x
                    x = y
                    y = swap[::-1]
                    nrot -= 1
        return data, x, y

    def getImageData(self, graph, graph_i, alter, ignoreNext=0):
        """
        graph: the Graph the Curve is in
        graph_i: index of Curve in graph
        alter: 2-elements list for alter
        """
        dataasxyz = self.attr("datafile_dataasxyz", False)
        xy1rowcol = self.attr("datafile_XY1rowcol", False)
        transpose = self.attr("transpose", False)
        rotate = self.attr("rotate", False)
        datafile = self.attr("datafile", None)

        data = np.zeros((2, 2))
        X, Y = None, None
        if datafile is not None:
            if graph is not None:
                datafile = graph.filenamewithpath(datafile)
            try:
                try:
                    from PIL import Image as PILimage
                except ImportError:
                    try:
                        from pillow import Image as PILimage
                    except ImportError:
                        msg = (
                            "Curve Image: cannot import either PIL or pillow. "
                            "Cannot open image."
                        )
                        print(msg)
                        return data, ignoreNext, X, Y
                data = PILimage.open(datafile)
                if rotate:
                    data = data.rotate(rotate)
                if transpose:
                    data = data.transpose(PILimage.TRANSPOSE)
            except OSError:  # file is not an image -> assume is data
                complement = {"readas": "generic", "_singlecurve": True}
                graphtmp = Graph(datafile, complement)
                if len(graphtmp) > 0:
                    data = graphtmp[0].getData()
                    if dataasxyz:
                        X, Y, data = self.xyz_to_x_y_z(data)
                    data, X, Y = self.process_matrix(
                        data, X, Y, xy1rowcol, transpose, rotate
                    )
                else:
                    msg2 = ""
                    if not os.path.isfile(datafile):
                        msg2 = ". File does not seem to exist."
                    print("Curve image cannot find data ({}). ".format(datafile) + msg2)
        else:  # file not provided
            # greedily aggregate data of following curves, provided x match
            data, numcurves = Curve_Image.aggregate_into_matrix_i(
                graph, graph_i, alter=alter
            )
            ignoreNext += numcurves - 1
            if dataasxyz:
                if data.shape[0] >= 3:
                    X, Y, data = self.xyz_to_x_y_z(data)
                    ignoreNext = 1  # only consider 3 columns (2 Curve), not matrix
                else:
                    msg = (
                        "WARNING Curve Image: data as xyz does not contain the "
                        "required 3 columns."
                    )
                    print(msg)
            data, X, Y = self.process_matrix(data, X, Y, xy1rowcol, transpose, rotate)
        return data, ignoreNext, X, Y

    @staticmethod
    def printHelp(self):
        print("*** *** ***")
        print("Class Curve_Image display images (imshow), contour and contourf plots.")
        print("Detailed help to be written...")
        # keyword type
        # data file
        # first row, col as coordinates; data as 3 column XYZ
        # transpose, rotate
        # extent
        # cmap vmin xmax
        # aspect ratio interpolation
        # levels, extend
        # convert
        return True
