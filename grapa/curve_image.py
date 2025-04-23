# -*- coding: utf-8 -*-
"""A subclass of Curve to deal with images, contour and contourf

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
from copy import deepcopy
import numpy as np
import matplotlib

from grapa import KEYWORDS_CURVE
from grapa.graph import Graph
from grapa.curve import Curve
from grapa.utils.funcgui import FuncGUI


class Curve_Image(Curve):
    """
    The purpose class Curve_Image is to provid GUI support to create images (imshow),
    as well as contour and contourf plots.
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
        item = FuncGUI(self.updateValuesDictkeys, "Set", {"keys": ["type"]})
        item.appendcbb(texttype, default, type_values)
        doc = "Toggle matplotlib plot function: {}.".format(", ".join(type_values))
        item.set_funcdocstring_alt(doc)
        out.append(item)

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
        item = FuncGUI(self.updateValuesDictkeys, "Set", {"keys": ["datafile"]})
        item.append("data file", self.attr("datafile"), options={"width": 40})
        item.set_funcdocstring_alt("Change filename of file containing data.")
        out.append(item)

        # X,Y 1st row column; data as XYZ
        opts_dfxy = {"options": {"field": "Checkbutton"}}
        item = FuncGUI(self.updateValuesDictkeys, "Set")
        item.set_hiddenvars({"keys": ["datafile_XY1rowcol", "datafile_dataasxyz"]})
        item.append("first row, column as coordinates", datafile_xy1rowcol, **opts_dfxy)
        item.append("data as 3-colum XYZ", datafile_dataasxyz, **opts_dfxy)
        doc = "First row, column as coordinates, and data as 3-column XYZ."
        item.set_funcdocstring_alt(doc)
        out.append(item)

        # transpose, rotate
        at = ["transpose", "rotate"]
        item = FuncGUI(self.updateValuesDictkeys, "Set", {"keys": at})
        item.appendcbb(at[0], self.attr(at[0]), ["", "True", "False"])
        item.append(at[1], self.attr(at[1]))
        item.set_funcdocstring_alt("Update attributes 'transpose' and 'rotate'.")
        out.append(item)

        # extent
        if not datafile_xy1rowcol:
            extent = list(self.attr("extent"))
            while len(extent) < 4:
                extent.append("")
            out.append(
                [
                    self.update_extent,
                    "Set",
                    ["extent left", "right", "bottom", "top"],
                    extent,
                ]
            )
        else:
            msg = "extent: not active if first row, column as coordinates"
            out.append([None, msg, [], []])

        # colorscale
        key = "colorbar"
        valscb = [self.attr(key)]
        if "colorbar" in KEYWORDS_CURVE["keys"]:
            i = KEYWORDS_CURVE["keys"].index(key)
            valscb += [str(v) for v in KEYWORDS_CURVE["guiexamples"][i]]
        item = FuncGUI(self.updateValuesDictkeys, "Set", {"keys": [key]})
        item.appendcbb(key, self.attr(key), valscb, options={"width": 50})
        item.set_funcdocstring_alt("Modifies the attribute 'colorbar'.")
        out.append(item)

        # colormap
        attxt = ["cmap (ignored if color image)", "vmin", "vmax"]
        at = [a.split(" ")[0] for a in attxt]
        item = FuncGUI(self.updateValuesDictkeys, "Set", {"keys": at})
        for a in attxt:
            item.append(a, self.attr(a.split(" ")[0]))
        item.set_funcdocstring_alt("Update keywords 'cmap', 'vmin', 'vmax'")
        out.append(item)

        # aspect ratio, interpolation, or levels
        if self.attr("type") == "imshow":
            at = ["aspect", "interpolation"]
            item = FuncGUI(self.updateValuesDictkeys, "Set", {"keys": at})
            item.appendcbb("aspect ratio", self.attr(at[0]), aspect, bind="beforespace")
            item.appendcbb("interpolation", self.attr(at[1]), interpolation)
            doc = "Update attributes 'aspect' (ratio) and 'interpolation'"
            item.set_funcdocstring_alt(doc)
            out.append(item)
        else:  # contour, contourf
            at = ["levels", "extend", "norm"]
            optse = {"field": "Combobox", "values": ["neither", "both", "min", "max"]}
            optsn = {
                "field": "Combobox",
                "values": list(matplotlib.scale.get_scale_names()),
            }
            item = FuncGUI(self.updateValuesDictkeys, "Set", {"keys": at})
            item.append("levels (list of values)", self.attr(at[0]))
            item.append("extend", self.attr(at[1]), options=optse)
            item.append(", norm", self.attr(at[2]), options=optsn)
            doc = "Updates matplotlib keywords 'levels', 'extend', 'horm'."
            item.set_funcdocstring_alt(doc)
            out.append(item)

            levels = self.attr("levels")
            if len(levels) == 0:
                levels = [np.min(self.x()), np.max(self.y())]
            item = FuncGUI(self.update_levels, "Set", {})
            choice = ["arange", "linspace", "logspace", "geomspace"]
            opts = {"field": "Combobox", "values": choice}
            item.append("levels:", choice[1], options=opts)
            item.append("start", np.min(levels))
            item.append("stop", np.max(levels))
            item.append("num or step", len(levels))
            out.append(item)

        # convert data matrix to 3 column xyz and vice-versa
        if graph is not None:
            opts_mtx = {"options": {"field": "Label"}}
            if datafile_dataasxyz:
                item = FuncGUI(self.convert_xyz_matrix, "Convert", {"graph": graph})
                item.append("xyz 3-column format into matrix", "", **opts_mtx)
                out.append(item)
            else:
                item = FuncGUI(self.convert_matrix_xyz, "Convert", {"graph": graph})
                item.append("matrix into xyz 3-column format", "", **opts_mtx)
                out.append(item)

        # help
        out.append(FuncGUI(self.print_help, "Help!"))
        self._funclistgui_memorize(out)
        return out

    def update_extent(self, *args):
        """Update the attribute extent. Must define all values, none left empt ''."""
        flag = False
        for a in args:
            if a != "":
                flag = True
        self.update({"extent": (list(args) if flag else "")})

    def update_levels(self, typ: str, start, stop, step):
        """Modifies the 'levels' used to colorize a plot.

        :param typ: possible values 'arange', 'linspace', 'logspace', 'geomspace'
        :param start: start of e.g. arange
        :param stop: stop of e.g. arange
        :param step: step of e.g. arange
        """
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

    def convert_matrix_xyz(self, *_args, graph=None):
        """Converts a 2D data meshgrid Z[x, y] into 3 columns X, Y, Z"""
        # args: to catch fake argument for GUI purpose
        data, num = self.aggregate_into_matrix(graph=graph)
        data = data.transpose()
        x, y = range(data.shape[0]), range(data.shape[1])
        xyz = self.x_y_z_to_xyz(x, y, data)
        attrs = self.get_attributes()
        out = Graph()
        out.append(Curve([xyz[:, 0], xyz[:, 1]], attrs))
        out.append(Curve([xyz[:, 0], xyz[:, 2]], {}))
        curvetype = out[0].attr("curve")
        if curvetype != "":
            out.castCurve(curvetype, 0)
        out[0].update({"datafile_dataasxyz": True})
        return out  # [c for c in out]

    def convert_xyz_matrix(self, *_args, graph=None):
        """Converts matrix data stored as 3-column xyz into a 2D data meshgrid."""
        print("Curve Image convert_xyz_matrix: About to loose x, y information.")
        data, num = self.aggregate_into_matrix(graph=graph)
        if data.shape[0] < 3:
            print("ERROR Curve Image convert_xyz_matrix: not enough data columns.")
            return False
        vectorx, vectory, matrix = self.xyz_to_x_y_z(data)
        out = Graph()
        for x in range(1, matrix.shape[0]):
            out.append(Curve([matrix[0, :], matrix[x, :]], {}))
        out[0].update(self.get_attributes())
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
            if not graph[j].visible():
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
            except KeyError:
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

    def get_image_data(self, graph, graph_i, alter, ignore_next=0):
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
                        return data, ignore_next, X, Y
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
            ignore_next += numcurves - 1
            if dataasxyz:
                if data.shape[0] >= 3:
                    X, Y, data = self.xyz_to_x_y_z(data)
                    ignore_next = 1  # only consider 3 columns (2 Curve), not matrix
                else:
                    msg = (
                        "WARNING Curve Image: data as xyz does not contain the "
                        "required 3 columns."
                    )
                    print(msg)
            data, X, Y = self.process_matrix(data, X, Y, xy1rowcol, transpose, rotate)
        return data, ignore_next, X, Y
