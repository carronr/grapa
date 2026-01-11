# -*- coding: utf-8 -*-
"""Classes and functions to deal with colors and colorscales.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import logging
from typing import TYPE_CHECKING

import colorsys
import numpy as np

from matplotlib import cm as mplcm
from matplotlib.colors import LinearSegmentedColormap

from grapa.shared.colors_conversion import (
    cielab_to_cielch,
    cielch_to_cielab,
    cielabnorm_to_cielchnorm,
    cielchnorm_to_cielabnorm,
    cielab_to_rgb,
    cielch_to_rgb,
    cielabnorm_to_rgb,
    cielchnorm_to_rgb,
    rgb_to_cielab,
    rgb_to_cielch,
    rgb_to_cielabnorm,
    rgb_to_cielchnorm,
)
from grapa.shared.maths import is_number
from grapa.shared.string_manipulations import strToVar
from grapa.shared.error_management import issue_warning, IncorrectInputError

if TYPE_CHECKING:
    from grapa.graph import Graph

if np.__version__ >= "2.0":
    np.set_printoptions(legacy="1.25")  # in fields: 1.0, not np.float64(1.0)


logger = logging.getLogger(__name__)

# Should make better use of matplotlib class Colormap - get rid of home-made ?
# at the moment handles colormaps both ways: simpler home-made, and names of
# matplotlib colormaps


class Color:
    """internally stores colors in RGB colorspace.
    Supported: RGB, HLS, HSV
    CIELab, CIELCh, and their normalized equivalents CIELabnorm, CIELChnorm"""

    SUPPORTED = [
        "RGB",
        "HLS",
        "HSV",
        "CIELab",
        "CIELCh",
        "CIELabnorm",
        "CIELChnorm",
    ]
    SUPPORTED_ALSOLOWER = SUPPORTED + [item.lower() for item in SUPPORTED]

    def __init__(self, code: tuple, space: str = "rgb"):
        self.space = space.lower()
        self.code = code

    def get(self, space: str = "rgb") -> tuple:
        """Returns the color in the requested colorspace."""
        space = space.lower()
        # avoid unnecessary conversions (ie. hls -> rgb -> hls)
        if space == self.space:
            return self.code

        # if possible, direct conversion without intermediate RGB because loss
        if self.space == "cielab" and space == "cielch":
            return cielab_to_cielch(*self.code)
        if self.space == "cielch" and space == "cielab":
            return cielch_to_cielab(*self.code)
        if self.space == "cielabnorm" and space == "cielchnorm":
            return cielabnorm_to_cielchnorm(*self.code)
        if self.space == "cielchnorm" and space == "cielabnorm":
            return cielchnorm_to_cielabnorm(*self.code)

        # needs a conversion via RGB: first from source to rgb, then to output space
        rgb = self.code
        if self.space != "rgb":
            if len(self.code) != 3:
                msg = (
                    "Color.get: please check input ({}), only RBGA accepts color "
                    "quadruplets (here colorspace is {})"
                )
                issue_warning(logger, msg.format(rgb, self.space))  # raise?
                rgb = rgb[:3]
            if self.space == "hls":
                rgb = colorsys.hls_to_rgb(*self.code)
            elif self.space == "hsv":
                rgb = colorsys.hsv_to_rgb(*self.code)
            elif self.space == "cielab":
                rgb = cielab_to_rgb(*self.code)
            elif self.space == "cielch":
                rgb = cielch_to_rgb(*self.code)
            elif self.space == "cielabnorm":
                rgb = cielabnorm_to_rgb(*self.code)
            elif self.space == "cielchnorm":
                rgb = cielchnorm_to_rgb(*self.code)
            else:
                msg = "Color.get, source colorspace not supported (%s). Supported: %s"
                raise IncorrectInputError(msg % (self.space, self.SUPPORTED))

        # convert rgb to output space
        if space == "rgb":
            return rgb
        if space == "hls":
            return colorsys.rgb_to_hls(*rgb)
        if space == "hsv":
            return colorsys.rgb_to_hsv(*rgb)
        if space == "cielab":
            return rgb_to_cielab(*rgb)
        if space == "cielch":
            return rgb_to_cielch(*rgb)
        if space == "cielabnorm":
            return rgb_to_cielabnorm(*rgb)
        if space == "cielchnorm":
            return rgb_to_cielchnorm(*rgb)

        msg = "Color.get, target space not supported (%s). Supported: %s."
        raise IncorrectInputError(msg % (space, self.SUPPORTED))


class Colorscale:
    """
    This class handles the color scales. Input 'colors' can be:

    - a matplotlib color str e.g. viridis

    - a list of colors. The colorscale will interpolate between these colors. The
      colors are assumed to be uniformly spaced on te value axis.
      If the last element of the color list is a string, it is interpreted as the
      colorspace e.g. rgb, hls, hsv. The color gradient is calculated in this colorspace
    """

    def __init__(self, colors=None, space="rgb", invert=False):
        """
        Colors a list of colors in the format [[1,0,0], [1,1,0.5], [0,1,0]]
        """
        self.colors_default = np.array([[1, 0, 0], [1, 1, 0.5], [0, 1, 0]])
        self.space = space.lower()
        self.invert = False
        if isinstance(colors, str):
            # for example 'jet', 'inferno', etc.
            self.colors = colors
        else:
            if colors is None:
                self.colors = self.colors_default
            else:
                # overrides space input if last element of colors is a string
                # (for example 'hls')
                if isinstance(colors[-1], str):
                    self.space = colors[-1].lower()
                    colors = colors[:-1]
                self.colors = np.array(colors)
        if invert:
            self.inverse_scale()

    def __str__(self):
        return str(self.colors)

    def inverse_scale(self):
        """Inverse the Colorscale"""
        if isinstance(self.colors, str):
            self.invert = not self.invert
        else:
            self.colors = self.colors[::-1]

    def get_colorscale(self):
        """Get the colorscale"""
        if self.space != "rgb":
            return list([list(o) for o in self.colors]) + [self.space]
        return self.colors

    def values_to_color(self, values, space_out="rgb", avoid_white=True):
        """
        Return the color corresponding to a value.
        values is a value np.array([value0, value1, value3, etc.]), with
        valuesN being between 0 and 1
        """
        space_out = space_out.lower()
        avoid_white_param = 0.85
        if is_number(values):
            color = self.values_to_color(
                [values], space_out=space_out, avoid_white=avoid_white
            )
            return list(color[0])

        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # from here we know values is a np.array
        # if predefined matplotlib colorsmap (input was a colormap name str,
        # i.e. 'inferno', etc.)
        if isinstance(self.colors, str):
            return self._values_to_color_str(
                self.colors, values, avoid_white, avoid_white_param, space_out
            )

        # normalize: colormap spaced by 1 every color, values scaled
        # [0, nb(color)]
        _3o4 = 4 if min([len(c) for c in self.colors]) > 3 else 3
        values *= len(self.colors) - 1
        i0 = np.floor(values).astype("int")
        i1 = np.ceil(values).astype("int")
        if avoid_white:
            if (self.colors[0][:3] == [1, 1, 1]).all():
                values = avoid_white_param * values + (1 - avoid_white_param) * (
                    len(self.colors) - 1
                )
            if (self.colors[-1][:3] == [1, 1, 1]).all():
                values = avoid_white_param * values
        # there is certainly a smarter way to program this than a loop
        out = [[]] * len(values)
        for i, val in enumerate(values):
            if i0[i] == i1[i]:  # if values[i] is integer
                out[i] = self.colors[i0[i]][0:_3o4]
            else:  # interpolate
                out[i] = self.colors[i0[i]][0:_3o4] * (
                    float(i1[i]) - val
                ) + self.colors[i1[i]][0:_3o4] * (val - float(i0[i]))
            out[i] = list(out[i])
        if space_out != self.space:
            out = [Color(tuple(o), self.space).get(space_out) for o in out]
        return out

    def _values_to_color_str(
        self, colors: str, values, avoid_white, avoid_white_param, space_out: str
    ):
        col = mplcm.get_cmap(colors, 255)  # , len(values)
        if self.invert:
            values = 1 - values
        # check if colorscale ends with white
        if avoid_white:
            if list(col(0.0))[0:3] == [1, 1, 1]:
                values = avoid_white_param * values + (1 - avoid_white_param)
            if list(col(1.0))[0:3] == [1, 1, 1]:
                values = avoid_white_param * values
        out = [tuple(o) for o in col(values)]
        if space_out != "rgb":
            out = [Color(o, "rgb").get(space_out) for o in out]
        return out

    def cmap(self, nbins=256, avoid_white=False):
        """Returns a matplotlib LinearSegmentedColormap object"""
        val = np.linspace(0, 1, nbins)
        colors = self.values_to_color(val, avoid_white=avoid_white)
        cm = LinearSegmentedColormap.from_list("custom_cmap", colors, N=nbins)
        return cm


def colorscales_from_config(graph: "Graph"):
    """Retrieve colorscales from graph config, or default.

    :param graph: a graph object, with config defined into it
    :return: a list of Colorscale
    """
    out = []
    vals = []
    try:
        kw = "gui_colorscale"
        attrs = dict(graph.config_all()["attributes"])
        for key in list(attrs.keys()):
            if not key.startswith(kw):
                del attrs[key]
        keys = list(attrs.keys())
        keys.sort()
        for key in keys:
            vals.append(strToVar(attrs[key]))
    except Exception:
        issue_warning(logger, "colorscales_from_config: Exception.", exc_info=True)
        vals = []
    # process colorscales and return Colorscales objects
    if len(vals) > 0:
        if not isinstance(vals, list):
            vals = [vals]
        for val in vals:
            if isinstance(val, list):
                if isinstance(val[-1], str):
                    out.append(Colorscale(np.array(val[:-1]), space=val[-1]))
                else:
                    out.append(Colorscale(np.array(val)))
            elif isinstance(val, str):
                try:
                    mplcm.get_cmap(val, 1)
                    out.append(Colorscale(val))
                except ValueError as e:
                    msg = "ValueError in Colorscale with keyword {}, {}."
                    issue_warning(logger, msg.format(val, e))
            else:
                msg = (
                    "colorscales_from_config when loading default colorscales, "
                    "cannot interpret element {}."
                )
                issue_warning(logger, msg.format(val))
        if len(out) > 0:
            return out
    out = [
        Colorscale(np.array([[1, 0, 0], [1, 1, 0.5], [0, 1, 0]])),
        Colorscale(np.array([[1, 0.3, 0], [0.7, 0, 0.7], [0, 0.3, 1]])),
        Colorscale(np.array([[1, 0.43, 0], [0, 0, 1]])),
        Colorscale(np.array([[0.91, 0.25, 1], [1.09, 0.75, 1]]), space="hls"),
        Colorscale(np.array([[0.70, 0.25, 1], [0.50, 0.75, 1]]), space="hls"),
        Colorscale(np.array([[1, 0, 1], [1, 0.5, 1], [0.5, 0.75, 1]]), space="hls"),
    ]
    keys = ["inferno", "gnuplot2", "viridis"]
    for key in keys:
        try:
            mplcm.get_cmap(key, 1)
            out.append(Colorscale(key))
        except ValueError:
            pass
    return out


def colorize_graph(
    graph: "Graph",
    colorscale: "Colorscale",
    same_if_empty_label=False,
    avoid_white=False,
    curvesselection=None,
):
    """Colorize a Graph using a Colorscale.

    :param graph: a Graph object
    :param colorscale: a Colorscale object
    :param same_if_empty_label: if True, the color is repeated if a Curve is hidden
           or if its label is hidden.
    :param avoid_white: avoids using colors too close to white.
    :param curvesselection: provide a list of curve index within the graph, to apply to
           colorscale on."""
    if not isinstance(colorscale, Colorscale):
        colorscale = Colorscale(colorscale)
    # determines which curves on want to colorize
    curves = range(len(graph))
    if curvesselection is not None:
        try:
            curves = [int(c) for c in curvesselection]
        except Exception as e:
            msg = "colorize_graph Exception, invalid curvesselection: %s. %s: %s."
            msa = (curvesselection, type(e), e)
            logger.error(msg, *msa)
            raise IncorrectInputError(msg % msa) from e
    # special cases
    if len(curves) < 1:
        return
    if len(curves) == 1:
        col = colorscale.values_to_color(0.0, avoid_white=avoid_white)
        graph[curves[0]].update({"color": list(col)})
        return
    # general case
    show = np.arange(len(curves))
    if same_if_empty_label:  # if it needs to have several curves with same color
        show = np.array([0.0] * len(curves))
        val = 0.0
        for i, c in enumerate(curves):
            show[i] = val
            labelhide = graph[c].attr("labelhide")
            if graph[c].attr("label") != "" and not labelhide and graph[c].visible():
                val += 1.0
            elif i > 0:
                show[i] = show[i - 1]
    # colorsize
    cols = colorscale.values_to_color(
        show / max(max(show), 1.0), avoid_white=avoid_white
    )
    for i, c in enumerate(curves):
        graph[c].update({"color": list(cols[i])})
