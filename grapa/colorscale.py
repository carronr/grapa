# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

import numpy as np
import colorsys
import warnings
# from matplotlib import colors as matcolors
from tkinter import PhotoImage

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import matplotlib.pyplot as plt

from grapa.mathModule import is_number, strToVar


# Should make better use of matplotlib class Colormap - get rid of home-made ?
# at the moment handles colormaps both ways: simpler home-made, and names of
# matplotlib colormaps

class Color:
    """ internally stores colors in RGB colorspace """

    def __init__(self, code, space='rgb'):
        self.space = space.lower()
        self.code = code

    def get(self, space='rgb'):
        space = space.lower()
        # want to avoid unnecessary conversions (ie. hls -> rgb -> hls)
        if space == self.space:
            return self.code
        # needs a conversion: need to convert to rbg, then to output space
        code = self.code
        if self.space != 'rgb':
            if len(self.code) != 3:
                print('ERROR color: please check input, only RBGA accepts',
                      'color quadruplets (here colorspace is', self.space, ')')
            if self.space == 'hls':
                code = colorsys.hls_to_rgb(*self.code)
            elif self.space == 'hsv':
                code = colorsys.hsv_to_rgb(*self.code)
            else:
                print('Class Color __init__ color space not supported',
                      '(' + self.space + ')')
        # now code is in rgb space
        if space == 'rgb':
            return code
        elif space == 'hls':
            return colorsys.rgb_to_hls(*code)
        elif space == 'hsv':
            return colorsys.rgb_to_hsv(*code)
        else:
            print('Class Color get color space not supported (' + space + ')')
        return None


class PhotoImageColorscale(PhotoImage):
    def __init__(self, width=32, height=32, **args):
        # typical call: (with=32, height=32)
        PhotoImage.__init__(self, width=width, height=height, **args)

    def pixel(self, pos, color):
        """ color in the form [r,g,b]; pos in the form (x,y) """
        [r, g, b] = [int(c*255) for c in color]
        x, y = pos
        self.put("#%02x%02x%02x" % (r, g, b), (x, y))

    def vline(self, x, color):
        """ color in the form [r,g,b]; x scalar """
        [r, g, b] = [int(c*255) for c in color[0:3]]
        hexcode = "#%02x%02x%02x" % (r, g, b)
        for he in range(self.height()):
            self.put(hexcode, (x, he))

    def hline(self, y, color):
        """ color in the form [r,g,b]; x scalar """
        [r, g, b] = [int(c*255) for c in color[0:3]]
        hexcode = "#%02x%02x%02x" % (r, g, b)
        for he in range(self.height()):
            self.put(hexcode, (he, y))

    def fillColorscale(self, colorscale):
        """ fills the image with a Colorscale gradient """
        if self.width() >= self.height():
            for x in range(self.width()):
                # print('_', x, x/(self.width()-1), colorscale.valuesToColor(x/(self.width()-1)))
                self.vline(x, colorscale.valuesToColor(x/(self.width()-1)))
        else:
            for y in range(self.height()):
                self.hline(y, colorscale.valuesToColor(y/(self.height()-1)))


class Colorscale:
    """
    This class handles the color scales.
    """

    def __init__(self, colors=None, space='rgb', invert=False):
        """
        Colors a list of colors in the format [[1,0,0], [1,1,0.5], [0,1,0]]
        """
        self.colorsDefault = np.array([[1, 0, 0], [1, 1, 0.5], [0, 1, 0]])
        self.space = space.lower()
        self.invert = False
        if isinstance(colors, str):
            # for example 'jet', 'inferno', etc.
            self.colors = colors
        else:
            if colors is None:
                self.colors = self.colorsDefault
            else:
                # overrides spaceInput if last element of colors is a string
                # (for example 'hls')
                if isinstance(colors[-1], str):
                    self.space = colors[-1].lower()
                    colors = colors[:-1]
                self.colors = np.array(colors)
        if invert:
            self.inverseScale()

    def __str__(self):
        return str(self.colors)

    def GUIdefaults(config='', **newGraphKwargs):
        from grapa.graph import Graph
        graph = Graph('', **newGraphKwargs)
        out = []
        # retrieve colorscales from config
        try:
            kw = 'gui_colorscale'
            attr = dict(graph._config.curve(0).getAttributes())
            keys = list(attr.keys())
            for key in keys:
                if len(key) < len(kw) or key[:len(kw)] != kw:
                    del attr[key]
            keys = list(attr.keys())
            keys.sort()
            vals = []
            for key in keys:
                val = strToVar(attr[key])
                vals.append(strToVar(attr[key]))
        except Exception as e:
            print('Exception', e)
            vals = []
        # process colorscales and return Colorscales objects
        if vals != []:
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
                        plt.get_cmap(val, 1)
                        out.append(Colorscale(val))
                    except ValueError:
                        # print('Exception in Colorscale with keyword', val, e)
                        pass
                else:
                    print('WARNING when loading default GUI colorscales:',
                          'cannot interpret element', val)
            if len(out) > 0:
                return out
        out = []
        out.append(Colorscale(np.array([[1, 0, 0], [1, 1, 0.5], [0, 1, 0]])))
        out.append(Colorscale(np.array([[1, 0.3, 0], [0.7, 0, 0.7], [0, 0.3, 1]])))
        out.append(Colorscale(np.array([[1, 0.43, 0], [0, 0, 1]])))  # ThW admittance colorscale
        out.append(Colorscale(np.array([[0.91, 0.25, 1], [1.09, 0.75, 1]]), space='hls'))
        out.append(Colorscale(np.array([[0.70, 0.25, 1], [0.50, 0.75, 1]]), space='hls'))
        out.append(Colorscale(np.array([[1, 0, 1], [1, 0.5, 1], [0.5, 0.75, 1]]), space='hls'))
        keys = ['inferno', 'gnuplot2', 'viridis']
        for key in keys:
            try:
                plt.get_cmap(key, 1)
                out.append(Colorscale(key))
            except ValueError:
                pass
        return out

    def inverseScale(self):
        if isinstance(self.colors, str):
            self.invert = not self.invert
        else:
            self.colors = self.colors[::-1]

    def getColorScale(self):
        if self.space != 'rgb':
            return list([list(o) for o in self.colors]) + [self.space]
        return self.colors

    def valuesToColor(self, values, spaceOut='rgb', avoidWhite=True):
        """
        Return the color corresponding to a value.
        values is a value np.array([value0, value1, value3, etc.]), with
        valuesN being between 0 and 1
        """
        spaceOut = spaceOut.lower()
        avoidWhiteParam = 0.85
        if is_number(values):
            return list(self.valuesToColor([values], spaceOut=spaceOut,
                                           avoidWhite=avoidWhite)[0])
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        # from here we know values is an np.array
        # if predefined matplotlib colorsmap (input was a colormap name str,
        # ie. 'inferno', etc)
        if isinstance(self.colors, str):
            col = plt.get_cmap(self.colors, 255)  # , len(values)
            if self.invert:
                values = 1 - values
            # check if colorscale ends with white
            if avoidWhite:
                if (list(col(0.0))[0:3] == [1, 1, 1]):
                    values = avoidWhiteParam * values + (1-avoidWhiteParam)
                if (list(col(1.0))[0:3] == [1, 1, 1]):
                    values = avoidWhiteParam * values
            out = [list(o) for o in col(values)]
            if spaceOut != 'rgb':
                out = [Color(o, 'rgb').get(spaceOut) for o in out]
        else:
            # normalize: colormap spaced by 1 every color, values scaled
            # [0, nb(color)]
            _3o4 = 4 if min([len(c) for c in self.colors]) > 3 else 3
            values *= (len(self.colors) - 1)
            i0 = np.floor(values).astype('int')
            i1 = np.ceil(values).astype('int')
            if avoidWhite:
                if (self.colors[0][:3] == [1, 1, 1]).all():
                    values = avoidWhiteParam * values + (1-avoidWhiteParam)*(len(self.colors)-1)
                if (self.colors[-1][:3] == [1, 1, 1]).all():
                    values = avoidWhiteParam * values
            out = [0] * len(values)
            # there iscertainly a smarter way to program this than a loop
            for i in range(len(values)):
                if i0[i] == i1[i]:  # if values[i] is integer
                    out[i] = self.colors[i0[i]][0:_3o4]
                else:  # interpolate
                    out[i] = self.colors[i0[i]][0:_3o4] * (float(i1[i])-values[i]) + self.colors[i1[i]][0:_3o4] * (values[i]-float(i0[i]))
                out[i] = list(out[i])
            if spaceOut != self.space:
                out = [Color(o, self.space).get(spaceOut) for o in out]
        return out

    def cmap(self, nbins=256, avoidWhite=False):
        from matplotlib.colors import LinearSegmentedColormap
        val = np.linspace(0, 1, nbins)
        colors = self.valuesToColor(val, avoidWhite=avoidWhite)
        cm = LinearSegmentedColormap.from_list('custom_cmap', colors, N=nbins)
        return cm
