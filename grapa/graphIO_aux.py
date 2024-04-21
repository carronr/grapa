# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:21:57 2018

@author: Romain
"""
from copy import copy
import numpy as np
import warnings

# with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
# import matplotlib.pyplot as plt

# from grapa.curve_inset import Curve_Inset
# from grapa.curve_subplot import Curve_Subplot


class ParserAxlines:
    # Possible inputs:
    # - 3
    # - [3, {'color':'r', 'xmin': 0.5]
    # - [1, 2, 4, {'color':'b'}]
    # - [[1, {'color':'c'}], [2, {'color':'m'}]]
    def __init__(self, data):
        self._list = []
        if not isinstance(data, list):  # handles "3"
            self.__init__([data])
            return
        # from now on data assumed to be a list
        kwdef = {"color": "k"}
        if isinstance(data[-1], dict):  # handles [..., {'color':'m'}]
            kwdef.update(data[-1])
            data = data[:-1]
        for el in data:
            kw = copy(kwdef)
            if not isinstance(el, list):  # handles [1,2,3...]
                self._list.append({"args": [el], "kwargs": kw})
            else:
                el_ = el
                # handles [[1,2, {'color':'y'], [3, {'color': 'm'}]]
                if isinstance(el[-1], dict):
                    kw.update(el[-1])
                    el_ = el[:-1]
                for e in el_:
                    self._list.append({"args": [e], "kwargs": kw})

    def __str__(self):
        return "self\t" + "\n\t".join([str(a) for a in self._list])

    def plot(self, method):
        for el in self._list:
            method(*el["args"], **el["kwargs"])


class ParserAxhline(ParserAxlines):
    def plot(self, method, curvedummy, alter, type_plot):
        for el in self._list:
            pos = curvedummy.y(
                alter=alter[1],
                xyValue=[1, el["args"][0]],
                errorIfxyMix=True,
                neutral=True,
            )
            if pos != 0 or type_plot not in ["semilogy", "loglog"]:
                try:
                    method(pos, **el["kwargs"])
                except Exception as e:
                    print(
                        "Keyword axhline, exception",
                        type(e),
                        "in ParserAxhline.plot",
                        e,
                    )
        return True


class ParserAxvline(ParserAxlines):
    def plot(self, method, curvedummy, alter, type_plot):
        for el in self._list:
            pos = curvedummy.x(
                alter=alter[0],
                xyValue=[el["args"][0], 1],
                errorIfxyMix=True,
                neutral=True,
            )
            if pos != 0 or type_plot not in ["semilogx", "loglog"]:
                try:
                    method(pos, **el["kwargs"])
                except Exception as e:
                    print(
                        "Keyword axvline, exception",
                        type(e),
                        "in ParserAxhline.plot",
                        e,
                    )
        return True


class GroupedPlotters:
    """class to handle boxplots, violin plots and related"""

    def __init__(self):
        self.plotters = {
            "boxplot": PlotterBoxplot(),
            "violinplot": PlotterViolin(),
        }
        Plotter.position = 0

    def add_curve(self, type_graph, curve, y, fmt, ax):
        for key in self.plotters:
            if type_graph == key:
                return self.plotters[key].add_curve(curve, y, fmt, ax)

    def plot(self, ax, ax_):
        for key, plotter in self.plotters.items():
            if len(plotter) > 0:
                plotter.plot(ax, ax_)


class Plotter:
    position = 0
    IMPORTERROR = "ImportError"

    SEABORN_STRIPPLOT = "seaborn.stripplot"
    SEABORN_SWARMPLOT = "seaborn.swarmplot"
    SCATTER = "scatter"
    SCATTER_JITTER_DEFAULT = 0.15

    def __init__(self):
        self.y = []
        self.callkw = {"positions": [], "labels": []}
        self.colors = []
        # aim: warnings to be shown only once per graph plotting; Instances of plotter
        self._showwarnings = {"swarmplot_jitter": True}

    def __len__(self):
        return len(self.y)

    def plot_addon_scatter(self, curve, y, position, ax):
        addon = curve.attr("boxplot_addon", None)  # seaborn_stripplot_kw
        if not isinstance(addon, list) or len(addon) != 2:
            return None
        fun, kw = addon[0], dict(addon[1])
        if fun == Plotter.SCATTER:
            jitter = Plotter.SCATTER_JITTER_DEFAULT
            if "jitter" in kw:
                jitter = kw["jitter"]
                del kw["jitter"]
            if "size" in kw and "s" not in kw:
                kw["s"] = kw["size"]
                del kw["size"]
            x = position + jitter * 2 * (np.random.rand(len(y)) - 0.5)
            return ax.scatter(x, y, **kw)
        elif fun == Plotter.SEABORN_STRIPPLOT:
            try:
                import seaborn as sns  # grapa must run even without seaborn

                kw.update({"ax": ax, "native_scale": True})
                return sns.stripplot({position: y[~np.isnan(y)]}, **kw)
            except ImportError:
                msg = (
                    "GroupedPlotters ImportError with library seaborn, cannot plot "
                    "stripplot."
                )
                print(msg)
                return self.IMPORTERROR
        elif fun == Plotter.SEABORN_SWARMPLOT:
            try:
                import seaborn as sns  # grapa must run even without seaborn

                kw.update({"ax": ax, "native_scale": True})
                if "jitter" in kw:
                    del kw["jitter"]
                    if self._showwarnings["swarmplot_jitter"]:
                        print(
                            "Plotter plot_addon_scatter swarmplot: removed keyword "
                            "jitter"
                        )
                        self._showwarnings["swarmplot_jitter"] = False
                with warnings.catch_warnings(record=True) as warns:
                    handle = sns.swarmplot({position: y[~np.isnan(y)]}, **kw)
                    if len(warns) > 0:
                        for warn in warns:
                            print(warn.category, warn.filename, ":", warn.message)
                    return handle
            except ImportError:
                msg = (
                    "GroupedPlotters ImportError with library seaborn, cannot plot "
                    "swarmplot."
                )
                print(msg)
                return self.IMPORTERROR
        return None

    @classmethod
    def funcListGUI(cls, curve):
        current = curve.attr("boxplot_addon")
        color = curve.attr("color", "grey")
        val0 = ["", cls.SCATTER, cls.SEABORN_STRIPPLOT, cls.SEABORN_SWARMPLOT]
        val1 = [
            "",
            {},
            {"color": curve.attr("color", "k")},
            {"color": color, "jitter": Plotter.SCATTER_JITTER_DEFAULT},
            {"color": color, "size": 5, "jitter": Plotter.SCATTER_JITTER_DEFAULT},
        ]
        if not isinstance(current, list):
            def0, def1 = "", ""
        else:
            while len(current) < 2:
                current.append("")
            def0 = current[0]
            def1 = current[1]
            if def1 not in val1 and isinstance(def1, dict):
                val1.insert(0, def1)
        out = [
            [
                curve.plotterBoxplotAddonUpdateCurve,
                "Save",
                ["also show data", "kw"],
                [def0, def1],
                {},
                [
                    {"field": "Combobox", "values": val0},
                    {"field": "Combobox", "values": val1, "width": 20},
                ],
            ]
        ]
        return out


class PlotterBoxplot(Plotter):
    def add_curve(self, curve, y, fmt, ax):
        handle = None
        # y: already computed from calling nearby environment
        if len(y) > 0 and not np.isnan(y).all():
            position = curve.attr("boxplot_position", Plotter.position)
            self.y.append(y[~np.isnan(y)])
            self.callkw["positions"].append(position)
            self.callkw["labels"].append(fmt["label"] if "label" in fmt else "")
            self.colors.append(fmt["color"] if "color" in fmt else "")
            # aggressive gather of possible parameters
            for key in fmt:
                if key not in self.callkw and key not in ["label", "color"]:
                    self.callkw[key] = fmt[key]
            # addon - seaborn swarmplot or stripplot
            handle = self.plot_addon_scatter(curve, y, position, ax)
            if handle == Plotter.IMPORTERROR:
                # do not mask fliers if seaborn import failed
                self.callkw["showfliers"] = True
                print("boxplot -> show fliers True")
                handle = None
            Plotter.position += 1
        return handle

    def plot(self, ax, ax_):
        handle = ax.boxplot(self.y, **self.callkw)
        # handle coloring
        nb_el_boxplot = {"whiskers": 2, "caps": 2}
        for key in handle:
            for i in range(len(handle[key])):
                j = (
                    i
                    if key not in nb_el_boxplot
                    else int(np.floor(i / nb_el_boxplot[key]))
                )
                if j < len(self.colors) and self.colors[j] != "":
                    handle[key][i].set_color(self.colors[j])
                    if hasattr(handle[key][i], "set_markeredgecolor"):
                        handle[key][i].set_markeredgecolor(self.colors[j])
        return handle


class PlotterViolin(Plotter):
    def __init__(self):
        super().__init__()
        del self.callkw["labels"]
        self.labels = []

    def add_curve(self, curve, y, fmt, ax):
        handle = None
        if len(y) > 0 and not np.isnan(y).all():
            position = curve.attr("boxplot_position", Plotter.position)
            self.y.append(y[~np.isnan(y)])
            self.callkw["positions"].append(position)
            self.labels.append(fmt["label"] if "label" in fmt else "")
            self.colors.append(fmt["color"] if "color" in fmt else "")
            # could use a more aggressive strategy to gather keywords
            for key in ["showmeans", "showmediansshowmedians", "showextrema"]:
                value = curve.attr(key)
                if value != "":
                    self.callkw[key] = value
            # addon - seaborn swarmplot or stripplot
            handle = self.plot_addon_scatter(curve, y, position, ax)
            if handle == Plotter.IMPORTERROR:
                handle = None
            Plotter.position += 1
        return handle

    def plot(self, ax, ax_):
        # print("violinplot plot self.callkw", self.callkw)
        handle = ax.violinplot(self.y, **self.callkw)
        # set color
        for key in handle:
            if isinstance(handle[key], list):
                for i in range(len(handle[key])):
                    if self.colors[i] != "":
                        handle[key][i].set_color(self.colors[i])
            else:
                handle[key].set_color([0, 0, 0])
        # violinplots labels not set automatically, unlike boxplot
        for i in range(len(self.labels)):
            xticks = list(ax_.get_xticks())
            xtklbl = list(ax_.get_xticklabels())
            xticks.append(self.callkw["positions"][i])
            xtklbl.append(self.labels[i])
            ax_.set_xticks(xticks)
            ax_.set_xticklabels(xtklbl)
        return handle


"""
# NOT READY YET
class PlotterSeabornStripplot(Plotter):
    def add_curve(self, curve, y, fmt, ax):
        if len(y) > 0 and not np.isnan(y).all():
            position = curve.attr("boxplot_position", None)
            position = Plotter.position if position is None else position

            self.y[position] = y[~np.isnan(y)]
            self.callkw["labels"].append(fmt["label"] if "label" in fmt else "")

            # self.colors.append(fmt['color'] if 'color' in fmt else '')  # colors: to handle through palette keyword?

            # TODO: handle keywords
            # self.plot_seaborn_stripswamplots(violinplot['positions'][-1], y, ax)

            Plotter.position += 1

    def plot(self, ax, ax_):
        self.callkw["ax"] = ax
        try:
            import seaborn as sns

            return sns.stripplot(self.y, **self.callkw)
        except ImportError:
            print("PlotterSeabornStripplot import error, cannot import seaborn")
        return None

        # kwstrip = self.attr("seaborn_stripplot_kw", None)
        # if kwstrip is not None:
        # try:
        # import seaborn as sns
        # kwstrip = dict(kwstrip)
        # kwstrip.update({"ax": ax, "native_scale": True})
        # sns.stripplot({position: y[~np.isnan(y)]}, **kwstrip)
        # except ImportError:
"""
