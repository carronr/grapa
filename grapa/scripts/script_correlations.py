import os
import sys
import copy
import glob
import numpy as np
from scipy.stats import pearsonr
import fnmatch

path = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if path not in sys.path:
    sys.path.append(path)

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.datatypes.graphScaps import GraphScaps


AS_DATATABLE = "AS_DATATABLE"
AS_SCAPS = "AS_SCAPS"
AUTO = "AUTO"

HIGHER = ">"
LOWER = "<"


def writefile_datatable(filename, pkeys, pvals):
    """Write the content of the data matix into a text file"""
    with open(filename, "w") as file:
        file.write("\t".join(pkeys) + "\n")
        data = np.transpose(np.array(pvals))
        for line in data:
            file.write("\t".join([str(li) for li in line]) + "\n")


def plot_parameters_1D(pkeys, pvals, newGraphKwargs={}):
    """array of normal plots"""
    xlim = values_to_lim(pvals[0])
    graphaux = Graph(**newGraphKwargs)
    graphaux.append(Curve([[0], [0]], {}))
    graphaux.castCurve("subplot", 0, silentSuccess=True)
    graphaux[0].update({"subplotupdate": {"xlabel": pkeys[0], "xlim": xlim}})
    graphshow = Graph(**newGraphKwargs)
    for i in range(1, pvals.shape[0]):
        graphshow.append(copy.deepcopy(graphaux[0]))
        spu = graphshow[-1].attr("subplotupdate")
        spu.update({"title": pkeys[i]})
        graphshow[-1].update({"subplotupdate": spu})
        graphshow.append(Curve([pvals[0], pvals[i, :]], {"label": pkeys[i]}))
    graphshow.update(
        {"subplots_adjust": [0.1, 0.1, 0.9, 0.9, 0.5, 0.5], "figsize": [8, 8]}
    )
    return graphshow


def plot_parameters_2D(pkeys, pvals, seriesx=None, seriesy=None, newGraphKwargs={}):
    """array of scatter plots"""
    if seriesx is None:
        seriesx = [0, 1]
    if seriesy is None:
        seriesy = range(2, len(pkeys))

    s_a = [0.1, 0.1, 0.9, 0.9, 0.5, 0.5]
    figsize = [8, 8]
    spanx = figsize[0] * (s_a[2] - s_a[0]) / (2 * 1 + 1 * s_a[4]) * 72  # in points
    npntsx = np.sqrt(pvals.shape[1]) * 1.5  # *2 safety marging, symbols not too large
    markersize = (spanx / npntsx) ** 2
    # print('markersize', markersize)

    graphaux = Graph(**newGraphKwargs)
    graphaux.append(Curve([[0], [0]], {}))
    attr0 = {
        "type": "scatter",
        "marker": "o",
        "markersize": markersize,
        "colorbar": {"adjust": [1.00, 0, 0.05, 1, "ax"]},
    }
    x0, x1 = pvals[seriesx[0], :], pvals[seriesx[1], :]
    graphaux.append(Curve([x0, x1], attr0))
    graphaux.castCurve("subplot", 0, silentSuccess=True)
    su = {
        "xlabel": pkeys[seriesx[0]],
        "ylabel": pkeys[seriesx[1]],
        "xlim": values_to_lim(x0),
        "ylim": values_to_lim(x1),
    }
    graphaux[0].update({"subplotupdate": su})

    graphshow = Graph(**newGraphKwargs)
    flagonlynan = True
    for i in seriesy:
        values, key = pvals[i, :], pkeys[i]
        if pkeys[i] in ["FF", "eta"]:
            values = np.array(values) * 100
            key = key + " (%)"
        graphshow.append(copy.deepcopy(graphaux[0]))
        # text annotations
        text, textxy, textargs = [], [], []
        colorthres = (np.min(values) + np.max(values)) / 2
        for j in range(len(x0)):
            color = "k" if values[j] > colorthres else "w"  # [0.5,0.5,0.5]
            text.append("{:#.3g}".format(values[j]))
            textxy.append("")
            textargs.append(
                {
                    "xytext": (x0[j], x1[j]),
                    "textcoords": "data",
                    "verticalalignment": "center",
                    "horizontalalignment": "center",
                    "fontsize": 8,
                    "color": color,
                }
            )  # , 'zorder':0
        spu = graphshow[-1].attr("subplotupdate")
        spu.update({"title": key, "text": text, "textxy": textxy, "textargs": textargs})
        graphshow[-1].update({"subplotupdate": spu})
        graphshow.append(graphaux[1])
        # add data
        graphshow.append(Curve([x0, values], {"type": "scatter_c", "label": key}))
        if not np.isnan(values).all():
            flagonlynan = False
    graphshow.update({"subplots_adjust": s_a, "figsize": figsize})
    if flagonlynan:
        return None
    return graphshow


def plot_correlations(
    pkeys, pvals, seriesx=None, seriesy=None, groupbyparam=True, newGraphKwargs={}
):
    """
    seriesX, seriesy: tuple of column to consider
    groupbyparam: if True, organize data as series where possible
    """
    if seriesx is None:
        seriesx = range(pvals.shape[0])
    if seriesy is None:
        seriesy = range(pvals.shape[1])

    # preparatory works
    islogx = [guess_is_logarithm_quantity(pvals[i, :]) for i in seriesx]
    islogy = [guess_is_logarithm_quantity(pvals[j, :]) for j in seriesy]
    xlim = []
    for k in range(len(seriesx)):
        d = pvals[seriesx[k], :]
        lim = list(np.exp(values_to_lim(np.log(d)))) if islogx[k] else values_to_lim(d)
        xlim.append(lim)
    ylim = []
    for k in range(len(seriesy)):
        d = pvals[seriesy[k], :]
        lim = list(np.exp(values_to_lim(np.log(d)))) if islogy[k] else values_to_lim(d)
        ylim.append(lim)
    # axis label & ticks removval. arbitraryfunc.: matplotlib bug with minor tick labels
    sburemoverx = {
        "xtickslabels": [None, []],
        "xlabel": "",
        "arbitraryfunctions": [["xaxis.set_minor_formatter", ["NullFormatter()"], {}]],
    }
    sburemovery = {
        "ytickslabels": [None, []],
        "ylabel": "",
        "arbitraryfunctions": [["yaxis.set_minor_formatter", ["NullFormatter()"], {}]],
    }

    lenx, leny = len(seriesx), len(seriesy)
    graph = Graph(**newGraphKwargs)
    plts = [3 * (lenx + (lenx - 1) * 0.0), 3 * (leny + (leny - 1) * 0.0)]
    fs = [plts[0] + 1 + 0.5, plts[1] + 1 + 0.5]
    sa = [1 / fs[0], 1 / fs[0], 1 - 0.5 / fs[0], 1 - 0.5 / fs[1], 0, 0]
    graph.update({"subplotsncols": lenx, "figsize": fs, "subplots_adjust": sa})
    pearson = np.zeros((lenx, leny, 2))
    pearson.fill(np.nan)
    for j_ in range(len(seriesy)):
        j = seriesy[j_]
        pvalsj = pvals[j, :]
        if np.isnan(pvalsj).all():
            continue  # no point to proceed if only nan in data
        for i_ in range(len(seriesx)):
            i = seriesx[i_]
            pvalsi = np.array(pvals[i, :])
            # Curve subplot
            attri = {
                "label": "subplot {}; {}".format(pkeys[i], pkeys[j]),
                "labelhide": 1,
            }
            graph.append(Curve([[0], [0]], attri))
            graph.castCurve("subplot", -1, silentSuccess=True)

            # curve with data
            sbu = {"xlabel": pkeys[i], "ylabel": pkeys[j]}
            sbu.update({"xlim": xlim[i_], "ylim": ylim[j_]})
            # remove axis labels, except for end graph.
            # NB: grapa not yet compatible with sharex and sharey axes
            if i != seriesx[0]:
                sbu.update(sburemovery)
            if j != seriesy[-1]:
                sbu.update(sburemoverx)
            if islogx[i_] and islogy[j_]:
                sbu.update({"typeplot": "loglog"})
            elif islogx[i_]:
                sbu.update({"typeplot": "semilogx"})
            elif islogy[j_]:
                sbu.update({"typeplot": "semilogy"})
            graph[-1].update({"subplotupdate": sbu})

            # if group by subseries with same values of other parameters
            flagappend = False
            if groupbyparam and i != j:
                signatures = []
                for m in range(len(pvalsi)):
                    signatures.append(
                        ";".join([str(pvals[k, m]) for k in seriesx if k != i])
                    )
                signatures = np.array(signatures)
                signunique = np.unique(signatures)
                if len(signunique) < len(pvalsi):
                    for sign in signunique:
                        test = signatures == sign
                        attrs = {"linespec": "o-", "label": sign, "labelhide": 1}
                        graph.append(Curve([pvalsi[test], pvalsj[test]], attrs))
                        flagappend = True
            if not flagappend:  # not group by parameter signature: data cloud
                graph.append(Curve([pvalsi, pvalsj], {"linespec": "o"}))
            mask = ~np.logical_or(np.isnan(pvalsi), np.isnan(pvalsj))
            try:
                pearson[i_, j_, :] = pearsonr(pvalsi[mask], pvalsj[mask])
            except ValueError:
                pass  # maybe not >= 2 numbers to compute correlation

    if np.isnan(pearson).all():
        print("Only Nan in Pearson matrix. Stop here.")
        return None, None
    return graph, pearson


def plot_pearson(pearson, keysx, keysy, newGraphKwargs={}):
    """
    plot Pearson correlation graph
    pearson: pearson correlation matrix
    """
    lenx, leny = pearson.shape[0], pearson.shape[1]
    # plot Pearson correlation coefficient
    graph = Graph(**newGraphKwargs)
    for c in range(1, pearson.shape[1]):
        graph.append(Curve([pearson[:, 0, 0], pearson[:, c, 0]], {}))
    graph.castCurve("image", 0, silentSuccess=True)
    one = [0.5, 0.5]
    mrgn = [2, 2.5, 1.5, 0.5]
    plts = [one[0] * lenx, one[1] * leny]
    fs = [mrgn[0] + mrgn[2] + plts[0], mrgn[1] + mrgn[3] + plts[1]]
    sa = [mrgn[0] / fs[0], mrgn[1] / fs[1], 1 - mrgn[2] / fs[0], 1 - mrgn[3] / fs[1]]
    graph.update(
        {
            "xtickslabels": [
                list(range(lenx)),
                keysx,
                {"rotation": 45, "ha": "right", "fontsize": 8},
            ],
            "ytickslabels": [list(range(leny)), keysy],
            "figsize": fs,
            "subplots_adjust": sa,
            "xlabel": " ",
            "ylabel": " ",
        }
    )
    graph[0].update(
        {
            "cmap": "RdBu",
            "vmin": -1,
            "vmax": 1,
            "aspect": "auto",  # may avoid discrepancy allocated and actual axes positio
            "colorbar": {
                "label": "Pearson correlation coefficient",
                "adjust": [1 + one[0] / 4 / plts[0], 0, one[0] / 2 / plts[0], 1, "ax"],
            },
        }
    )
    return graph


def guess_is_logarithm_quantity(series):
    sunique = np.unique(series)
    if 0 in sunique or (sunique < 0).any():
        return False
    if len(sunique) > 2:
        ratios = sunique[1:] / sunique[:-1]
        ratiosda = np.abs(ratios - ratios[0])
        if np.max(ratiosda) < 1e-15:
            return True
    return False


def values_to_lim(values):
    mi, ma = np.nanmin(values), np.nanmax(values)
    s = ma - mi
    lim = [mi - s / 10, ma + s / 10]
    if lim[0] == lim[1]:
        return ""
    return lim


def filter_pvals(pkeys, pvals, filters: list = None):
    # cleanup of data input
    if filters is None:
        filters = []

    # filter data
    flagged = []
    pvalsshape = list(pvals.shape)
    for j in range(pvals.shape[1]):
        flag = True
        for fil in filters:
            i = None
            if isinstance(fil[0], int):
                i = fil[0]
            else:
                try:
                    i = pkeys.index(fil[0])
                except ValueError:
                    for i in range(len(pkeys)):
                        if fnmatch.fnmatch(pkeys[i], fil[0]):
                            break  # True
            if fil[1] == HIGHER and not pvals[i, j] > fil[2]:
                flag = False
            elif fil[1] == LOWER and not pvals[i, j] < fil[2]:
                flag = False
        if not flag:
            flagged.append(j)
    # print(pvals)
    # print(pvals.shape)
    # print(flagged)
    # print(pkeys)
    pvals = np.delete(pvals, flagged, axis=1)
    # print text message
    if len(filters) > 0:
        criteria = " Filters: " + ", ".join(["{} {} {}".format(*f) for f in filters])
        msg = "Filtering data: from {} to {} rows in input data.{}"
        print(msg.format(pvalsshape[1], pvals.shape[1], criteria))
    else:
        print("Filtering data: no filtering.")
    return pvals


def process_datatable(
    filenamebase, pkeys, pvals, seriesx=None, seriesy=None, **ngkwargs
):
    graphshow = None
    if len(seriesx) == 1:
        graphshow = plot_parameters_1D(pkeys, pvals, **ngkwargs)
        graphshow.plot(filenamebase + "_summary")
    if len(seriesx) == 2:
        graphshow = plot_parameters_2D(pkeys, pvals, seriesx, seriesy, **ngkwargs)
        if graphshow is not None:
            graphshow.plot(filenamebase + "_summary")

    # correlation graph
    graphtable, pearson = plot_correlations(
        pkeys, pvals, seriesx=seriesx, seriesy=seriesy, **ngkwargs
    )
    # Pearson summary
    graphpearson = None
    if graphtable is not None:
        # pearson graph
        graphpearson = plot_pearson(
            pearson,
            [pkeys[p] for p in seriesx],
            [pkeys[p] for p in seriesy],
            **ngkwargs
        )
        graphtable.plot(filenamebase + "_correlation_data")
        graphpearson.plot(filenamebase + "_correlation_pearson")
    return graphtable, graphpearson, pearson


class Helper:
    @classmethod
    def seriesxy(cls, pkeys):
        seriesx = range(len(pkeys))
        seriesy = range(len(pkeys))
        return seriesx, seriesy

    @classmethod
    def as_labels_datatable(cls, graph, keys):
        pkeys = keys
        pvals = []
        for key in keys:
            pvals.append([])
            for curve in graph:
                pvals[-1].append(curve.attr(key, np.nan))
        return pkeys, np.array(pvals), len(pkeys)


class HelperScaps(Helper):
    @classmethod
    def seriesxy(cls, pkeys):
        nresults = 6
        nparams = len(pkeys) - nresults
        nresults = 4  # hack, ignore last 2 results
        seriesx = range(nparams)
        seriesy = range(nparams, nparams + nresults)
        return seriesx, seriesy

    @classmethod
    def as_labels_datatable(cls, graph, *args):
        pkeys = []
        pvals = []
        keys_of_interest = []
        nparam = 0
        # retrieve parameters. number not known a priori
        while True:
            test = "Batch parameters " + str(nparam) + " key"
            if graph[0].attr(test, "") != "":
                key = graph[0].attr(test)
                pkeys.append(key)
                keys_of_interest.append(key)
                nparam += 1
                continue
            break
        # known keys for PV values
        jv_params = [
            "solar cell parameters deduced from calculated IV-curve Voc (Volt)",
            "solar cell parameters deduced from calculated IV-curve Jsc (mA/cm2)",
            "solar cell parameters deduced from calculated IV-curve FF",
            "solar cell parameters deduced from calculated IV-curve eta",
            "solar cell parameters deduced from calculated IV-curve V_MPP (Volt)",
            "solar cell parameters deduced from calculated IV-curve J_MPP (mA/cm2)",
        ]
        keys_of_interest += jv_params
        pkeys += [
            p.replace("solar cell parameters deduced from calculated IV-curve ", "")
            for p in jv_params
        ]
        for key in keys_of_interest:
            pvals.append([])
            for curve in graph:
                pvals[-1].append(curve.attr(key, np.nan))
        # print(pkeys)
        # print(pvals)
        return pkeys, np.array(pvals), nparam


class HelperDatatable(Helper):
    @classmethod
    def as_labels_datatable(cls, graph, *args):
        """
        Transforms a Graph into a list of labels pkeys, data table pvals
        """
        pkeys = graph.attr("collabelsdetail", None)
        if isinstance(pkeys, list):
            if (
                len(pkeys) == 1
                and isinstance(pkeys[0], list)
                and len(pkeys[0]) == len(graph) + 1
            ):
                pkeys = pkeys[0]
            if len(pkeys) != len(graph) + 1:
                print(
                    "HelperDatatable collabelsdetail not good pick.",
                    len(pkeys),
                    len(graph),
                )
                print(graph.attr("collabelsdetail", None))
                pkeys = None
        else:  # let's try something else
            pkeys = None
        if pkeys is None:
            pkeys = graph.attr("collabels")
            if len(pkeys) == len(graph):  # missing first columns
                pkeys.insert(0, graph.attr("xlabel"))
            if len(pkeys) < len(graph) + 1:
                print(
                    "HelperDatatable.as_labels_datatable: Alternative guess for pkeys"
                )
                pkeys = [graph.attr("xlabel")]
                for curve in graph:
                    pkeys.append(curve.attr("label"))
        # cleanup grapa "salting" of column labels
        attrs = graph[0].getAttributes()
        for key in attrs:
            if pkeys[0].endswith(key):
                flag = True
                for c in range(len(graph)):
                    if not pkeys[c + 1].endswith(graph[c].attr(key)):
                        flag = False
                if flag:
                    le = len(pkeys[0]) - len(key)
                    pkeys = [p[le:] for p in pkeys]
                    print("HelperDatatable.as_labels_datatable: pkeys cleaned up!")
                    break
        # extract data
        pvals = [graph[0].x()]
        for curve in graph:
            pvals.append(curve.y())
        if len(pkeys) != len(pvals):
            print(
                "ERROR HelperDatatable.as_labels_datatable size issue",
                len(pkeys),
                len(pvals),
            )
            print(pkeys)
        return pkeys, np.array(pvals), len(pkeys)


def process_file(
    filename,
    datakeys=AUTO,
    filters: list = None,
    seriesx=None,
    seriesy=None,
    newGraphKwargs={},
):
    """
    :param filename: file to process
    :param datakeys: how to interpret the file content.
        AS_DATATABLE: open the file as a datatable
        AS_SCAPS: assumes this is the output of Scaps simulations. also preselect series
            of interest
        AUTO: first open the graph, then auto detect
        [key1, key2, ...]: to retrieve from graph Graph, each Curve is one "experiment"
    :param filters: list of conditions to exclude specific "experiments" (e.g. rows in
        a table) should be excluded. e.g. [["Jsc_mApcm2", HIGHER, 10]]
    :param seriesx: list/range of data to consider for the correlation plots
        By default, whow all columns
    :param seriesy: list/range of data to consider for the correlation plots
        By default, whow all columns
    :param newGraphKwargs: specific to Grapa, to e.g. have consistent config file
    :return:
    """
    print("Processing file", filename)
    ngkwargs = {"newGraphKwargs": newGraphKwargs}
    helper = Helper
    helper_dict = {AS_SCAPS: HelperScaps, AS_DATATABLE: HelperDatatable}
    if isinstance(datakeys, str) and datakeys in helper_dict:
        helper = helper_dict[datakeys]
        print("Interpret input data {}.".format(datakeys.lower().replace("_", " ")))

    # open input data
    graph = Graph(filename, **newGraphKwargs)

    if datakeys == AUTO:
        if graph.attr("meastype") == GraphScaps.FILEIO_GRAPHTYPE:
            print("Datatype AUTO, detected Scaps.")
            datakeys = AS_SCAPS
            helper = HelperScaps
        if len(str(graph.attr("collabels", ""))) > 0 and graph.attr("meastype") in [
            "Undetermined data type",
            "Database",
        ]:
            print("Datatype AUTO, detected data table.")
            datakeys = AS_DATATABLE
            helper = HelperDatatable

    # some checks
    if len(graph) == 0:
        print("Could not find data in the file. Script end.")
        return False
    fnamebase = os.path.splitext(filename)[0]
    dataext = graph[0].attr("curve").replace("Curve ", "")

    print("Running script assuming data", datakeys)

    # if datakeys != AS_DATATABLE:
    graph.plot(fnamebase + "_parseddata" + dataext)

    # specific to file format to retrieve the parameters of interest
    # SCAPS: data as data table: parameter0, parameter1, ..., Voc, Jsc, FF, Eff
    # otherwise: datakeys as list of keys to retrieve info, e.g. from a grapa file
    pkeys, pvals, nparams = helper.as_labels_datatable(graph, datakeys)

    # filter data
    pvalshape = list(pvals.shape)
    pvals = filter_pvals(pkeys, pvals, filters=filters)
    # if np.sum(pvalshape) != np.sum(pvals.shape):
    #    print(pvalshape, pvals.shape)
    #    fnamebase += "_filter"

    if np.isnan(pvals).all():
        print("Data table of parameters contains only NaN, script stops here.")
        print(pkeys)
        print(pvals.shape)
        return graph

    # export and plot results
    writefile_datatable(fnamebase + "_correlation_table.txt", pkeys, pvals)

    # by default,  full correlation matrix
    # helper handles generic or Scaps variant
    serx, sery = seriesx, seriesy
    seriesx_, seriesy_ = helper.seriesxy(pkeys)
    if serx is None:
        serx = seriesx_
        print("Automatic choice of seriesx: {}.".format(serx))
    if sery is None:
        sery = seriesy_
        print("Automatic choice of seriesy: {}.".format(sery))

    # process datatable
    graphtable, graphpearson, pearson = process_datatable(
        fnamebase, pkeys, pvals, seriesx=serx, seriesy=sery, **ngkwargs
    )

    print("Script ended successfully")
    if graphtable is not None:
        return graphtable
    return graph


if __name__ == "__main__":
    datakeys = AUTO
    filters = []
    seriesx = None
    seriesy = None

    # Example
    # filename = r'G:\CIGS\RC\_simulations\20230508_Scaps_windowlayers\CIGS_RC\test\CdSX_CdSd_CdSn_ZnOd_ZnOX_AZOX.iv'
    # filename = r'G:\CIGS\RC\_simulations\20230508_Scaps_windowlayers\CIGS_RC\test\CIGStau_CIGSp.iv'
    # datakeys = ['batch parameters 0 value', 'batch parameters 1 value','temperature [k]']
    # filters = [["i_ZnO*affinit*", HIGHER, 4.45], [1, '>', 220]]

    # Example
    filename = r"..\examples\JV\SAMPLE_B_3layerMo\I-V_SAMPLE_B_3LayerMo_Param.txt"
    # datakeys = AS_DATATABLE  # not necessarily needed, autodetection should work
    filters = [["Jsc_mApcm2", HIGHER, 10]]
    seriesx = range(11)

    # Example
    # filename = r"..\examples\JV\SAMPLE_B_3layerMo\export_SAMPLE_B_3LayerMo_summary_allJV.txt"
    # datakeys = ["Voc", "Jsc", "FF", 'area']  # , 'Eff', 'Rp', 'acquis soft rs']
    # filters = [["Jsc", HIGHER, 10]]

    process_file(
        filename, datakeys=datakeys, filters=filters, seriesx=seriesx, seriesy=seriesy
    )

    # files = glob.glob('test/*.iv')
    # print('Files:', files)
    # for file in files:
    #     process_scaps_file(file)
