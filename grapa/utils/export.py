"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import logging
from typing import TYPE_CHECKING
from re import findall as refindall

import numpy as np

from grapa import KEYWORDS_CURVE
from grapa.curve import Curve
from grapa.utils.string_manipulations import varToStr
from grapa.utils.error_management import issue_warning, FileNotCreatedError


if TYPE_CHECKING:
    from grapa.graph import Graph

logger = logging.getLogger(__name__)


def export_filesave_default(graph: "Graph", filesave=""):
    """(private) default filename when saving."""
    if filesave == "":
        fname = graph.filename.split("\\")[-1].split("/")[-1].replace(".", "_")
        filesave = fname + "_"
        if len(graph) > 0:
            filesave += str(graph[-1].attr("complement"))
        if filesave == "_":
            filesave = "graphexport"
        if len(graph) > 1:
            filesave += "_series"
    return filesave


class GraphExporter:
    """A library of functions usefule to export Graphs into a txt file.
    Packed into a unique class for encapsulation purpose."""

    @classmethod
    def export_as_text(
        cls,
        graph: "Graph",
        filesave: str = "",
        save_altered=False,
        if_template=False,
        if_compact=True,
        if_only_labels=False,
        if_clipboard_export=False,
    ) -> str:
        """
        Exports content of Graph object into a human- and machine-readable format.
        """
        # retrieve information of data alteration (plotting mode)
        alter = graph.get_alter()
        # handle saving of original data or altered data
        attr_restore = {}
        if save_altered:
            # if modified parameter, save modified data, and not save the alter attribute
            attr_restore.update({"alter": graph.attr("alter")})
            graph.update({"alter": ""})
        else:
            alter = ["", ""]  # want to save original data, and alter attribute

        # format graph content into one (large) string
        out = ""
        if not if_only_labels:
            out += cls._format_graph_attr(graph, if_template)
        keys_list, curvedata, separator = cls._prepare_keyslist_curvedata_separator(
            graph, alter, save_altered, if_only_labels, if_compact, if_template
        )
        out += cls._format_curves_attr(graph, keys_list, separator)
        out += cls._format_curves_data(graph, curvedata, separator, if_template)

        # restore initial state of Graph object
        graph.update(attr_restore)

        # end: return (large) string, or save file
        if if_clipboard_export:
            return out

        # output
        filename = export_filesave_default(graph, filesave) + ".txt"
        if not graph.attr("savesilent", True):
            print("Graph data saved as", filename.replace("/", "\\"))
        graph.fileexport = os.path.normpath(filename)
        cls.writefile(filename, out)
        return filename

    @staticmethod
    def export_as_xml(graph: "Graph", filesave: str = "", **kwargs) -> str:
        """
        Exports Graph as .xml file.
        Not quite mature nor useful (no clear advantage over default format)

        :param graph: a Graph object
        :param filesave:
        :param kwargs: empty.
        :return: name of the saved file
        """
        for key, value in kwargs.items():
            msg = "export_xml: received keyword {}: {}, don't know what to do with it."
            issue_warning(logger, msg.format(key, value))
        try:
            from lxml import etree
        except ImportError as e:
            msg = "export_xml ImportError lxml. try 'pip install lxml'. %s"
            logger.error(msg, e, exc_info=True)
            raise FileNotCreatedError(msg % e)

        root = etree.Element("graph")
        # write headers
        for container in ["headers", "graphInfo"]:
            if hasattr(graph, container) and len(getattr(graph, container)):
                tmp = etree.SubElement(root, container)
                for key in getattr(graph, container):
                    tmp.set(key, str(getattr(graph, container)[key]))
        # write curves
        for c, curve in enumerate(graph):
            tmp = etree.SubElement(root, "curve" + str(c))
            attr = etree.SubElement(tmp, "attributes")
            for key in curve.get_attributes():
                attr.set(key, str(curve.attr(key)))
            data = etree.SubElement(tmp, "data")
            x = etree.SubElement(data, "x")
            y = etree.SubElement(data, "y")
            x.text = ",".join([str(e) for e in curve.x()])
            y.text = ",".join([str(e) for e in curve.y()])
        # save
        filesave = export_filesave_default(graph, filesave)
        if len(filesave) < 4 or filesave[-4:] != ".xml":
            filesave += ".xml"
        tree = etree.ElementTree(root)
        tree.write(filesave, pretty_print=True, xml_declaration=True, encoding="utf-8")
        if not graph.attr("savesilent"):
            print("Graph data saved as", filesave.replace("/", "\\"))
        return filesave

    @staticmethod
    def writefile(filename, text):
        """Writes text to a file."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                try:
                    f.write(text)
                except UnicodeEncodeError as e:
                    # give some details to the user, otherwise can become quite
                    # difficult to identify where the problem originates from
                    msgfull = "UnicodeEncodeError in Exporter: could not save the file."
                    ident = "in position "
                    errmsg = str(e)
                    if ident in errmsg:
                        idx = errmsg.find(ident) + len(ident)
                        chari = refindall("[0-9]+", errmsg[idx:])
                        if len(chari) > 0:
                            chari = int(chari[0])
                            msg = text[max(0, chari - 20) : min(len(text), chari + 15)]
                            msgfull += " --> surrounding characters: "
                            msgfull += msg.replace("\n", "\\n")
                    msgfull += ". {}, {}.".format(type(e), e)
                    logger.error(msgfull, exc_info=True)
                    raise FileNotCreatedError(msgfull)
        except PermissionError as e:
            msg = "PermissionError in graphIO.export, filename %s. %s."
            msa = (filename, e)
            logger.error(msg, *msa)
            raise FileNotCreatedError(msg % msa)

    @staticmethod
    def _format_graph_attr(graph: "Graph", if_template):
        out = ""
        # ifOnlyLabels: only export label argument
        for key, value in graph.graphinfo.items():
            out = out + key + "\t" + varToStr(value) + "\n"
        if not if_template:
            keylist_headers = ["meastype"]
            for key in keylist_headers:
                if key in graph.headers:
                    out = out + key + "\t" + varToStr(graph.headers[key]) + "\n"
        return out

    @staticmethod
    def _format_curves_data(graph: "Graph", curvedata, separator, if_template):
        out = ""
        data_len = 0
        if len(graph) > 0:
            data_len = np.max([len(data[0]) for data in curvedata])
        separator[0] = "\t\t\t"
        for i in range(data_len):
            for c, _curve in enumerate(graph):
                if if_template:
                    out += "1" + "\t" + "0" + "\t\t"
                else:
                    if i < len(curvedata[c][0]):  # curve.shape(1):
                        if c > 0:
                            out += "\t"
                            if len(separator[c]) == 3:
                                out += "\t"
                        if len(separator[c]) == 3:
                            # vix = curvedata[c][0][i]  #funcx(curve,index=i,alter=alter[0])
                            # viy = curvedata[c][1][i]  #funcy(curve,index=i,alter=alter[1])
                            out += (
                                str(curvedata[c][0][i]) + "\t" + str(curvedata[c][1][i])
                            )
                        else:  # funcy(curve, index=i, alter=alter[1]))
                            out += str(curvedata[c][1][i])
                    else:
                        if c == 0:
                            separator[0] = "\t"
                        out += separator[c]

            out += "\n"
            if if_template:
                break
        return out

    @staticmethod
    def _format_curves_attr(graph: "Graph", keys_list, separator):
        out = ""
        keys_list.sort()
        for key in keys_list:
            out = out + key
            for c, curve in enumerate(graph):
                value = curve.attr(key)
                if isinstance(value, np.ndarray):
                    value = list(value)
                out = out + separator[c] + varToStr(value)
            out = out.rstrip("\t") + "\n"
        return out

    @staticmethod
    def _prepare_keyslist_curvedata_separator(
        graph, alter, save_altered, if_only_labels, if_compact, if_template
    ):
        # format information for graph specific for each curve
        # first establish list of attributes, then loop over curves to construct export
        keys_list = ["label"]
        if not if_only_labels:  # if ifOnlyLabels: only export 'label'
            for curve in graph:
                for attr in curve.get_attributes():
                    if attr not in keys_list:
                        if attr in KEYWORDS_CURVE["keys"] or (not if_template):
                            keys_list.append(attr)
        # if save altered Curve export curves modified for alterations and
        # offsets, and delete offset and muloffset key
        if save_altered:
            funcx, funcy = Curve.x_offsets, Curve.y_offsets
            if "offset" in keys_list:
                keys_list.remove("offset")
            if "muloffset" in keys_list:
                keys_list.remove("muloffset")
        else:
            funcx, funcy = Curve.x, Curve.y
        curvedata = [
            [funcx(c, alter=alter[0]), funcy(c, alter=alter[1])] for c in graph
        ]

        # if compact mode, loop over curves to identify consecutive with identical xaxis
        separator = ["\t"] + ["\t\t\t"] * (len(graph) - 1) + ["\t"]
        if if_compact and not if_template:
            for c in range(1, len(graph)):
                if np.array_equal(curvedata[c][0], curvedata[c - 1][0]):
                    separator[c] = "\t"
        return keys_list, curvedata, separator


def export(
    graph: "Graph",
    filesave: str = "",
    save_altered=False,
    if_template=False,
    if_compact=True,
    if_only_labels=False,
    if_clipboard_export=False,
) -> str:
    """
    Exports content of Graph object into a human- and machine-readable format.
    """
    if filesave.endswith(".xml"):
        return GraphExporter.export_as_xml(
            graph, filesave=filesave, saveAltered=save_altered
        )

    return GraphExporter.export_as_text(
        graph,
        filesave=filesave,
        save_altered=save_altered,
        if_template=if_template,
        if_compact=if_compact,
        if_only_labels=if_only_labels,
        if_clipboard_export=if_clipboard_export,
    )
