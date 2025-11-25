"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import logging
from typing import Optional, List, TYPE_CHECKING
from re import findall as refindall

import numpy as np

from grapa.mathModule import is_number
from grapa.utils.string_manipulations import strToVar
from grapa.utils.error_management import FileNotReadError, issue_warning
from grapa.curve import Curve

if TYPE_CHECKING:
    from grapa.graph import Graph

logger = logging.getLogger(__name__)


class FileParserDatabase:
    """
    fileContent: list of lists containing file content (imagine csv file)
    kwargs: useless here
    """

    GRAPHTYPE_DATABASE = "Database"

    @classmethod
    def read(
        cls,
        graph: "Graph",
        attributes: dict,
        filecontent: Optional[List[List[str]]] = None,
        **_kwargs
    ):
        """Read the file, populate Graph object"""

        graph.update({"meastype": cls.GRAPHTYPE_DATABASE})

        content = filecontent
        if content is None:
            content = cls.read_file_content(graph.filename)

        cls._suppress_empty_lines(content)
        max_n_cols = max(len(line) for line in content)  # max length of a line

        # start processing - identify column names
        graph.update({"collabelsdetail": [], "rowlabels": [], "rowlabelnums": []})
        # check if further lines as header - must be pure text lines.
        # we store lines consisting only in a pair of elements. In this case
        # this should not be a column name, instead we will graph.update()
        to_update, nlines_headers = cls._what_is_to_update(content, graph)
        # combine all column names identified so far
        labels = cls._get_labels(graph, max_n_cols)
        # parsing content
        if len(graph) > 0:
            msg = (
                "readDataFromFileDatabase, data container NOT empty. Content of"
                " data deleted, previous data lost."
            )
            issue_warning(logger, msg)
        graph._curves.clear()

        # fill the Curve object with list of Curves, still empty at this stage
        for i in range(max_n_cols - 1):
            graph.append(
                Curve(np.zeros((2, len(content) - nlines_headers)), attributes)
            )
            graph[-1].update({"label": labels[i]})
            graph[-1].update({"_collabels": ["", labels[i]]})
        graph.update({"collabels": labels[:]})  # used in differetn scripts

        cls._apply_attr_update(graph, to_update)
        cls._populate_data(graph, content, nlines_headers)

    @staticmethod
    def read_file_content(filename: str):
        """read the file, returns content as list of lists"""
        content = None
        # this seems to parse the file satisfyingly
        try:
            with open(filename, "r") as file:
                content = [line.strip(":\r\n").split("\t") for line in file]
        except UnicodeError as e:
            try:
                with open(filename, "r", encoding="ascii") as file:
                    content = [line.strip(":\r\n").split("\t") for line in file]
                print("FileParserDatabase unicode error, opened as ascii")
            except Exception:
                msg = "Cannot open file %s: %s."
                msa = (filename, e)
                logger.error(msg, *msa, exc_info=True)
                raise FileNotReadError(msg % msa)
        return content

    @staticmethod
    def _suppress_empty_lines(content: List[list]):
        # suppress empty lines
        for i in np.arange(len(content) - 1, -1, -1):
            if sum([len(content[i][j]) for j in range(len(content[i]))]) == 0:
                del content[i]
            else:
                for j in range(len(content[i])):
                    content[i][j] = content[i][j].replace("\\n", "\n")

    @staticmethod
    def _what_is_to_update(content: List[list], graph: "Graph"):
        to_update = {}
        lencols = sorted([len(line) for line in content], reverse=True)
        maxlencols2 = lencols[1] if len(lencols) > 1 else lencols[0]
        for i, line in enumerate(content):
            only_text = True
            n_val = 0
            for val in line:
                if is_number(val):
                    only_text = False
                    break
                if val != "":
                    n_val = n_val + 1
            if not only_text and not (len(line) == 2 and maxlencols2 > 2):
                break  # should accept pairs property - numerical value

            if n_val == 1 and line[0] != "":
                split = line[0].split(": ")
                if len(split) > 1:
                    to_update.update({split[0]: split[1:]})
                elif len(line) == 2 and is_number(line[1]):
                    to_update.update({line[0]: float(line[1])})
                else:
                    to_update.update({split[0]: ""})
            elif n_val == 2 and line[0] != "" and line[1] != "":
                to_update.update({line[0]: line[1]})
            else:
                graph.headers["collabelsdetail"].append(content[i][0:])
        nlines_headers = len(graph.attr("collabelsdetail")) + len(to_update)
        return to_update, nlines_headers

    @staticmethod
    def _get_labels(graph: "Graph", max_n_cols: int) -> List[str]:
        collabelsdetail = graph.attr("collabelsdetail")
        labels = [""] * (max_n_cols - 1)
        for detail in collabelsdetail:
            for j in range(1, len(detail)):
                labels[j - 1] = labels[j - 1] + "\n" + detail[j]
        for i, label in enumerate(labels):
            labels[i] = label[1:].strip("\n ")
        return labels

    @staticmethod
    def _apply_attr_update(graph: "Graph", to_update: dict):
        for key in to_update:
            if key in ["color"]:
                to_update[key] = strToVar(to_update[key])

            if key == "label":
                for curve in graph:
                    old = curve.attr("label")
                    if not curve.has_attr("label_initial"):
                        curve.update({"label_initial": old})
                    curve.update({"label": "{} {}".format(to_update["label"], old)})
                if len(graph) > 0:
                    graph[0].update({"label": to_update["label"]})
            else:
                graph.update({key: to_update[key]}, if_all=True)

    @staticmethod
    def _populate_data(graph: "Graph", content: List[List], nlines_headers: int):
        rowlabels: list = graph.attr("rowlabels")
        rowlabelnums: list = graph.attr("rowlabelnums")
        for i, line in enumerate(content[nlines_headers:]):
            rowlabels.append(line[0])
            try:
                # self.headers['rowlabelnums'].append(float(''.join(ch for ch
                # in line[0] if ch.isdigit() or ch=='.')))
                split = refindall("([0-9]*[.]*[0-9]*)", line[0])
                split = [s for s in split if len(s) > 0]
                # print (line[0], split)
                rowlabelnums.append(float(split[0]) if len(split) > 0 else np.nan)
            except Exception:
                rowlabelnums.append(np.nan)
            for j in range(len(line) - 1):
                graph[j].setX(rowlabelnums[-1], index=i)
                if is_number(line[j + 1]):
                    graph[j].setY(float(line[j + 1]), index=i)
        graph.update({"rowlabels": rowlabels, "rowlabelnums": rowlabelnums})
