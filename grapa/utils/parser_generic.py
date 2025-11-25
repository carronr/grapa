"""Generic file parser, to read data organized in columns,
with some header lines at the beginning.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import ast
from re import sub as resub
import logging
from typing import List, TYPE_CHECKING

import numpy as np

from grapa import KEYWORDS_GRAPH, KEYWORDS_CURVE, KEYWORDS_HEADERS
from grapa.curve import Curve
from grapa.mathModule import is_number
from grapa.utils.error_management import issue_warning
from grapa.utils.string_manipulations import strUnescapeIter, strToVar
from grapa.utils.container_metadata import MetadataContainer

if TYPE_CHECKING:
    from grapa.graph import Graph

logger = logging.getLogger(__name__)


def _identify_delimiter(lines: list) -> str:
    # determine data character separator - only useful if reads file,
    # otherwise is supposed to be already separated
    threshold = np.ceil(max(1, len(lines) - 5) / 2)
    delimiters = ["\t", ",", ";", " "]
    numbers = []
    for delimiter in delimiters:
        numbers.append(np.sum([line.count(delimiter) for line in lines]))
        if numbers[-1] > threshold:
            if delimiter != delimiters[0]:
                print("Identified '{}' as delimiter.".format(delimiter))
            return delimiter

    idx = np.argmax(numbers)
    print(idx)
    msg = "WARNING _identify_delimiter: no suitable delimiter found, choose '{}'."
    print(msg.format(delimiters[0]))
    return delimiters[0]


class FileParserGeneric:
    """Methods to parse a file, organized with header key- values,
    then multicolumn data"""

    @classmethod
    def read(
        cls,
        graph: "Graph",
        attributes: dict,
        fileContent=None,
        ifReplaceCommaByPoint=False,
        **kwargs
    ):
        """
        Reads the file as a column-organized data file, with some headers lines at the
        beginning.

        :param graph: the graph to put the data in
        :param attributes: provided when intantiating the Graph object. Special keyword:

               - _singlecurve: parse all data into a single Curve object

        :param fileContent: if content to parse is given as a string, not from file read
        :param ifReplaceCommaByPoint: if numbers are formatted with ',' instead of '.'
        :param kwargs: the following keywords are handled:

               - delimiter: if provided, assumed the values is the text delimiter.

               - delimiterHeaders: same as delimiter, but only for the headers section

               - lstrip: if provided, a lstrip function is applied to each line when
                 parsing the file. Values: None (lstrip(None)), True, or '# ' etc.
        """
        # Ideas for improvement:
        # - read_lines: use it as a generator (through class?). To avoid loading the
        #   entire file in memory, only header

        # parsing inputs
        singlecurve = cls._get_singlecurve(attributes)
        is_source_file = bool(fileContent is None)
        lstriparg = cls._get_lstriparg(kwargs)
        filename = graph.filename

        # update default information
        graph._curves.clear()

        # open file, start to process content
        lines = cls.read_lines(filename, fileContent, ifReplaceCommaByPoint, lstriparg)
        delimiter, delimiter_headers = cls._get_delimiters(lines, kwargs)
        last_linesplit, collabels, attr_curves, skip_header = cls._update_graph_header(
            graph,
            lines,
            is_source_file,
            attributes,
            delimiter,
            delimiter_headers,
        )

        # validate inputs, set correct values to later parse data
        graph.text_check_valid()
        skip_header = cls._finalize_skip_header(
            singlecurve, last_linesplit, lines, is_source_file, delimiter, skip_header
        )
        skip_footer = cls._get_skip_footer(lines)
        collabels = cls._update_graph_labels(graph, attr_curves, collabels)
        if skip_header >= len(lines):
            return

        # retrive data, located after headers
        if is_source_file:
            first_line_content = lines[skip_header]
            data = cls._data_from_genfromtxt(
                filename,
                skip_header,
                skip_footer,
                first_line_content,
                delimiter,
                ifReplaceCommaByPoint,
            )
        else:
            data = cls._data_from_filecontent(fileContent, skip_header, skip_footer)
        cols_nan, data = cls._are_cols_empty(data, lines, skip_header, attr_curves)

        cls._resolve_conflicts_attributes(attributes, attr_curves)
        # create curves
        if singlecurve:
            coly_idx = cls._create_curve_one(graph, data, attributes, collabels)
        else:
            coly_idx = cls._create_curves(graph, data, attributes, cols_nan, collabels)
        if len(coly_idx) > 0:
            cls._apply_attr_curves(graph, attr_curves, coly_idx)
            cls.cast_curves(graph)

    @staticmethod
    def _get_lstriparg(kwargs):
        """kwargs["lstrip"]: bool, str, or None (default lstrip), or not specified
        Returns as a list, or None.
        None: lstrip action skipped.
        [None]: lstrip()
        [str]: lstrip(str)"""
        if "lstrip" in kwargs:
            if isinstance(kwargs["lstrip"], bool) and kwargs["lstrip"]:
                return [None]
            if isinstance(kwargs["lstrip"], str) or kwargs["lstrip"] is None:
                return [kwargs["lstrip"]]
            msg = "lstrip argument not understood, ignored ({})."
            issue_warning(logger, msg, kwargs["lstrip"])
        return None

    @classmethod
    def _get_singlecurve(cls, attributes) -> bool:
        singlecurve = False
        if "_singlecurve" in attributes and attributes["_singlecurve"]:
            singlecurve = True
            del attributes["_singlecurve"]
        return singlecurve

    @classmethod
    def read_lines(
        cls,
        filename: str,
        file_content,
        replace_comma_dots: bool,
        lstriparg,
    ) -> List[str]:
        """Read a file, returns the lines as a list of str

        :param filename: str of the filename to open
        :param file_content: None, or alternative content (if so, filename is not open).
        :param replace_comma_dots: bool
        :param lstriparg: if not None, .lstrip(\\*lstriparg) on lines.
                          [str | None] | None]
        :returns: a list of str, one str per line in file
        """
        # if some content was provided
        if file_content is not None:
            return file_content

        # parse content of file
        with open(filename, "r") as file:
            lines = [line.rstrip(":\r\n\t") for line in file]
        if replace_comma_dots:
            lines = [
                resub(
                    "(?P<a>[0-9]),(?P<b>[0-9])",
                    lambda word: word.group("a") + "." + word.group("b"),
                    line,
                )
                for line in lines
            ]
        if lstriparg is not None:
            lines = [line.lstrip(*lstriparg) for line in lines]
        return lines

    @staticmethod
    def _get_delimiters(lines, kwargs):
        # identify data separator (tab, semicolumn, etc.)
        if "delimiter" in kwargs and isinstance(kwargs["delimiter"], str):
            delimiter = kwargs["delimiter"]
        else:
            delimiter = _identify_delimiter(lines)
        delimiter_headers = kwargs.get("delimiterHeaders", None)
        return delimiter, delimiter_headers

    @classmethod
    def _get_collabels(cls, attributes, last_line):
        # if some column names were identified, try to define
        # labels by concatenating filename stripped from numerics
        # with colum name
        lbl = [""]
        if "label" in attributes:
            lbl = attributes["label"].split(" ")
        for i in range(len(lbl) - 1, -1, -1):
            try:
                float(lbl[i])
                del lbl[i]
            except ValueError:
                if lbl[i] in ["File"]:
                    del lbl[i]
        suff = " ".join(lbl)
        if len(suff) > 1:
            suff += " "
        collabels = [suff + str(l).replace('"', "") for l in last_line]
        if "label" in attributes:
            del attributes["label"]
        return collabels

    @classmethod
    def _convert_keywords_legacy(cls, graph, linesplit: list, val):
        """conversion from legacy keywords and matlab terminology"""
        strreplace = {"â²": "2"}
        for old, new in strreplace.items():
            linesplit[0] = linesplit[0].replace(old, new)
        rename = {
            "facealpha": "alpha",
            "legendlocation": "legendproperties",
            "legend": "label",
            "xycurve": "curve",
        }
        if linesplit[0] in rename:
            linesplit[0] = rename[linesplit[0]]
        # figuresize
        if linesplit[0] == "figuresize":
            linesplit[0] = "figsize"
            if isinstance(val, str):
                try:
                    if len(ast.literal_eval(val)) == 4:
                        val = ast.literal_eval(val)
                        val = val[2:4]
                        if min(val) > 30:
                            # assumes units are given in pixels -> convert
                            # in inches, assuming FILEIO_DPI screen resolut
                            val = [x / graph.FILEIO_DPI for x in val]
                except Exception as e:
                    msg = "Exception readDataFromFileGeneric figuresize. {}: {}."
                    issue_warning(logger, msg, type(e), e)
        return linesplit, val

    @classmethod
    def _update_graph_header_endloop(
        cls, attributes, last_line, last_keyword, delimiter
    ):
        collabels = []
        is_label_default = False
        if "filename" in attributes:
            fname = attributes["filename"].split("\\")[-1].split("/")[-1]
            lbltest = ".".join(fname.split(".")[:-1]).replace("_", " ")
            if "label" in attributes and attributes["label"] == lbltest:
                is_label_default = True

        if len(last_line) == 1:
            couple2 = cls._splitline(last_line[0], delimiter)
            if len(couple2) > 1:
                last_line = couple2

        if last_keyword != "" and (
            "label" not in attributes
            or (
                is_label_default
                and len(last_line) > 1
                and last_keyword
                not in KEYWORDS_GRAPH["keys"]
                + KEYWORDS_HEADERS["keys"]
                + KEYWORDS_CURVE["keys"]
            )
        ):
            collabels = cls._get_collabels(attributes, last_line)
        return collabels, last_line

    @classmethod
    def _update_graph_header(
        cls,
        graph,
        lines: list,
        is_source_file: bool,
        attributes: dict,
        delimiter: str,
        delimiter_headers,
    ):
        skip_header = 0
        last_linesplit = ""
        collabels = []
        attr_curves = {}

        # process header lines
        last_keyword = ""  # key of last saved parameter
        n_empty = 0
        for line in lines:
            if is_source_file:
                linesplit = cls._splitline(line, delimiter)
            else:
                linesplit = line
            # stop condition - end of headers: first line starting with numerical value
            if is_number(linesplit[0]):
                collabels, last_linesplit = cls._update_graph_header_endloop(
                    attributes, last_linesplit, last_keyword, delimiter
                )
                break

            skip_header += 1
            # delimiter_headers if provided - after stop condition (based on data)
            if delimiter_headers is not None and is_source_file:
                linesplit = cls._splitline(line, delimiter_headers)

            # interpret data
            if linesplit[0] == "":
                if len(linesplit) == 1 or len("".join(linesplit)) == 0:
                    continue  # empty line, ignore (yet skip_header + 1)

                linesplit[0] = "empty" + str(n_empty)  # avoid overwrite header lines
                n_empty += 1

            val = linesplit[0]
            if len(linesplit) > 1:
                val = float(linesplit[1]) if is_number(linesplit[1]) else linesplit[1]

            linesplit[0] = MetadataContainer.format(linesplit[0])  # keywords lowercase
            linesplit, val = cls._convert_keywords_legacy(graph, linesplit, val)
            attr_curves.update(cls._header_to_graph(graph, linesplit, val))
            last_keyword = linesplit[0]
            last_linesplit = linesplit
        return last_linesplit, collabels, attr_curves, skip_header

    @staticmethod
    def _splitline(line: str, delimiter: str) -> List[str]:
        couple_ = line.split(delimiter)
        return [couple_[i_].strip(" :") for i_ in range(len(couple_))]

    @classmethod
    def _header_to_graph(cls, graph, linesplit: list, val_header) -> dict:
        keyword = linesplit[0]
        # identify headers, graphinfo values
        if keyword in KEYWORDS_HEADERS["keys"]:
            graph.update({keyword: strToVar(val_header)})
            return {}
        if keyword in KEYWORDS_GRAPH["keys"] or keyword.startswith("subplots"):
            graph.update({keyword: strToVar(val_header)})
            return {}

        # send everything else in Curve, leave nothing for sampleInfo
        # do not use val
        # format values
        if keyword in ["filename"]:  # no backslashes in 'filename'
            for i in range(1, len(linesplit)):
                linesplit[i] = linesplit[i].replace("\\", "/")
        # deal with escape sequences
        for i in range(1, len(linesplit)):
            if keyword in ["label"]:  # enforce type str (e.g. label '1' and not 1.00)
                linesplit[i] = strUnescapeIter(linesplit[i])
            else:
                linesplit[i] = strToVar(linesplit[i])
        # store to use later
        return {linesplit[0]: linesplit}

    @classmethod
    def _finalize_skip_header(
        cls,
        singlecurve,
        last_linesplit,
        lines: list,
        is_source_file: bool,
        delimiter: str,
        skip_header: int,
    ):
        # if must parse last "header" line nevertheless (first element blank)
        if (
            singlecurve
            and len(last_linesplit) > 1
            and last_linesplit[0].startswith("empty")
        ):
            linei = skip_header - 1
            # retrieve last header line and first content and compare if
            # positions of numeric match
            couple1 = []
            if is_source_file:
                couple0 = cls._splitline(lines[linei], delimiter)
                if linei + 1 < len(lines):
                    couple1 = cls._splitline(lines[linei + 1], delimiter)
            else:
                couple0 = lines[linei]
                if linei + 1 < len(lines):
                    couple1 = lines[linei + 1]

            if len(couple0) == len(couple1):
                isnum0 = [is_number(v) for v in couple0]
                isnum1 = [is_number(v) for v in couple1]
                test = [bool(isnum0[i] == isnum1[i]) for i in range(1, len(isnum0))]
                if np.array(test).all():
                    skip_header = max(skip_header - 1, 0)
        return skip_header

    @classmethod
    def _update_graph_labels(cls, graph: "Graph", attr_curves: dict, collabels: list):
        # if last text line was not empty -> interpret that as x-y-labels
        labeltest = collabels
        if len(labeltest) == 0:
            if "label" in attr_curves:
                labeltest = attr_curves["label"]
        if len(labeltest) > 0:
            if not graph.has_attr("xlabel"):
                if labeltest[0] not in ["", "label"]:
                    graph.update({"xlabel": labeltest[0]})
            if not graph.has_attr("ylabel") and len(labeltest) > 1:
                if len(labeltest[1]) not in [""]:
                    graph.update({"ylabel": labeltest[1]})
        return collabels

    @staticmethod
    def _get_skip_footer(lines):
        skip_footer = 0
        while lines[-1] == "":
            lines.pop()
            skip_footer += 1
        # do not understand it... genfromtxt can fail when files ends with
        # several \n and this is not set!?
        skip_footer = 0
        return skip_footer

    @classmethod
    def _data_from_genfromtxt(
        cls,
        filename,
        skip_header,
        skip_footer,
        first_line_content,
        delimiter,
        replace_commas_dots,
    ):
        # default behavior
        usecols = range(0, len(first_line_content.split(delimiter)))
        dict_converters = {}
        if replace_commas_dots:
            lambd = lambda x: float(
                str(str(x, "UTF-8") if not isinstance(x, str) else x).replace(",", ".")
            )
            for i in usecols:
                dict_converters.update({i: lambd})
        kwargs_genfromtxt = {
            "usecols": usecols,
            "delimiter": delimiter,
            "skip_header": skip_header,
            "skip_footer": skip_footer,
            "invalid_raise": False,
        }
        if len(dict_converters) > 0:
            kwargs_genfromtxt.update({"converters": dict_converters})
        data = np.transpose(np.genfromtxt(filename, **kwargs_genfromtxt))
        return data

    @classmethod
    def _data_from_filecontent(
        cls,
        file_content,  # gets modified
        skip_header,
        skip_footer,
    ):
        # if some content was provided
        for i in range(skip_header, len(file_content) - skip_footer):
            for j in range(len(file_content[i])):
                file_content[i][j] = (
                    np.float64(file_content[i][j])
                    if file_content[i][j] != "" and is_number(file_content[i][j])
                    else np.nan
                )
        data = np.transpose(np.array(file_content[skip_header:]))
        return data

    @classmethod
    def _are_cols_empty(cls, data, lines, skip_header, attr_curves):
        # some checks,
        # and build array test -> know which data column are not empty (returned as nan
        # values)
        if len(data.shape) < 2 and len(lines) == skip_header + 1:
            # only 1 data row
            cols_nan = [np.isnan(v) for v in data]
            data = data.reshape((len(data), 1))
        else:
            if len(data.shape) < 2 and len(lines) > skip_header + 1:
                # if only 1 data colum
                data = np.array([range(len(data)), data])
                for key in attr_curves:
                    attr_curves[key] = [attr_curves[key][0]] + attr_curves[key]
            cols_nan = [np.isnan(data[i, :]).all() for i in range(data.shape[0])]
        return cols_nan, data

    @classmethod
    def _resolve_conflicts_attributes(cls, attributes, attr_curves):
        # do not fill in 'filename', or 'label' default value if filename
        # is amongst the file parameters
        for key in attr_curves:
            if key in attributes:
                del attributes[key]

    @classmethod
    def _create_curve_one(cls, graph: "Graph", data, attributes, collabels):
        collabelx = collabels[0] if len(collabels) > 0 else ""
        collabely = collabels[0] if len(collabels) > 0 else ""
        graph.append(Curve(data, attributes))
        graph[-1].update({"_collabels": [collabelx, collabely]})
        return [1]

    @classmethod
    def _create_curves(cls, graph, data, attributes, cols_nan, collabels):
        coly_idx = []
        # parse through data
        col_x = 0
        for col_y in range(1, len(cols_nan)):  # normal column guessing
            # if nan -> column empty -> start new xy pair
            if cols_nan[col_y]:
                col_x = col_y + 1
            elif col_y > col_x:
                # removing of nan pairs is performed in the Curve constructor
                reshape = (2, len(data[col_y, :]))
                curvedata = np.append(data[col_x, :], data[col_y, :]).reshape(reshape)
                graph.append(Curve(curvedata, attributes))
                collabelx = collabels[col_x] if col_x < len(collabels) else ""
                collabely = collabels[col_y] if col_y < len(collabels) else ""
                graph[-1].update({"_collabels": [collabelx, collabely]})
                coly_idx.append(col_y)
        return coly_idx

    @classmethod
    def _apply_attr_curves(cls, graph, attr_curves, coly_idx):
        # apply parsed_attr to different curves
        for key, value in attr_curves.items():
            # attr_curves[key] has form [key, val_txt_col0, val_txt_col1, val_txt_col2,]
            # need to remap onto curves index
            if len(value) < max(coly_idx) + 1:
                attr_curves[key].extend([""] * max(coly_idx))
            for c, idx in enumerate(coly_idx):
                try:
                    val = value[idx]
                except IndexError:
                    msg = (
                        "IndexError in _apply_attr_curves. Proceed further. "
                        "%s. %s, c %s, idx %s, %s."
                    )
                    issue_warning(logger, msg, graph.filename, key, c, idx, value)
                else:
                    graph[c].update({key: val})
        # find label in case not specified
        if "label" not in attr_curves:
            for curve in graph:
                if curve.attr("label") == "":
                    curve.update({"label": curve.attr("_collabels")[1]})

    @classmethod
    def cast_curves(cls, graph: "Graph"):
        """cast Curves as child class if specified by attr 'curve'"""
        for c, curve in enumerate(graph):
            newtype = curve.attr("curve")
            if newtype not in ["", "curve", "curvexy"]:
                graph.castCurve(newtype, c, silentSuccess=True)
