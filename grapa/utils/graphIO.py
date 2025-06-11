# -*- coding: utf-8 -*-
"""Class and functions assisting with import and export of data.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import ast
import logging
from re import split as resplit, findall as refindall, sub as resub

import numpy as np

from grapa import KEYWORDS_GRAPH, KEYWORDS_CURVE, KEYWORDS_HEADERS
from grapa.mathModule import is_number
from grapa.curve import Curve
from grapa.utils.string_manipulations import strToVar, strUnescapeIter, varToStr

logger = logging.getLogger(__name__)


def file_read_first3lines(filename: str, silent=True) -> tuple:
    """Returns the first 3 lines of a file.

    :param filename: str, filename to open
    :param silent: if True, does not complain in case of issue.
    :return: (str, str, str) first 3 lines of the file
    """
    try:
        with open(filename, "r", errors="backslashreplace") as file:
            linesraw = [file.readline() for _ in range(3)]
            lines = [
                line.encode("ascii", errors="ignore").decode().rstrip(" \r\n\t")
                for line in linesraw
            ]
            return tuple(lines)
    except (IOError, OSError):
        if not silent:
            msg = "Exception read_file_first3lines"
            logger.error(msg, exc_info=True)
        return "", "", ""


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


class GraphIO:
    """
    By default, 3 different reading modes are possible:
    1. directly from data (filename=[[0,1,2,3],[5,2,7,4]])

    2. column-organized datafile, with data starting at first row with a number
    in the first field

    3. database, with no number in the first column

    File opening strategy:
        1/ readDataFile

            - do some stuff with complementary input

            - read first 3 lines of the file (file_read_first3lines)

            - check if any of Graph child class can handle the file

            - if not, calls readDataFromFileTxt

        2/ readDataFromFileTxt

            - look if file has a number in the first column (_funcReadDataFile).
                If so, call readDataFromFileGeneric,
                else call readDataFromFileDatabase.
    """

    GRAPHTYPE_DATABASE = "Database"
    GRAPHTYPE_GRAPH = "Graph"
    GRAPHTYPE_UNDETERMINED = "Undetermined data type"

    @staticmethod
    def append_curve_from_datainput(graph, data, attributes):
        """Initializes content of object using data given in constructor.

        :meta private:"""
        graph.append(Curve(data, {}))
        if not isinstance(attributes, dict):
            if attributes != "":
                msg = "append_curve_from_datainput: ignored attributes '{}'."
                logger.warning(msg.format(attributes))
            attributes = {}
        graph.update(attributes, if_all=False)
        # nevertheless want to import all subclass types, if not already done
        graph.get_list_subclasses_parse()

    @classmethod
    def readDataFile(cls, graph, complement=""):
        """
        Reads a file, opens graph.filename
        graph: a graph object
        Complement: ...

        :meta private:"""
        if not isinstance(complement, dict):
            if len(complement) > 0:
                msg = "readDataFile, data loss complement {}."
                logger.warning(msg.format(complement))
            complement = {}

        graph.isFileContent = False
        if "isfilecontent" in complement and complement["isfilecontent"]:
            graph.isFileContent = True

        # reads the file to identify types
        if not graph.isFileContent:
            fileName, file_ext = os.path.splitext(graph.filename)
            file_ext = file_ext.lower()
        else:
            fileName, file_ext = "", ".txt"
        fileNameBase = fileName.split("/")[-1].split("\\")[-1]
        attributes = {"filename": graph.filename}
        if len(complement) > 0:
            for key in complement:
                attributes.update({key.lower(): complement[key]})
            # attributes.update({"complement": complement})
        # default attributes of curve
        if "label" not in attributes:
            labeldefault = fileName.split("/")[-1].split("\\")[-1]
            labeldefault = labeldefault.replace("_", " ").replace("  ", " ")
            attributes.update({"label": labeldefault})
        # default (mandatory) attributes of Graph
        for key, default in graph.DEFAULT.items():
            if key in attributes:
                graph.update({key: attributes[key]})
                del attributes[key]
            else:
                graph.update({key: default})
        # guess keys of complement which are for Graph and not for Curves
        toDel = []
        for key in complement:
            try:
                if key.lower() in KEYWORDS_GRAPH["keys"]:
                    graph.update({key: complement[key]})
                    toDel.append(key)
            except Exception:
                pass
        for key in toDel:
            del complement[key]

        # read first line to identify datatype
        line1, line2, line3 = "", "", ""
        if not graph.isFileContent:
            line1, line2, line3 = file_read_first3lines(
                graph.filename, silent=graph.silent
            )
        else:
            graph.filename = graph.filename.replace("\r", "\n")
            line1 = graph.filename.split("\n")[0]
            if "isfilecontent" in attributes:
                del attributes["isfilecontent"]
            if attributes["label"] == "":
                del attributes["label"]

        msg_end = ""
        loaded = False
        # first check if instructed to open the file in a certain way
        if "readas" in attributes and attributes["readas"].lower() in [
            "database",
            "generic",
        ]:
            msg_end = "opened using standard methods."
            cls.readDataFromFileTxt(graph, attributes)
            loaded = True
        # if not, then try every possibility
        if not loaded:
            subclasses = graph.get_list_subclasses_parse()
            kwlines = {"line1": line1, "line2": line2, "line3": line3}
            for subclass in subclasses:
                if subclass.isFileReadable(fileNameBase, file_ext, **kwlines):
                    msg_end = "opened as " + subclass.FILEIO_GRAPHTYPE + "."
                    graph.update({"meastype": subclass.FILEIO_GRAPHTYPE})
                    res = subclass.readDataFromFile(
                        graph, attributes, fileName=fileNameBase, **kwlines
                    )
                    if res is None or res != False:
                        loaded = True
                        break

        # if not, then try default methods
        if not loaded:
            # other not readable formats
            if file_ext in [".pdf"]:
                msg_end = "do not know how to open."
            else:
                # keep the default opening mechanism in the main class
                msg_end = "opened using standard methods (default)."
                cls.readDataFromFileTxt(graph, attributes)

        if msg_end != "":
            if not graph.silent:
                msg = "File " + fileName.split("/")[-1] + "... " + msg_end
                logger.warning(msg)

    @classmethod
    def _funcReadDataFile(cls, content, attributes=None):
        """Try to guess which method is best suited to open the file"""
        if attributes is None:
            attributes = {}
        if "readas" in attributes:
            if attributes["readas"].lower() == "database":
                return cls.readDataFromFileDatabase
            if attributes["readas"].lower() == "generic":
                return cls.readDataFromFileGeneric
        # test to identify if file is organized as series of headers + x-y columns,
        # or as database
        for line in content:
            # return fileGeneric only if the first element of any row can be interpreted
            # as a number, ie. float(element) do not raise ValueError
            try:
                float(line[0])
                return cls.readDataFromFileGeneric
            except ValueError:
                pass
        return cls.readDataFromFileDatabase

    @classmethod
    def readDataFromFileTxt(cls, graph, attributes) -> bool:
        """
        Reads content of a .txt file, and parse it as a database or as a generic file.
        """
        if not graph.isFileContent:
            try:
                with open(graph.filename, "r", errors="replace") as file:
                    content = [resplit("[\t ,;]", line.strip(":\r\n")) for line in file]
            except FileNotFoundError:
                return False
            except (IOError, OSError, UnicodeError):
                logger.error("readDataFromFile Exception", exc_info=True)
                return False
        else:
            content = graph.filename.split("\n")
            delimiter = _identify_delimiter(content)
            content = [resplit(delimiter, line) for line in content]
            while len(content) > 0 and content[-1] == [""]:
                del content[-1]

        func = cls._funcReadDataFile(content, attributes)
        if not graph.isFileContent:
            # intentionally do not provide file content, function will read file
            # again with the desired data processing
            func(graph, attributes)
        else:
            graph.filename = ".txt"  # free some memory
            attributes.update({"filename": "From clipboard"})
            func(graph, attributes, fileContent=content)
            graph.isFileContent = False
        return True

    @classmethod
    def readDataFromFileGeneric(
        cls, graph, attributes, fileContent=None, ifReplaceCommaByPoint=False, **kwargs
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

               - lstrip: a lstrip function is applied to each line when parsing the
                 file. Values: True, or '# ' or similar

        """
        # idea:
        # 1/ read headers, store values in headers or in graphInfo.
        # 2/ read data, identify data structure
        # 3/ look back at headers to fill in dataInfo according to data
        #    structure
        # if fileContent provided, file is not read and fileContent is used
        # instead. Supposed to be a 2-dimensional list [line][col]

        # parsing inputs
        lstriparg = None
        if "lstrip" in kwargs:
            if isinstance(kwargs["lstrip"], bool) and kwargs["lstrip"]:
                lstriparg = []
            elif isinstance(kwargs["lstrip"], str):
                lstriparg = [kwargs["lstrip"]]
        # update default information
        graph.data = []
        parsedAttr = {}
        graph.update({"meastype": cls.GRAPHTYPE_UNDETERMINED})
        singlecurve = False
        if "_singlecurve" in attributes and attributes["_singlecurve"]:
            singlecurve = True
            del attributes["_singlecurve"]

        skip_header = 0
        skip_footer = 0
        lastSampleInfo = ""  # key of last saved parameter
        lastLine = ""
        col_labels = []

        # parse content of file
        if fileContent is None:  # default behavior
            with open(graph.filename, "r") as file:
                lines = [line.rstrip(":\r\n\t") for line in file]
            if ifReplaceCommaByPoint:
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
        else:  # if some content was provided
            lines = fileContent

        # identify data separator (tab, semicolumn, etc.)
        if "delimiter" in kwargs and isinstance(kwargs["delimiter"], str):
            delimiter = kwargs["delimiter"]
        else:
            delimiter = _identify_delimiter(lines)
        delimiterHeaders = kwargs.get("delimiterHeaders", None)

        def splitline(line, delimiter_):
            couple_ = line.split(delimiter_)
            return [couple_[i_].strip(" :") for i_ in range(len(couple_))]

        # process header lines
        numEmpty = 0
        for line in lines:
            if fileContent is None:
                couple = splitline(line, delimiter)
            else:
                couple = line
            # stops looping at first line with numerical values
            if is_number(couple[0]):
                isLabelDefault = False
                if (
                    "label" in attributes
                    and "filename" in attributes
                    and attributes["label"]
                    == ".".join(
                        attributes["filename"]
                        .split("\\")[-1]
                        .split("/")[-1]
                        .split(".")[:-1]
                    ).replace("_", " ")
                ):
                    isLabelDefault = True
                if len(lastLine) == 1:
                    couple2 = splitline(lastLine[0], delimiter)
                    if len(couple2) > 1:
                        lastLine = couple2
                if lastSampleInfo != "" and (
                    "label" not in attributes
                    or (
                        isLabelDefault
                        and len(lastLine) > 1
                        and lastSampleInfo
                        not in KEYWORDS_GRAPH["keys"]
                        + KEYWORDS_HEADERS["keys"]
                        + KEYWORDS_CURVE["keys"]
                    )
                ):
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
                    # column labels
                    col_labels = [suff + str(l).replace('"', "") for l in lastLine]
                    if "label" in attributes:
                        del attributes["label"]
                # don't save last couple, intepreted as column name
                break
            # end if with break

            # if not numerical value -> still information
            skip_header += 1
            # change delimiter is provided
            if delimiterHeaders is not None and fileContent is None:
                couple = splitline(line, delimiterHeaders)
            # interpret data
            if couple[0] == "":
                if len(couple) == 1 or len("".join(couple)) == 0:
                    continue
                couple[0] = "empty" + str(numEmpty)
                numEmpty += 1
            val = couple[0]
            if len(couple) > 1:
                val = float(couple[1]) if is_number(couple[1]) else couple[1]
            # all keywords in lowercase
            if not is_number(couple[0]):
                couple[0] = couple[0].lower()

            # conversion from matlab terminology, or legacy keywords
            if couple[0] == "figuresize":
                couple[0] = "figsize"
                try:
                    if len(ast.literal_eval(val)) == 4:
                        val = ast.literal_eval(val)
                        val = val[2:4]
                        if min(val) > 30:
                            # assumes units are given in pixels -> convert
                            # in inches, assuming FILEIO_DPI screen resolut
                            val = [x / graph.FILEIO_DPI for x in val]
                except Exception:
                    logger.error("readDataFromFileGeneric, when setting figure size")
            replace = {"facealpha": "alpha", "legendlocation": "legendproperties"}
            if couple[0] in replace:
                couple[0] = replace[couple[0]]
            # end of matlab conversion
            # identify headers values
            if couple[0] in KEYWORDS_HEADERS["keys"]:
                graph.headers[couple[0]] = strToVar(val)
            # identify graphInfo values
            elif couple[0] in KEYWORDS_GRAPH["keys"] or couple[0].startswith(
                "subplots"
            ):
                graph.graphinfo[couple[0]] = strToVar(val)
            # send everything else in Curve, leave nothing for sampleInfo
            else:
                # do not use val
                rep = {"legend": "label", "â²": "2", "xycurve": "curve"}
                for key, value in rep.items():
                    couple[0] = couple[0].replace(key, value)
                # format values
                if couple[0] in ["filename"]:  # don't want backslashes
                    for i in range(1, len(couple)):
                        couple[i] = couple[i].replace("\\", "/")
                # deal with escape sequences
                for i in range(1, len(couple)):
                    if couple[0] in ["label"]:
                        # enforce type str (e.g. label '1' and not 1.00)
                        couple[i] = strUnescapeIter(couple[i])
                    else:
                        couple[i] = strToVar(couple[i])
                # store to use later
                parsedAttr.update({couple[0]: couple})

            lastSampleInfo = couple[0]
            lastLine = couple
        # check that text input are correct
        graph.text_check_valid()
        # print('GraphIO parsedAttr', parsedAttr)
        # if must parse last "header" line nevertheless (first element blank)
        if singlecurve and len(lastLine) > 1 and lastLine[0].startswith("empty"):
            linei = skip_header - 1
            # retrieve last header line and first content and compare if
            # positions of numeric match
            couple1 = []
            if fileContent is None:
                couple0 = splitline(lines[linei], delimiter)
                if linei + 1 < len(lines):
                    couple1 = splitline(lines[linei + 1], delimiter)
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

        # if last text line was not empty -> interpret that as x-y-labels
        if len(col_labels) > 0:
            graph.update({"collabels": col_labels})
        if not graph.has_attr("collabels"):
            if "label" in parsedAttr:
                graph.update({"collabels": parsedAttr["label"]})
        if len(col_labels) > 0:
            if not graph.has_attr("xlabel"):
                graph.update({"xlabel": col_labels[0]})
            if not graph.has_attr("ylabel") and len(col_labels) > 1:
                graph.update({"ylabel": col_labels[1]})

        while lines[-1] == "":
            lines.pop()
            skip_footer += 1
        # do not understand it... genfromtxt can fail when files ends with
        # several \n and this is not set!?
        skip_footer = 0

        # look for data in file after headers
        if skip_header < len(lines):
            # default behavior
            if fileContent is None:
                usecols = range(0, len(lines[skip_header].split(delimiter)))
                dictConverters = {}
                if ifReplaceCommaByPoint:
                    lambd = lambda x: float(
                        str(str(x, "UTF-8") if not isinstance(x, str) else x).replace(
                            ",", "."
                        )
                    )
                    for i in usecols:
                        dictConverters.update({i: lambd})
                kwargs_genfromtxt = {
                    "skip_header": skip_header,
                    "delimiter": delimiter,
                    "invalid_raise": False,
                    "usecols": usecols,
                    "skip_footer": skip_footer,
                }
                if len(dictConverters) > 0:
                    kwargs_genfromtxt.update({"converters": dictConverters})
                data = np.transpose(np.genfromtxt(graph.filename, **kwargs_genfromtxt))
            else:
                # if some content was provided
                for i in range(skip_header, len(fileContent) - skip_footer):
                    for j in range(len(fileContent[i])):
                        fileContent[i][j] = (
                            np.float64(fileContent[i][j])
                            if fileContent[i][j] != "" and is_number(fileContent[i][j])
                            else np.nan
                        )
                data = np.transpose(np.array(fileContent[skip_header:]))
            colX = 0
            cols = []

            # some checks, and build array test -> know which data column are
            # not empty (returned as nan values)
            if len(data.shape) < 2 and len(lines) == skip_header + 1:
                # only 1 data row
                test = [np.isnan(v) for v in data]
                data = data.reshape((len(data), 1))
            else:
                if len(data.shape) < 2 and len(lines) > skip_header + 1:
                    # if only 1 data colum
                    data = np.array([range(len(data)), data])
                    for key in parsedAttr:
                        parsedAttr[key] = [parsedAttr[key][0]] + parsedAttr[key]
                test = [np.isnan(data[i, :]).all() for i in range(data.shape[0])]
            # if still cannot define colLabels
            if not graph.has_attr("collabels"):
                value_collabels = [""] * len(test)
                if len(data.shape) < 2:
                    value_collabels = [""] * 2
                graph.update({"collabels": value_collabels})
            # do not fill in 'filename', or 'label' default value if filename
            # is amongst the file parameters
            for key in parsedAttr:
                if key in attributes:
                    del attributes[key]
            if "label" not in parsedAttr and graph.has_attr("collabels"):
                parsedAttr["label"] = graph.attr("collabels")

            # parse through data
            if singlecurve:
                graph.append(Curve(data, attributes))
                cols.append(1)
            for colY in range(1, len(test)):  # normal column guessing
                # if nan -> column empty -> start new xy pair
                if test[colY]:
                    colX = colY + 1
                elif colY > colX:
                    # removing of nan pairs is performed in the Curve constructor
                    curvedata = np.append(data[colX, :], data[colY, :]).reshape(
                        (2, len(data[colY, :]))
                    )
                    graph.append(Curve(curvedata, attributes))
                    cols.append(colY)

            if len(cols) > 0:
                # correct parsedAttr to match data structure
                if len(graph.headers["collabels"]) <= max(cols):
                    graph.headers["collabels"].extend([""] * max(cols))
                graph.headers["collabels"] = [
                    graph.headers["collabels"][i] for i in cols
                ]
                # apply parsedAttr to different curves
                for key, value in parsedAttr.items():
                    if len(value) < max(cols):
                        parsedAttr[key].extend([""] * max(cols))
                    for i, coli in enumerate(cols):
                        try:
                            val = value[coli]
                        except IndexError:
                            continue
                        if val != "":
                            graph[i].update({key: val})
                # cast Curve in child class if required by parameter read
                for c, curve in enumerate(graph):
                    newtype = curve.attr("curve")
                    if newtype not in ["", "curve", "curvexy"]:
                        graph.castCurve(newtype, c, silentSuccess=True)

    @classmethod
    def readDataFromFileDatabase(cls, graph, attributes, fileContent=None, **_kwargs):
        """
        fileContent: list of lists containing file content (imagine csv file)
        kwargs: useless here
        """
        content = fileContent
        graph.update({"meastype": cls.GRAPHTYPE_DATABASE})
        if content is None:
            # this seems to parse the file satisfyingly
            try:
                with open(graph.filename, "r") as file:
                    content = [line.strip(":\r\n").split("\t") for line in file]
            except UnicodeError:
                msg = "Cannot open file {}."
                logger.error(msg.format(graph.filename), exc_info=True)
                return False

        # suppress empty lines
        for i in np.arange(len(content) - 1, -1, -1):
            if sum([len(content[i][j]) for j in range(len(content[i]))]) == 0:
                del content[i]
            else:
                for j in range(len(content[i])):
                    content[i][j] = content[i][j].replace("\\n", "\n")
        # max length of a line
        fileNcols = max([len(content[i]) for i in range(len(content))])

        # start processing - identify column names
        graph.update({"collabelsdetail": [], "rowlabels": [], "rowlabelnums": []})
        # check if further lines as header - must be pure text lines.
        # we store lines consisting only in a pair of elements. In this case
        # this should not be a column name, instead we will self.update()
        toUpdate = {}
        for i in range(0, len(content)):
            line = content[i]
            onlyText = True
            nVal = 0
            for val in line:
                if is_number(val):
                    onlyText = False
                    break
                if val != "":
                    nVal = nVal + 1
            if not onlyText:
                break

            if nVal == 1 and line[0] != "":
                split = line[0].split(": ")
                if len(split) > 1:
                    toUpdate.update({split[0]: split[1:]})
                else:
                    toUpdate.update({split[0]: ""})
            elif nVal == 2 and line[0] != "" and line[1] != "":
                toUpdate.update({line[0]: line[1]})
            else:
                graph.headers["collabelsdetail"].append(content[i][0:])
        # combine all column names identified so far
        nLinesHeaders = len(graph.headers["collabelsdetail"]) + len(toUpdate)
        labels = [""] * (fileNcols - 1)
        for i in range(len(graph.headers["collabelsdetail"])):
            for j in range(1, len(graph.headers["collabelsdetail"][i])):
                labels[j - 1] = (
                    labels[j - 1] + "\n" + graph.headers["collabelsdetail"][i][j]
                )
        for i in range(len(labels)):
            labels[i] = labels[i][1:].strip("\n ")
        graph.update({"collabels": labels[:]})  # want a hard-copy here

        # parsing content
        if len(graph.data) > 0:
            logger.warning(
                "readDataFromDatabaseFile, data container NOT empty. Content of data"
                " deleted, previous data lost."
            )
        graph.data = []
        # fill the Curve object with list of Curves, still empty at this stage
        for i in range(fileNcols - 1):
            graph.append(Curve(np.zeros((2, len(content) - nLinesHeaders)), attributes))
            graph[-1].update({"label": labels[i]})

        # update the header info
        # first change datatype for display purposes
        for key in toUpdate:
            if key in ["color"]:
                toUpdate[key] = strToVar(toUpdate[key])
        graph.update(toUpdate, if_all=True)

        # loop over rows
        for i in range(len(content) - nLinesHeaders):
            line = content[i + nLinesHeaders]
            graph.headers["rowlabels"].append(line[0])
            try:
                # self.headers['rowlabelnums'].append(float(''.join(ch for ch
                # in line[0] if ch.isdigit() or ch=='.')))
                split = refindall("([0-9]*[.]*[0-9]*)", line[0])
                split = [s for s in split if len(s) > 0]
                # print (line[0], split)
                if len(split) > 0:
                    graph.headers["rowlabelnums"].append(float(split[0]))
                else:
                    graph.headers["rowlabelnums"].append(np.nan)
            except Exception:
                graph.headers["rowlabelnums"].append(np.nan)
            for j in range(len(line) - 1):
                graph.data[j].setX(graph.headers["rowlabelnums"][-1], index=i)
                if is_number(line[j + 1]):
                    graph.data[j].setY(float(line[j + 1]), index=i)


def export_filesave_default(graph, filesave=""):
    """(private) default filename when saving."""
    if filesave == "":
        fileName = graph.filename.split("\\")[-1].split("/")[-1].replace(".", "_")
        filesave = fileName + "_"
        if len(graph) > 0:
            filesave += str(graph[-1].attr("complement"))
        if filesave == "_":
            filesave = "graphExport"
        if len(graph.data) > 1:
            filesave = filesave + "_series"
    return filesave


def export(
    graph,
    filesave: str = "",
    save_altered=False,
    if_template=False,
    if_compact=True,
    if_only_labels=False,
    if_clipboard_export=False,
) -> str:
    """
    Exports content of Grah object into a human- and machine-readable
    format.
    """
    if len(filesave) > 4 and filesave[-4:] == ".xml":
        return export_xml(graph, filesave=filesave, saveAltered=save_altered)

    # retrieve information of data alteration (plotting mode)
    alter = graph.get_alter()
    # handle saving of original data or altered data
    tempAttr = {}
    if save_altered:
        # if modified parameter, save modified data, and not save the alter attribute
        tempAttr.update({"alter": graph.attr("alter")})
        graph.update({"alter": ""})
    else:
        alter = ["", ""]  # want to save original data, and alter attribute
    # preparation
    out = ""
    # format general information for graph
    # template: do not wish any other information than graphInfo -> ok
    if not if_only_labels:
        # ifOnlyLabels: only export label argument
        for key, value in graph.graphinfo.items():
            out = out + key + "\t" + varToStr(value) + "\n"
        if not if_template:
            keyListHeaders = ["meastype"]
            for key in keyListHeaders:
                if key in graph.headers:
                    out = out + key + "\t" + varToStr(graph.headers[key]) + "\n"

    # format information for graph specific for each curve
    # first establish list of attributes, then loop over curves to construct export
    keys_list = ["label"]
    if not if_only_labels:  # if ifOnlyLabels: only export label argument
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
    curvedata = [[funcx(c, alter=alter[0]), funcy(c, alter=alter[1])] for c in graph]

    # if compact mode, loop over curves to identify consecutive with identical xaxis
    separator = ["\t"] + ["\t\t\t"] * (len(graph) - 1) + ["\t"]
    if if_compact and not if_template:
        for c in range(1, len(graph)):
            if np.array_equal(curvedata[c][0], curvedata[c - 1][0]):
                separator[c] = "\t"

    # proceed with file writing
    keys_list.sort()
    for key in keys_list:
        out = out + key
        for c, curve in enumerate(graph):
            value = curve.attr(key)
            if isinstance(value, np.ndarray):
                value = list(value)
            out = out + separator[c] + varToStr(value)
        out = out.rstrip("\t") + "\n"

    # format data
    data_len = 0
    if len(graph) > 0:
        data_len = np.max([len(data[0]) for data in curvedata])
    separator[0] = "\t\t\t"
    for i in range(data_len):
        for c, curve in enumerate(graph):
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
                        out += str(curvedata[c][0][i]) + "\t" + str(curvedata[c][1][i])
                    else:  # funcy(curve, index=i, alter=alter[1]))
                        out += str(curvedata[c][1][i])
                else:
                    if c == 0:
                        separator[0] = "\t"
                    out += separator[c]

        out += "\n"
        if if_template:
            break
    # restore initial state of Graph object
    graph.update(tempAttr)

    # end: return data, or save file
    if if_clipboard_export:
        return out

    # output
    filename = export_filesave_default(graph, filesave) + ".txt"
    if not graph.attr("savesilent", True):
        print("Graph data saved as", filename.replace("/", "\\"))
    graph.fileexport = os.path.normpath(filename)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            try:
                f.write(out)
            except UnicodeEncodeError as e:
                # give some details to the user, otherwise can become quite
                # difficult to identify where the problem originates from
                logger.error("export: could not save the file!", exc_info=True)
                ident = "in position "
                errmsg = str(e)
                if ident in errmsg:
                    idx = errmsg.find(ident) + len(ident)
                    chari = refindall("[0-9]+", errmsg[idx:])
                    if len(chari) > 0:
                        chari = int(chari[0])
                        outmsg = out[max(0, chari - 20) : min(len(out), chari + 15)]
                        msg = "--> surrounding characters:" + outmsg.replace(
                            "\n", "\\n"
                        )
                        logger.error(msg)
    except PermissionError:
        msg = "PermissionError in graphIO.export, filename {}."
        logger.error(msg.format(filename))  # maybe just print?
    return filename


def export_xml(graph, filesave: str = "", **kwargs) -> str:
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
        logger.warning(msg.format(key, value))
    try:
        from lxml import etree
    except ImportError:
        logger.error("export_xml ImportError", exc_info=True)
        return ""
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
