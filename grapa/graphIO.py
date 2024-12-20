# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:38:05 2017

@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import sys
import ast
import importlib
import numpy as np
import warnings

from re import split as resplit, findall as refindall, sub as resub
from copy import deepcopy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import is_number, strToVar, varToStr, strUnescapeIter
from grapa.curve_inset import Curve_Inset
from grapa.curve_subplot import Curve_Subplot
from grapa.curve_image import Curve_Image  # needed for curve casting
from grapa.graphIO_aux import (
    ParserAxhline,
    ParserAxvline,
    ParserMisc,
    GroupedPlotters,
    SubplotsAdjuster,
)


FILEIO_GRAPHTYPE_DATABASE = "Database"
FILEIO_GRAPHTYPE_GRAPH = "Graph"
FILEIO_GRAPHTYPE_UNDETERMINED = "Undetermined data type"


class GraphIO(Graph):
    """
    Subclass of Graph handling the file reading and writing.
    Want a separate file for a bad reason (Graph was too big). Thus the class
    is written assuming the self is not a Graph instance, but not a GraphIO.
    By default, 3 different reading modes are possible:
    - directly from data (filename=[[0,1,2,3],[5,2,7,4]])
    - column-organized datafile, with data starting at first row with a number
      in the first field
    - database, with no number in the first column
    File opening strategy:
        1/ readDataFile
            - do some stuff with complementary input
            - read first 3 lines of the file (readDataFileLine123)
            - identify child classes which (_listGraphChildClasses)
            - check if any of Graph child class can handle the file
            - if not, calls readDataFromFileTxt
        2/ readDataFromFileTxt
            - look if file has a number in the first column (_funcReadDataFile)
              if so, call readDataFromFileGeneric
              else call readDataFromFileDatabase
    """

    childClasses = None
    # list of child classes of Graph

    def dataFromVariable(self, data, attributes):
        """Initializes content of object using data given in constructor."""
        self.data.append(Curve(data, {}))
        self.update(attributes, ifAll=False)
        # nevertheless want to import all Curve types, if not already done
        # so Curve will know all its possible subtypes and can tell to GUI
        GraphIO._listGraphChildClasses()

    @classmethod
    def _listGraphChildClasses(cls):
        """
        Returns the child classes of Graph which have implemented file
        reading. Seeps through files with names starting with 'curve' or
        'graph', check the presence of required methods,
        and if so returns the Graph child class.
        """
        # do not reimport everything each time, only first time
        if cls.childClasses is not None:
            return cls.childClasses
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datatypes")
        required = ["FILEIO_GRAPHTYPE", "isFileReadable", "readDataFromFile"]
        subclasses = []
        for filestart in ["graph", "curve"]:
            for file in os.listdir(folder):
                fileName, fileExt = os.path.splitext(file)
                if (
                    fileExt == ".py"
                    and fileName[: len(filestart)] == filestart
                    and len(fileName) > len(filestart)
                ):
                    end = fileName[len(filestart) :]
                    module = importlib.import_module("grapa.datatypes." + fileName)
                    if not hasattr(module, "Graph" + end):
                        continue
                    typeM = getattr(module, "Graph" + end)
                    isValid = True
                    for attr in required:
                        if not hasattr(typeM, attr):
                            isValid = False
                            break
                    if isValid:
                        subclasses.append(typeM)
        Graph.childClasses = subclasses  # class constant
        return subclasses

    @classmethod
    def readDataFileLine123(cls, filename):
        """
        Returns the first 3 lines of a file.
        """
        line1, line2, line3 = "", "", ""
        try:
            f = open(filename, "r")
        except Exception:  # if file does not exist
            # do NOT want to append Curve object if file empty
            return line1, line2, line3
        # if file exist, try to open first line
        try:
            line1 = f.readline().rstrip(" \r\n\t").replace("ï»¿", "")
            try:
                line2 = f.readline().rstrip(" \r\n\t").replace("ï»¿", "")
                try:
                    line3 = f.readline().rstrip(" \r\n\t").replace("ï»¿", "")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass
        f.close()
        return line1, line2, line3

    def readDataFile(self, complement=""):
        """
        Reads a file.
        Complement: ...
        Opens file self.filename
        """
        self.isFileContent = False
        if (
            isinstance(complement, dict)
            and "isfilecontent" in complement
            and complement["isfilecontent"]
        ):
            self.isFileContent = True
        # reads the file to identify types
        if not self.isFileContent:
            fileName, fileExt = os.path.splitext(self.filename)
            fileExt = fileExt.lower()
        else:
            fileName, fileExt = "", ".txt"
        fileNameBase = fileName.split("/")[-1].split("\\")[-1]
        attributes = {"filename": self.filename}
        if complement != "":
            if isinstance(complement, dict):
                for key in complement:
                    attributes.update({key.lower(): complement[key]})
            else:
                attributes.update({"complement": complement})
        # default attributes of curve
        if "label" not in attributes:
            labeldefault = fileName.split("/")[-1].split("\\")[-1]
            labeldefault = labeldefault.replace("_", " ").replace("  ", " ")
            attributes.update({"label": labeldefault})
        # default (mandatory) attributes of Graph
        for key in ["subplots_adjust", "fontsize"]:
            if key in attributes:
                self.update({key: attributes[key]})
                del attributes[key]
            else:
                self.update({key: Graph.DEFAULT[key]})
        # guess keys of complement which are for Graph and not for Curves
        toDel = []
        for key in complement:
            try:
                if key.lower() in Graph.KEYWORDS_GRAPH["keys"]:
                    self.update({key: complement[key]})
                    toDel.append(key)
            except Exception:
                pass
        for key in toDel:
            del complement[key]
        # read first line to identify datatype
        line1, line2, line3 = "", "", ""
        if not self.isFileContent:
            if not os.path.exists(self.filename):
                if not self.silent:
                    print("Class Graph: file do not exist!", self.filename)
                return False
            line1, line2, line3 = GraphIO.readDataFileLine123(self.filename)
        else:
            self.filename = self.filename.replace("\r", "\n")
            line1 = self.filename.split("\n")[0]
            if "isfilecontent" in attributes:
                del attributes["isfilecontent"]
            if attributes["label"] == "":
                del attributes["label"]
        msg = ""
        # (uncomplete) list of classes handled:
        # with corresponding Curve types: CV, Cf, JV, TIV, MBE, MCA, SIMS
        # without: MCA fit (html output of XRF), MBElog
        loaded = False
        # first check if instructed to open the file in a certain way
        if "readas" in attributes and attributes["readas"].lower() in [
            "database",
            "generic",
        ]:
            msg = "opened using standard methods."
            GraphIO.readDataFromFileTxt(self, attributes)
            loaded = True
        # if not, then try every possibility
        if not loaded:
            childClasses = GraphIO._listGraphChildClasses()
            #        print(childClasses)
            for child in childClasses:
                if child.isFileReadable(
                    fileNameBase, fileExt, line1=line1, line2=line2, line3=line3
                ):
                    msg = "opened as " + child.FILEIO_GRAPHTYPE + "."
                    self.headers.update({"meastype": child.FILEIO_GRAPHTYPE})
                    res = child.readDataFromFile(
                        self,
                        attributes,
                        fileName=fileNameBase,
                        line1=line1,
                        line2=line2,
                        line3=line3,
                    )
                    if res is None or not res != False:
                        loaded = True
                        break
        # if not, then try default methods
        if not loaded:
            # other not readable formats
            if fileExt in [".pdf"]:
                msg = "do not know how to open."
            else:
                # keep the default opening mechanism in the main class
                msg = "opened using standard methods."
                GraphIO.readDataFromFileTxt(self, attributes)
        if msg != "":
            if not self.silent:
                print("File " + fileName.split("/")[-1] + "... " + msg)

    def _funcReadDataFile(self, filecontent, attributes=None):
        """Try to guess which method is best suited to open the file"""
        if attributes is None:
            attributes = {}
        if "readas" in attributes:
            if attributes["readas"].lower() == "database":
                return GraphIO.readDataFromFileDatabase
            if attributes["readas"].lower() == "generic":
                return GraphIO.readDataFromFileGeneric
        # test to identify if file is organized as series of
        # headers + x-y columns, or as database
        for i in range(len(filecontent)):
            try:
                float(filecontent[i][0])
                # return fileGeneric only if the first element of any row can be
                # interpreted as a number, ie. float(element) do not raise a
                # ValueError
                return GraphIO.readDataFromFileGeneric
            except ValueError:
                pass
        return GraphIO.readDataFromFileDatabase

    def readDataFromFileTxt(self, attributes):
        """
        Reads content of a .txt file, and parse it as a database or as a
        generic file.
        """
        if not self.isFileContent:
            try:
                fileContent = [
                    resplit("[\t ,;]", line.strip(":\r\n"))
                    for line in open(self.filename, "r")
                ]
            except UnicodeDecodeError as e:
                print("Exception", type(e), "in GraphIO.readDataFromFileTxt:")
                print(e)
                return False
            except FileNotFoundError:
                return False
        else:
            fileContent = self.filename.split("\n")
            delimiter = GraphIO._readDataIdentifyDelimiter(self, fileContent)
            fileContent = [resplit(delimiter, line) for line in fileContent]
            while len(fileContent) > 0 and fileContent[-1] == [""]:
                del fileContent[-1]
        f = GraphIO._funcReadDataFile(self, fileContent, attributes)
        if not self.isFileContent:
            # intentionally do not provide file content, function
            # will read file again with the desired data processing
            f(self, attributes)
        else:
            self.filename = ".txt"  # free some memory
            attributes.update({"filename": "From clipboard"})
            f(self, attributes, fileContent=fileContent)
            self.isFileContent = False

    def _readDataIdentifyDelimiter(self, lines):
        # determine data character separator - only useful if reads file,
        # otherwise is supposed to be already separated
        delimiter = "\t"
        nTab = np.sum([line.count("\t") for line in lines])
        threshold = np.ceil(max(1, len(lines) - 5) / 2)
        if nTab < threshold:
            # if really not enough tabs in the file to be reasonable that data
            # are tab-separated
            delimiters = [",", ";", " "]
            for s in delimiters:
                n = np.sum([line.count(s) for line in lines])
                if n > threshold:
                    delimiter = s
                    print(
                        'Graph file generic: data delimiter identified as "' + s + '".'
                    )
                    break
        return delimiter

    def readDataFromFileGeneric(
        self, attributes, fileContent=None, ifReplaceCommaByPoint=False, **kwargs
    ):
        """
        Reads the file as a column-organized data file, with some headers
        lines at the beginning.
        kwargs: are handled the following keywords:
        - delimiter: if provided, assumed the values is the text delimiter.
        - delimiterHeaders: same as delimiter, but only for the headers section
        - lstrip: a lstrip function is applied to each line when parsing they
          file. Values: True, or '# ' or similar
        """
        # idea:
        # 1/ read headers, store values in headers or in sampleInfo.
        # 2/ read data, identify data structure
        # 3/ look back at headers to fill in dataInfo according to data
        #    structure
        # if fileContent provided, file is not read and fileContent is used
        # instead. Supposed to be a 2-dimensional list [line][col]

        # parsing inputs
        lstriparg = None
        if "lstrip" in kwargs:
            if kwargs["lstrip"] == True:
                lstriparg = []
            elif isinstance(kwargs["lstrip"], str):
                lstriparg = [kwargs["lstrip"]]
        # update default information
        self.data = []
        parsedAttr = {}
        fileName, fileext = os.path.splitext(self.filename)  # , fileExt=
        self.headers.update({"meastype": FILEIO_GRAPHTYPE_UNDETERMINED})
        singlecurve = False
        if "_singlecurve" in attributes and attributes["_singlecurve"]:
            singlecurve = True
            del attributes["_singlecurve"]

        skipLines = 0
        skipFooters = 0
        lastSampleInfo = ""  # key of last saved parameter
        lastLine = ""
        colLabels = []
        # load content of file
        if fileContent is None:  # default behavior
            lines = [line.rstrip(":\r\n\t") for line in open(self.filename, "r")]
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
            delimiter = GraphIO._readDataIdentifyDelimiter(self, lines)
        delimiterHeaders = (
            None if "delimiterHeaders" not in kwargs else kwargs["delimiterHeaders"]
        )

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
                        not in self.KEYWORDS_GRAPH["keys"]
                        + self.KEYWORDS_HEADERS["keys"]
                        + self.KEYWORDS_CURVE["keys"]
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
                    # colLabels
                    colLabels = [suff + str(l).replace('"', "") for l in lastLine]
                    if "label" in attributes:
                        del attributes["label"]
                # don't save last couple, intepreted as column name
                break
            # if not numerical value -> still information
            else:
                skipLines += 1
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
                                val = [x / self.FILEIO_DPI for x in val]
                    except Exception:
                        print(
                            "GraphIO.readDataFromFileGeneric: Error when",
                            "setting figure size",
                        )
                replace = {"facealpha": "alpha", "legendlocation": "legendproperties"}
                if couple[0] in replace:
                    couple[0] = replace[couple[0]]
                # end of matlab conversion
                # identify headers values
                if couple[0] in self.KEYWORDS_HEADERS["keys"]:
                    self.headers[couple[0]] = strToVar(val)
                # identify graphInfo values
                elif couple[0] in self.KEYWORDS_GRAPH["keys"] or couple[0].startswith(
                    "subplots"
                ):
                    self.graphInfo[couple[0]] = strToVar(val)
                # send everything else in Curve, leave nothing for sampleInfo
                else:
                    # do not use val
                    rep = {"legend": "label", "â²": "2", "xycurve": "curve"}
                    for key in rep:
                        couple[0] = couple[0].replace(key, rep[key])
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
        self.checkValidText()
        # print('GraphIO parsedAttr', parsedAttr)
        # if must parse last "header" line nevertheless (first element blank)
        if singlecurve and len(lastLine) > 1 and lastLine[0].startswith("empty"):
            linei = skipLines - 1
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
                    skipLines = max(skipLines - 1, 0)

        # if last text line was not empty -> interpret that as x-y-labels
        if colLabels != []:
            self.headers["collabels"] = colLabels
        if "collabels" not in self.headers:
            if "label" in parsedAttr:
                self.headers["collabels"] = parsedAttr["label"]
        if colLabels != []:
            if "xlabel" not in self.graphInfo:
                self.graphInfo["xlabel"] = colLabels[0]
            if "ylabel" not in self.graphInfo and len(colLabels) > 1:
                self.graphInfo["ylabel"] = colLabels[1]

        while lines[-1] == "":
            lines.pop()
            skipFooters += 1
        # do not understand it... genfromtxt can fail when files ends with
        # several \n and this is not set!?
        skipFooters = 0

        # look for data in file after headers
        if skipLines < len(lines):
            # default behavior
            if fileContent is None:
                usecols = range(0, len(lines[skipLines].split(delimiter)))
                dictConverters = {}
                if ifReplaceCommaByPoint:
                    lambd = lambda x: float(str(str(x, "UTF-8")).replace(",", "."))
                    for i in usecols:
                        dictConverters.update({i: lambd})
                genFromTxtOpt = {}
                if len(dictConverters) > 0:
                    genFromTxtOpt.update({"converters": dictConverters})
                data = np.transpose(
                    np.genfromtxt(
                        self.filename,
                        skip_header=skipLines,
                        delimiter=delimiter,
                        invalid_raise=False,
                        usecols=usecols,
                        skip_footer=skipFooters,
                        **genFromTxtOpt
                    )
                )
            else:
                # if some content was provided
                for i in range(skipLines, len(fileContent) - skipFooters):
                    for j in range(len(fileContent[i])):
                        fileContent[i][j] = (
                            np.float64(fileContent[i][j])
                            if fileContent[i][j] != "" and is_number(fileContent[i][j])
                            else np.nan
                        )
                data = np.transpose(np.array(fileContent[skipLines:]))
            colX = 0
            cols = []

            # some checks, and build array test -> know which data column are
            # not empty (returned as nan values)
            if len(data.shape) < 2 and len(lines) == skipLines + 1:
                # only 1 data row
                test = [np.isnan(v) for v in data]
                data = data.reshape((len(data), 1))
            else:
                if len(data.shape) < 2 and len(lines) > skipLines + 1:
                    # if only 1 data colum
                    data = np.array([range(len(data)), data])
                    for key in parsedAttr:
                        parsedAttr[key] = [parsedAttr[key][0]] + parsedAttr[key]
                test = [np.isnan(data[i, :]).all() for i in range(data.shape[0])]
            # if still cannot define colLabels
            if "collabels" not in self.headers:
                self.headers["collabels"] = [""] * len(test)
                if len(data.shape) < 2:
                    self.headers["collabels"] = [""] * 2
            # do not fill in 'filename', or 'label' default value if filename
            # is amongst the file parameters
            for key in parsedAttr:
                if key in attributes:
                    del attributes[key]
            if "label" not in parsedAttr and "collabels" in self.headers:
                parsedAttr["label"] = self.headers["collabels"]

            # parse through data
            if singlecurve:
                self.data.append(Curve(data, attributes))
                cols.append(1)
            for colY in range(1, len(test)):  # normal column guessing
                # if nan -> column empty -> start new xy pair
                if test[colY]:
                    colX = colY + 1
                elif colY > colX:
                    # removing of nan pairs is performed in the Curve constructor
                    self.data.append(
                        Curve(
                            np.append(data[colX, :], data[colY, :]).reshape(
                                (2, len(data[colY, :]))
                            ),
                            attributes,
                        )
                    )
                    cols.append(colY)
            if len(cols) > 0:
                # correct parsedAttr to match data structure
                if len(self.headers["collabels"]) <= max(cols):
                    # print('graphio file', self.filename, 'extend collables',
                    #       len(self.headers['collabels']),
                    #       type(self.headers['collabels']),
                    #       self.headers['collabels'])
                    self.headers["collabels"].extend([""] * max(cols))
                self.headers["collabels"] = [self.headers["collabels"][i] for i in cols]
                # apply parsedAttr to different curves
                for key in parsedAttr:
                    if len(parsedAttr[key]) < max(cols):
                        parsedAttr[key].extend([""] * max(cols))
                    for i in range(len(cols)):
                        try:
                            val = parsedAttr[key][cols[i]]
                        except IndexError:
                            continue
                        if val != "":
                            self.curve(i).update({key: val})
                # cast Curve in child class if required by parameter read
                for c in range(self.length()):
                    newtype = self.curve(c).attr("curve")
                    if newtype not in ["", "curve", "curvexy"]:
                        self.castCurve(newtype, c, silentSuccess=True)

    def readDataFromFileDatabase(self, attributes, fileContent=None, **_kwargs):
        """
        fileContent: list of lists containing file content (imagine csv file)
        kwargs: useless here
        """
        self.headers.update({"meastype": FILEIO_GRAPHTYPE_DATABASE})
        if fileContent is None:
            # this seems to parse the file satisfyingly
            fileContent = [
                line.strip(":\r\n").split("\t") for line in open(self.filename, "r")
            ]
        # suppress empty lines
        for i in np.arange(len(fileContent) - 1, -1, -1):
            if sum([len(fileContent[i][j]) for j in range(len(fileContent[i]))]) == 0:
                del fileContent[i]
            else:
                for j in range(len(fileContent[i])):
                    fileContent[i][j] = fileContent[i][j].replace("\\n", "\n")
        # max length of a line
        fileNcols = max([len(fileContent[i]) for i in range(len(fileContent))])

        # start processing - identify column names
        self.headers.update(
            {"collabelsdetail": [], "rowlabels": [], "rowlabelnums": []}
        )
        # check if further lines as header - must be pure text lines.
        # we store lines consisting only in a pair of elements. In this case
        # this should not be a column name, instead we will self.update()
        toUpdate = {}
        for i in range(0, len(fileContent)):
            line = fileContent[i]
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
                self.headers["collabelsdetail"].append(fileContent[i][0:])
        # combine all column names identified so far
        nLinesHeaders = len(self.headers["collabelsdetail"]) + len(toUpdate)
        labels = [""] * (fileNcols - 1)
        for i in range(len(self.headers["collabelsdetail"])):
            for j in range(1, len(self.headers["collabelsdetail"][i])):
                labels[j - 1] = (
                    labels[j - 1] + "\n" + self.headers["collabelsdetail"][i][j]
                )
        for i in range(len(labels)):
            labels[i] = labels[i][1:].strip("\n ")
        self.headers.update({"collabels": labels[:]})  # want a hard-copy here

        # parsing content
        if len(self.data) > 0:
            print(
                "Class GraphIO.readDataFromDatabaseFile, data container NOT"
                "empty. Content of data deleted, previous data lost."
            )
        self.data = []
        # fill the Curve object with list of Curves, still empty at this stage
        for i in range(fileNcols - 1):
            self.data.append(
                Curve(np.zeros((2, len(fileContent) - nLinesHeaders)), attributes)
            )
            self.data[-1].update({"label": labels[i]})

        # update the header info
        # first change datatype for display purposes
        for key in toUpdate:
            if key in ["color"]:
                toUpdate[key] = strToVar(toUpdate[key])
        self.update(toUpdate, ifAll=True)

        # loop over rows
        for i in range(len(fileContent) - nLinesHeaders):
            line = fileContent[i + nLinesHeaders]
            self.headers["rowlabels"].append(line[0])
            try:
                # self.headers['rowlabelnums'].append(float(''.join(ch for ch
                # in line[0] if ch.isdigit() or ch=='.')))
                split = refindall("([0-9]*[.]*[0-9]*)", line[0])
                split = [s for s in split if len(s) > 0]
                # print (line[0], split)
                if len(split) > 0:
                    self.headers["rowlabelnums"].append(float(split[0]))
                else:
                    self.headers["rowlabelnums"].append(np.nan)
            except Exception:
                self.headers["rowlabelnums"].append(np.nan)
            for j in range(len(line) - 1):
                self.data[j].setX(self.headers["rowlabelnums"][-1], index=i)
                if is_number(line[j + 1]):
                    self.data[j].setY(float(line[j + 1]), index=i)

    def filesave_default(self, filesave=""):
        """(private) default filename when saving."""
        if filesave == "":
            fileName = self.filename.split("\\")[-1].split("/")[-1].replace(".", "_")
            filesave = fileName + "_"
            if self.length() > 0:
                filesave += self.curve(-1).attr("complement")
            if filesave == "_":
                filesave = "graphExport"
            if len(self.data) > 1:
                filesave = filesave + "_series"
        return filesave

    def export(
        self,
        filesave="",
        saveAltered=False,
        ifTemplate=False,
        ifCompact=True,
        ifOnlyLabels=False,
        ifClipboardExport=False,
    ):
        """
        Exports content of Grah object into a human- and machine-readable
        format.
        """
        if len(filesave) > 4 and filesave[-4:] == ".xml":
            return GraphIO.exportXML(self, filesave=filesave, saveAltered=saveAltered)
        # retrieve information of data alteration (plotting mode)
        alter = self.attr("alter")
        if alter == "":
            alter = ["", ""]
        if isinstance(alter, str):
            alter = ["", alter]
        # handle saving of original data or altered data
        tempAttr = {}
        if saveAltered:
            # if modified parameter, want to save modified data, and not save
            # the alter attribute
            tempAttr.update({"alter": self.attr("alter")})
            self.update({"alter": ""})
        else:
            alter = ["", ""]  # want to save original data, and alter atribute
        # preparation
        dataLe1 = 0
        out = ""
        # format general information for graph
        # template: do not wish any other information than graphInfo -> ok
        if not ifOnlyLabels:
            # ifOnlyLabels: only export label argument
            for key in self.graphInfo:
                out = out + key + "\t" + varToStr(self.graphInfo[key]) + "\n"
            if not ifTemplate:
                keyListHeaders = ["meastype"]
                for key in keyListHeaders:
                    if key in self.headers:
                        out = out + key + "\t" + varToStr(self.headers[key]) + "\n"
        # if compact mode, loop over curves to identify consecutive with identical xaxis
        separator = ["\t"] + ["\t\t\t"] * (self.length() - 1) + ["\t"]
        if ifCompact and not ifTemplate:
            for i in range(self.length()):
                if i > 0 and np.array_equal(self[i].x(), self[i - 1].x()):
                    separator[i] = "\t"
        # format information for graph specific for each curve
        # first establish list of attributes, then loop over curves to
        # construct export
        keysList = ["label"]
        if not ifOnlyLabels:  # if ifOnlyLabels: only export label argument
            for c in range(self.length()):
                for attr in self.curve(c).getAttributes():
                    if attr not in keysList:
                        if attr in self.KEYWORDS_CURVE["keys"] or (not ifTemplate):
                            keysList.append(attr)
        # if save altered Curce export curves modified for alterations and
        # offsets, and delete offset and muloffset key
        if saveAltered:
            funcx, funcy = Curve.x_offsets, Curve.y_offsets
            if "offset" in keysList:
                keysList.remove("offset")
            if "muloffset" in keysList:
                keysList.remove("muloffset")
        else:
            funcx, funcy = Curve.x, Curve.y
        # proceed with file writing
        keysList.sort()
        for key in keysList:
            out = out + key
            for c in range(len(self)):
                value = self[c].attr(key)
                if isinstance(value, np.ndarray):
                    value = list(value)
                out = out + separator[c] + varToStr(value)
            out = out.rstrip("\t") + "\n"
        # format data
        for curve in self.data:
            if curve.shape(1) > dataLe1:
                dataLe1 = curve.shape(1)
        separator[0] = "\t\t\t"
        for i in range(dataLe1):
            for c in range(self.length()):
                if ifTemplate:
                    out += "1" + "\t" + "0" + "\t\t"
                else:
                    curve = self.curve(c)
                    if i < curve.shape(1):
                        if c > 0:
                            out += "\t"
                            if len(separator[c]) == 3:
                                out += "\t"
                        if len(separator[c]) == 3:
                            out += (
                                str(funcx(curve, index=i, alter=alter[0]))
                                + "\t"
                                + str(funcy(curve, index=i, alter=alter[1]))
                            )
                        else:
                            out += str(funcy(curve, index=i, alter=alter[1]))
                    else:
                        if c == 0:
                            separator[0] = "\t"
                        out += separator[c]

            out += "\n"
            if ifTemplate:
                break
        # restore initial state of Graph object
        self.update(tempAttr)
        # end: return data, or save file
        if ifClipboardExport:
            return out
        else:
            # output
            filename = GraphIO.filesave_default(self, filesave) + ".txt"
            if self.attr("saveSilent") != True:
                print("Graph data saved as", filename.replace("/", "\\"))
            self.fileexport = os.path.normpath(filename)
            f = open(filename, "w")
            try:
                f.write(out)
            except UnicodeEncodeError as e:
                # give some details to the user, otherwise can become quite
                # difficult to identify where the progrem originates from
                print("ERROR! Could not save the file!")
                print(
                    "Exception", type(e), "when exporting the graph.", "Exception:", e
                )
                ident = "in position "
                errmsg = e.__str__()
                if ident in errmsg:
                    idx = errmsg.find(ident) + len(ident)
                    from re import findall

                    chari = findall(("[0-9]+"), errmsg[idx:])
                    if len(chari) > 0:
                        chari = int(chari[0])
                        outmsg = out[max(0, chari - 20) : min(len(out), chari + 15)]
                        print(
                            "--> surrounding characters:", outmsg.replace("\n", "\\n")
                        )
            f.close()
        return filename

    def exportXML(self, filesave="", **kwargs):
        """
        Exports Graph as .xml file
        NOT implemented:
            saveAltered=False, ifTemplate=False
        Non-sense here:
            ifCompact=True, ifClipboardExport=False
        """
        for key in kwargs:
            msg = (
                "WARNING GraphIO.exportXML: received keyword {}: {}, don't know "
                "what to do with it."
            )
            print(msg.format(key, kwargs[key]))
        try:
            from lxml import etree
        except ImportError as e:
            print("Exception", type(e), "in GraphIO.graphToXML:")
            print(e)
            return
        root = etree.Element("graph")
        # write headers
        for container in ["headers", "sampleInfo", "graphInfo"]:
            if hasattr(self, container) and len(getattr(self, container)):
                tmp = etree.SubElement(root, container)
                for key in getattr(self, container):
                    tmp.set(key, str(getattr(self, container)[key]))
        # write curves
        for c in range(self.length()):
            curve = self.curve(c)
            tmp = etree.SubElement(root, "curve" + str(c))
            attr = etree.SubElement(tmp, "attributes")
            for key in curve.getAttributes():
                attr.set(key, str(curve.attr(key)))
            data = etree.SubElement(tmp, "data")
            x = etree.SubElement(data, "x")
            y = etree.SubElement(data, "y")
            x.text = ",".join([str(e) for e in curve.x()])
            y.text = ",".join([str(e) for e in curve.y()])
        # save
        filesave = GraphIO.filesave_default(self, filesave)
        if len(filesave) < 4 or filesave[-4:] != ".xml":
            filesave += ".xml"
        tree = etree.ElementTree(root)
        tree.write(filesave, pretty_print=True, xml_declaration=True, encoding="utf-8")
        if self.attr("saveSilent") != True:
            print("Graph data saved as", filesave.replace("/", "\\"))
        return filesave

    def _createSubplotsGridspec(self, figAx=None, ifSubPlot=False):
        """
        Generates a Figure (if not provided) with a gridpec axes matrix
        """
        # TODO: Handle subplotskwargs: sharex True, 'col'
        import matplotlib.gridspec as gridspec

        # set fig as the active figure, if provided. Else with current figure
        if figAx is not None:
            fig, ax = figAx
            plt.figure(fig.number)  # bring existing figure to front
            # delete all except the provided axis
            if not ifSubPlot:
                for ax_ in fig.get_axes():
                    if ax_ is not ax:
                        plt.delaxes(ax_)
        else:
            fig = plt.figure()  # create a new figure
            ax = None
        # count number of graphs to be plotted
        axes = [{"ax": None, "activenext": True} for _curve in self]
        ncurvesondefault = 0  # do not create default axis if no curve on it
        isthereasubplot = False
        ngraphs = 0
        for c in range(len(self)):
            curve = self[c]
            if isinstance(curve, Curve_Inset):
                axes[c].update({"ax": "inset", "activenext": False})
                # default can be overridden if Graph file has length 0
            elif isinstance(curve, Curve_Subplot) and not curve.isHidden():
                rowspan = int(curve.attr("subplotrowspan", 1))
                colspan = int(curve.attr("subplotcolspan", 1))
                ngraphs += rowspan * colspan
                isthereasubplot = True
            elif ngraphs == 0:
                ncurvesondefault += 1
                ngraphs = 1
        ngraphs = max(1, ngraphs)
        # transpose?
        transpose = self.attr("subplotstranspose", False)
        # determine axes matrix shape
        ncols = int(self.attr("subplotsncols", (1 if ngraphs < 2 else 2)))
        nrows = int(np.ceil(ngraphs / ncols))
        # width, heigth ratios?
        gridspeckwargs = {}
        val = list(self.attr("subplotswidth_ratios", ""))
        if len(val) > 0:
            target = ncols if not transpose else nrows
            if len(val) != target:
                val += [1] * max(0, (target) - len(val))
                while len(val) > target:
                    del val[-1]
                msg = (
                    "GraphIO._createSubplotsGridspec: corrected width_ratios to "
                    "match ncols {}: {}."
                )
                print(msg.format(ncols, val))
            gridspeckwargs.update({"width_ratios": val})
        val = list(self.attr("subplotsheight_ratios", ""))
        if len(val) > 0:
            target = nrows if not transpose else ncols
            if len(val) != target:
                val += [1] * max(0, (target) - len(val))
                while len(val) > target:
                    del val[-1]
                msg = (
                    "GraphIO._createSubplotsGridspec: corrected height_ratios to "
                    "match nrows {}: {}."
                )
                print(msg.format(nrows, val))
            gridspeckwargs.update({"height_ratios": val})
        # generate axes matrix: either gs, either ax is created
        gs, matrix = None, np.array([[]])
        if ax is None:
            if ngraphs == 1 and not isthereasubplot:
                ax = fig.add_subplot(111)
                ax.ticklabel_format(useOffset=False)
                ax.patch.set_alpha(0.0)
                # hide axis if no Curve will be plotted on it
                if ncurvesondefault == 0:
                    ax.axis("off")
            else:
                matrix = np.ones((nrows, ncols)) * (-1)
                if transpose:
                    gs = gridspec.GridSpec(ncols, nrows, **gridspeckwargs)
                else:
                    gs = gridspec.GridSpec(nrows, ncols, **gridspeckwargs)

        # coordinates of the plot id
        subplotsid = self.attr("subplotsid", False)
        if subplotsid is not False and (
            not isinstance(subplotsid, (list, tuple)) or len(subplotsid) != 2
        ):
            subplotsid = (-0.03, 0.00)
        # misc adjustments to self
        if ngraphs > 1:  # do not want default values if multiple subplots
            self.update({"xlabel": "", "ylabel": ""})
        # return
        return [fig, ax], gs, matrix, subplotsid, ngraphs

    def _newaxis(
        self, curve, fig, gs, matrix, subplotsid, subplotscounter, subplotidkwargs
    ):
        rowspan = int(curve.attr("subplotrowspan", 1))
        colspan = int(curve.attr("subplotcolspan", 1))
        gspos = gs.get_grid_positions(fig)
        flag = False
        ax = None
        for i_ in range(len(matrix)):
            for j_ in range(len(matrix[i_])):
                if matrix[i_, j_] == -1:
                    # first free spot found: start there
                    if rowspan > matrix.shape[0] - i_:
                        # calculation of number of rows probably faulty
                        msg = (
                            "WARNING GraphIO.plot: rowspan larger than can handled "
                            "for subplot {}, value coerced to {}."
                        )
                        print(msg.format(subplotscounter, matrix.shape[0] - i_))
                    if colspan > matrix.shape[1] - j_:
                        # either faulty calculation of number of rows,
                        # either that colspan cannot fit in ncols
                        msg = (
                            "WARNING GraphIO.plot: colspan larger than can handled "
                            "for subplot {}, value coerced to {}."
                        )
                        print(msg.format(subplotscounter, matrix.shape[1] - j_))
                    rowspan = min(rowspan, matrix.shape[0] - i_)
                    colspan = min(colspan, matrix.shape[1] - j_)
                    if self.attr("subplotstranspose", False):
                        if subplotsid:
                            coords = (
                                gspos[2][i_] + subplotsid[0],
                                gspos[1][j_] + subplotsid[1],
                            )
                            txt = "(" + chr(ord("a") + subplotscounter) + ")"
                            self.addText(txt, coords, subplotidkwargs)
                        # ax = plt.subplot(gs[j_ : (j_ + colspan), i_ : (i_ + rowspan)])
                        ax = fig.add_subplot(
                            gs[j_ : (j_ + colspan), i_ : (i_ + rowspan)]
                        )
                    else:
                        if subplotsid:
                            coords = (
                                gspos[2][j_] + subplotsid[0],
                                gspos[1][i_] + subplotsid[1],
                            )
                            txt = "(" + chr(ord("a") + subplotscounter) + ")"
                            self.addText(txt, coords, subplotidkwargs)
                        # ax = plt.subplot(gs[i_ : (i_ + rowspan), j_ : (j_ + colspan)])
                        ax = fig.add_subplot(
                            gs[i_ : (i_ + rowspan), j_ : (j_ + colspan)]
                        )
                    matrix[i_ : (i_ + rowspan), j_ : (j_ + colspan)] = subplotscounter
                    flag = True
                    break
            if flag:
                break
        if ax is None:
            print("ERROR GraphIO.plot create subplot: ax is None")
            print("matrix", matrix)
            raise Exception
        ax.ticklabel_format(useOffset=False)
        ax.patch.set_alpha(0.0)
        return ax

    # create graph from data with headers additional data
    def plot(
        self,
        filesave="",
        imgFormat="",
        figsize=(0, 0),
        ifSave=True,
        ifExport=True,
        figAx=None,
        ifSubPlot=False,
        handles=None,
    ):
        """
        Plot the content of the object.
        imgFormat: by default image will be .png. Possible format are the ones
            accepted by plt.savefig
        figsize: default figure size is a class consant
        filesave: (optional) filename for the saved graph
        ifExport: if True, reate a human- and machine-readable .txt file
            containing all information of the graph
        figAx= [fig, ax] provided. Useful when wish to embedd graph in a GUI.
        ifSubPlot: False, unless fig to be drawn in a subplot. Not handled by
            the GUI, but might be useful in scripts. Prevents deletion of
            existing axes.
        """
        self._PLOT_PRINTEDERROR_ATTRIBUTE = False
        self._PLOT_PRINTEDERROR_VALUE = False

        def sca_errorhandled(ax, txt=""):
            try:
                plt.sca(ax)
                # actually, show avoid using pyplot together with tkagg !!!
            except AttributeError:
                if not self._PLOT_PRINTEDERROR_ATTRIBUTE:
                    # print('WARNING sca(ax)', txt, '. AttributeError catched, cause to investigate...')
                    self._PLOT_PRINTEDERROR_ATTRIBUTE = True
            except ValueError:
                if not self._PLOT_PRINTEDERROR_VALUE:
                    # print('WARNING sca(ax)', txt, '. ValueError catched, cause to investigate...')
                    self._PLOT_PRINTEDERROR_VALUE = True

        # treat filesave, and store info if not already done. required for
        # relative path to subplots or insets
        filename = GraphIO.filesave_default(self, filesave)
        if not hasattr(self, "filename"):
            self.filename = filename

        # store some attributes which might be modified upon execution
        restore = {}
        for attr in ["text", "textxy", "textargs"]:
            restore.update({attr: self.attr(attr)})

        # retrieve default axis positions subplotAdjustDef
        subplotAdjustDef = SubplotsAdjuster.default()
        subplotColorbar = [
            0.90,
            subplotAdjustDef["bottom"],
            0.05,
            subplotAdjustDef["top"] - subplotAdjustDef["bottom"],
        ]
        # shift fdefault right if there is a colorbar defined in a Curve
        for c in self:
            if c.attr("colorbar", None) is not None:
                subplotAdjustDef["right"] -= 0.1
                break

        # other data we want to retrieve now
        fontsize = Graph.DEFAULT["fontsize"]
        if "fontsize" in self.graphInfo:
            fontsize = self.graphInfo["fontsize"]
        alter = self._getAlter()
        if not isinstance(alter, list):
            # CHECK WITH PRINTOUT. may lead to bug with data copy
            print("SOMETHING FISHY HERE graphIO alter", alter, type(alter))
        ignoreXLim = True if alter[0] != "" else False
        ignoreYLim = True if alter[1] != "" else False

        axTwinX, axTwinY, axTwinXY = None, None, None

        # check
        if len(self.data) <= 0:
            if ifSubPlot:
                pass
            else:
                print("Warning GraphIO.plot", self.filename, ": no data to plot!")
                if figAx is not None:
                    fig, ax = figAx[0], figAx[1]
                    if ax is not None:
                        sca_errorhandled(ax)
                    plt.cla()
                # return 1

        # retrieve figure size
        if figsize == (0, 0):
            # if figure size not imposed at function call, look for
            # instructions in measurement data
            figsize = self.FIGSIZE_DEFAULT
            if "figsize" in self.graphInfo:
                figsize = self.graphInfo["figsize"]

        # create graph, initialize axes
        subplotscounter = 0
        (
            [fig, ax],
            gs,
            matrix,
            subplotsid,
            subplotsngraphs,
        ) = GraphIO._createSubplotsGridspec(self, figAx, ifSubPlot=ifSubPlot)

        # HOPEFULLY SOLVED BUG - grapa fails to plot figures especially with subplots
        # seems that ax is sometimes not None whereas the Figure does not contain any ax
        # print("Graph.plot. {}. # {}.".format(fig, plt.get_fignums()), "ax: {}. {}".format(ax, len(fig.get_axes())), os.path.basename(self.attr("filename")))
        # if len(fig.get_axes()) == 0 and ax is not None:
        #    print("MAYBE FAILS!?!")
        #    print("   ", matrix, gs)

        # either ax is None and gs is not None, either the other way around
        subplotsncols, subplotsnrows = matrix.shape
        subplotidkwargs = {
            "textcoords": "figure fraction",
            "fontsize": fontsize,
            "horizontalalignment": "right",
            "verticalalignment": "top",
        }
        axes = []
        if ax is not None:
            axes.append(ax)
        fig.patch.set_alpha(0.0)
        if not ifSubPlot:  # set figure size if fig is not a subplot
            fig.set_size_inches(*figsize, forward=True)
        colorbar_ax = []

        # adjust positions of the axis within the graph (subplots_adjust)
        attr_spa = self.attr("subplots_adjust")
        subplot_adjust = SubplotsAdjuster.merge(subplotAdjustDef, attr_spa, fig)
        if fig is not None:
            fig.subplots_adjust(**subplot_adjust)
        if gs is not None:
            gs.update(**subplot_adjust)
        # auxiliary graph, to be created when new axis, filled with curves and
        # plotted when new axis is created
        graphAux = None  # auxiliary Graph, calling plot() when creating new axis
        graphAuxKw = {"config": self._config.filename}

        # plot data
        type_plot = self.attr("typeplot")
        # set default graph scales (lin, log, etc.)
        if type_plot != "" and gs is None:
            xarg = "log" if type_plot in ["semilogx", "loglog"] else "linear"
            try:
                ax.set_xscale(xarg)
            except (ValueError, AttributeError):
                print("Display error, try to go back to linear Transform ?")
            yarg = "log" if type_plot in ["semilogy", "loglog"] else "linear"
            try:
                ax.set_yscale(yarg)
            except (ValueError, AttributeError):
                print("Display error, try to go back to linear Transform?")

        groupedplotters = GroupedPlotters()

        ignoreNext = 0
        if handles is None or not isinstance(handles, list):
            handles = []
        pairsAxCurve = []
        # Start looping on the curves
        ax_ = ax
        for curve_i in range(len(self)):
            handle = None
            if ignoreNext > 0:
                ignoreNext -= 1
                continue  # ignore this curve, go the next one...
                # for example if previous curve type was scatter
            curve = self.curve(curve_i)
            attr = curve.getAttributes()
            # if curve is hidden: ignore it
            if curve.isHidden():
                continue

            # Inset: if curve contains information for an inset in the Graph?
            if isinstance(curve, Curve_Inset):
                val = curve.attr("insetfile")
                inset = Graph(self.filenamewithpath(val), **graphAuxKw)
                coords = (
                    list(attr["insetcoords"])
                    if "insetcoords" in attr
                    else [0.3, 0.2, 0.4, 0.3]
                )
                if "insetupdate" in attr and isinstance(attr["insetupdate"], dict):
                    inset.update(attr["insetupdate"])
                axInset = fig.add_axes(coords)
                axInset.ticklabel_format(useOffset=False)
                axInset.patch.set_alpha(0.0)
                if curve.attr("insetfile") in ["", " "] or inset.length() == 0:
                    # nothing in provided Graph -> created axis becomes active one
                    if graphAux is not None:
                        # if there was already auxiliary graph: display it, create anew
                        GraphIO.plot(
                            graphAux,
                            ifSave=False,
                            ifExport=False,
                            figAx=[fig, ax],
                            ifSubPlot=True,
                        )
                    graphAux = inset
                    ax = axInset
                    sca_errorhandled(ax)
                    axTwinX, axTwinY, axTwinXY = None, None, None
                else:
                    # found a Graph, place it in inset. Next Curve in existing
                    # axes. No change for graphAux
                    inset.plot(
                        figAx=[fig, axInset],
                        ifSave=False,
                        ifExport=False,
                        ifSubPlot=True,
                    )
                    continue  # go to next Curve

            # Subplots: if more than 1 subplot is expected
            if gs is not None:
                # if required, create the new axis
                if ax is None or isinstance(curve, Curve_Subplot):
                    if graphAux is not None:
                        GraphIO.plot(
                            graphAux,
                            ifSave=False,
                            ifExport=False,
                            figAx=[fig, ax],
                            ifSubPlot=True,
                        )
                        graphAux = None
                    ax = GraphIO._newaxis(
                        self,
                        curve,
                        fig,
                        gs,
                        matrix,
                        subplotsid,
                        subplotscounter,
                        subplotidkwargs,
                    )
                    sca_errorhandled(ax)
                    axTwinX, axTwinY, axTwinXY = None, None, None
                    subplotscounter += 1
                    axes.append(ax)
                else:  # we go on, not enough information to create subplot
                    pass
            # shall we plot a Graph object instead of a Curve in this new axis?
            if isinstance(curve, Curve_Subplot):
                # new axes, so create new graphAux (is None by construction)
                val = curve.attr("subplotfile")
                if val not in [" ", "", None]:
                    graphAux = Graph(self.filenamewithpath(val), **graphAuxKw)
                else:
                    graphAux = Graph("", **graphAuxKw)
                upd = {"subplots_adjust": ""}
                if isinstance(curve.attr("subplotupdate"), dict):
                    upd.update(curve.attr("subplotupdate"))
                graphAux.update(upd)
                if val not in [" ", "", None]:
                    continue  # go to next Curve
                # if no file then try to plot data of the Curve

            if graphAux is not None:
                # if active axis is not main one, place Curve in graphAux
                if isinstance(curve, (Curve_Inset, Curve_Subplot)):
                    graphAux.append(Curve(curve.data, curve.getAttributes()))
                else:
                    graphAux.append(curve)
                continue

            # twin axis: look which axis to use
            ax_ = ax
            ifTwinX = curve.attr("ax_twinx")  # if True
            ifTwinY = curve.attr("ax_twiny")
            if ifTwinX and ifTwinY:
                if axTwinXY is None:
                    axTwinXY = ax.twinx().twiny()
                ax_ = axTwinXY
            elif ifTwinX:
                if axTwinX is None:
                    axTwinX = ax.twinx()
                ax_ = axTwinX
            elif ifTwinY:
                if axTwinY is None:
                    axTwinY = ax.twiny()
                ax_ = axTwinY

            # do the actual plotting
            try:
                handle, ignoreNext = curve.plot(
                    ax_,
                    groupedplotters,
                    graph=self,
                    graph_i=curve_i,
                    type_plot=type_plot,
                    ignoreNext=ignoreNext,
                )
            except Exception as e:
                # need to catch exceptions as we want to proceed with graph
                msg = "{} Exception occured in Curve.plot(), Curve no {}. Error message: {}"
                print(msg.format(type(e).__name__, curve_i, e))
                print(sys.exc_info()[0].__name__, sys.exc_info()[-1].tb_lineno)

            # remind pairs ax - curve
            pairsAxCurve.append({"ax": ax_, "curve": curve})

            # Add colorbar if required
            if handle is not None and curve.attr("colorbar", False):
                kwargs = {}
                if isinstance(attr["colorbar"], dict):
                    kwargs = deepcopy(attr["colorbar"])
                adjust = deepcopy(subplotColorbar)
                if "adjust" in kwargs:
                    adjust = kwargs["adjust"]
                    del kwargs["adjust"]
                if isinstance(adjust[-1], str):
                    if adjust[-1] == "ax":
                        # coordinates relative to the ax and not to figure
                        axbounds = ax_.get_position().bounds
                        adjust[0] = adjust[0] * axbounds[2] + axbounds[0]
                        adjust[1] = adjust[1] * axbounds[3] + axbounds[1]
                        adjust[2] = adjust[2] * axbounds[2]
                        adjust[3] = adjust[3] * axbounds[3]
                    else:
                        print(
                            "WARNING GraphOI.plot colorbar: cannot",
                            "interpret last keyword (must be numeric, or",
                            "'ax'.",
                        )
                    del adjust[-1]
                if "labelsize" in kwargs:
                    pass  # TODO
                colorbar_ax.append({"ax": fig.add_axes(adjust), "adjusted": False})
                colorbar_ax[-1]["cbar"] = fig.colorbar(
                    handle, cax=colorbar_ax[-1]["ax"], **kwargs
                )
                try:
                    colorbar_ax[-1]["cbar"].solids.set_rasterized(True)
                    colorbar_ax[-1]["cbar"].solids.set_edgecolor("face")
                except Exception:
                    pass
                subplotColorbar[0] -= 0.11  # default for next colorbar
                # set again ax_ as the current plt active ax, not the colorbar
                sca_errorhandled(ax_, "ax_ after colorbar")

            # store handle in list
            if handle is not None:
                if isinstance(handle, list):
                    for h in handle:
                        handles.append({"handle": h})
                else:
                    handles.append({"handle": handle})
                    if curve.attr("type") == "scatter":
                        handles[-1].update({"setlegendcolormap": True})
        # end of loop over Curves

        # generates boxplot and violinplot, after plotting of other curves
        groupedplotters.plot(ax, ax_)

        # final display of auxiliary axes/graph
        if graphAux is not None:
            GraphIO.plot(
                graphAux, ifSave=False, ifExport=False, figAx=[fig, ax], ifSubPlot=True
            )
            graphAux = None
            # main graph: work on initial axis
            ax = axes[0]
            sca_errorhandled(ax, "- 2")

        # usual graph cosmetics
        curvedummy = self.curve(-1) if len(self) > 0 else Curve([[0], [0]], {})
        fontsizeset = []

        if "title" in self.graphInfo:
            out = ParserMisc.setAxisLabel(ax.set_title, self.graphInfo["title"], self)
            if out["size"]:
                fontsizeset.append(ax.title)
        if "xlabel" in self.graphInfo and gs is None:
            # gs check because xlabel could be automatically set upong file parsing
            out = ParserMisc.setAxisLabel(ax.set_xlabel, self.graphInfo["xlabel"], self)
            if out["size"]:
                fontsizeset.append(ax.xaxis.label)
        if "ylabel" in self.graphInfo and gs is None:
            # gs check because ylabel could be automatically set upong file parsing
            out = ParserMisc.setAxisLabel(ax.set_ylabel, self.graphInfo["ylabel"], self)
            if out["size"]:
                fontsizeset.append(ax.yaxis.label)
        # labels twin axis
        if "twinx_ylabel" in self.graphInfo:
            if axTwinX is not None:
                val = self.graphInfo["twinx_ylabel"]
                out = ParserMisc.setAxisLabel(axTwinX.set_ylabel, val, self)
                if out["size"]:
                    fontsizeset.append(axTwinX.yaxis.label)
            elif axTwinXY is not None:
                val = self.graphInfo["twinx_ylabel"]
                out = ParserMisc.setAxisLabel(axTwinXY.set_ylabel, val, self)
                if out["size"]:
                    fontsizeset.append(axTwinXY.yaxis.label)
        if "twiny_xlabel" in self.graphInfo:
            if axTwinY is not None:
                val = self.graphInfo["twiny_xlabel"]
                out = ParserMisc.setAxisLabel(axTwinY.set_xlabel, val, self)
                if out["size"]:
                    fontsizeset.append(axTwinY.xaxis.label)
            elif axTwinXY is not None:
                val = self.graphInfo["twiny_xlabel"]
                out = ParserMisc.setAxisLabel(axTwinXY.set_xlabel, val, self)
                if out["size"]:
                    fontsizeset.append(axTwinXY.xaxis.label)

        # lcoation of labels
        if "xlabel_coords" in self.graphInfo:
            val = self.graphInfo["xlabel_coords"]
            if isinstance(val, list):
                ax.xaxis.set_label_coords(val[0], val[1])
            else:
                ax.xaxis.set_label_coords(0.5, val)
        if "ylabel_coords" in self.graphInfo:
            val = self.graphInfo["ylabel_coords"]
            if isinstance(val, list):
                ax.yaxis.set_label_coords(val[0], val[1])
            else:
                ax.yaxis.set_label_coords(val, 0.5)

        # other
        if "axhline" in self.graphInfo:
            lst = ParserAxhline(self.graphInfo["axhline"])
            lst.plot(ax.axhline, curvedummy, alter, type_plot)
        if "axvline" in self.graphInfo:
            lst = ParserAxvline(self.graphInfo["axvline"])
            lst.plot(ax.axvline, curvedummy, alter, type_plot)

        # xlim, ylim. Start with xtickslabels as this guy would override xlim
        if "xtickslabels" in self.graphInfo:
            val = self.graphInfo["xtickslabels"]
            ticklocs, kw = ParserMisc.set_xyticklabels(val, ax.xaxis)
            if "size" in kw:
                fontsizeset.append("ax.get_xticklabels()")
        if "ytickslabels" in self.graphInfo:
            val = self.graphInfo["ytickslabels"]
            ticklocs, kw = ParserMisc.set_xyticklabels(val, ax.yaxis)
            if "size" in kw:
                fontsizeset.append("ax.get_yticklabels()")

        # xlim should be after xtickslabels
        aac = [alter, curvedummy]
        if "xlim" in self.graphInfo:
            ParserMisc.alterLim(ax, self.graphInfo["xlim"], "x", *aac)
        if "ylim" in self.graphInfo:
            ParserMisc.alterLim(ax, self.graphInfo["ylim"], "y", *aac)
        if "twinx_ylim" in self.graphInfo:
            if axTwinXY is not None:
                ParserMisc.alterLim(axTwinXY, self.graphInfo["twinx_ylim"], "y", *aac)
            if axTwinX is not None:
                ParserMisc.alterLim(axTwinX, self.graphInfo["twinx_ylim"], "y", *aac)
        if "twiny_xlim" in self.graphInfo:
            if axTwinXY is not None:
                ParserMisc.alterLim(axTwinXY, self.graphInfo["twiny_xlim"], "x", *aac)
            if axTwinY is not None:
                ParserMisc.alterLim(axTwinY, self.graphInfo["twiny_xlim"], "x", *aac)

        # xticksstep
        if "xticksstep" in self.graphInfo and not ignoreXLim:
            val = self.graphInfo["xticksstep"]
            ParserMisc.set_xytickstep(val, ax.xaxis, ax.get_xlim())
        if "yticksstep" in self.graphInfo and not ignoreYLim:
            val = self.graphInfo["yticksstep"]
            ParserMisc.set_xytickstep(val, ax.yaxis, ax.get_ylim())

        # text annotations
        if "text" in self.graphInfo:
            textxyDefault = (0.05, 0.95)
            text = deepcopy(self.attr("text"))
            texy = deepcopy(self.attr("textxy", None))
            args = deepcopy(self.attr("textargs"))
            if not isinstance(text, list):
                text = [text]
            if not isinstance(texy, list):
                texy = [texy] * len(text)
            else:  # can be for example [0.05,0.95] -> should be duplicated
                if (
                    len(texy) == 2
                    and not isinstance(texy[0], (list, tuple))
                    and not isinstance(texy[1], (list, tuple))
                ):
                    if texy == ["", ""]:
                        texy = [None] * len(text)
                    else:
                        texy = [texy] * len(text)
            while len(texy) < len(text):  # duplicate if missing elements
                texy.append(texy[-1])
            if not isinstance(args, list):
                args = [args]
            args = [deepcopy(a) for a in args]  # might be dplicate by construction
            while len(args) < len(text):
                args.append(deepcopy(args[-1]))
            for i in range(len(text)):
                if text[i] != "":
                    if (
                        texy[i] != ""
                        and texy[i] is not None
                        and texy[i] not in [["", ""], ("", "")]
                    ):
                        if "xytext" in args[i]:
                            msg = "Graph plot annotate: {} textxy {} override {}"
                            print(msg.format(text[i], texy[i], args[i]["xytext"]))
                        args[i].update({"xytext": texy[i]})
                    if "xytext" not in args[i]:
                        if "xy" in args[i]:
                            args[i].update({"xytext": args[i]["xy"]})
                        else:
                            args[i].update({"xytext": textxyDefault})
                    if "xy" not in args[i]:
                        args[i].update(
                            {"xy": args[i]["xytext"]}
                        )  # this one exists for sure
                    if "xycoords" not in args[i] and "textcoords" in args[i]:
                        args[i].update({"xycoords": args[i]["textcoords"]})
                    if "xycoords" in args[i] and "textcoords" not in args[i]:
                        args[i].update({"textcoords": args[i]["xycoords"]})
                    if "textcoords" not in args[i]:
                        args[i].update({"textcoords": "figure fraction"})
                    if "xycoords" not in args[i]:
                        args[i].update({"xycoords": "figure fraction"})
                    if "fontsize" not in args[i] and "fontsize" in self.graphInfo:
                        args[i].update({"fontsize": self.graphInfo["fontsize"]})
                    # set_clip_box to draw all that can be show,
                    # to couple with 'annotation_clip'=False
                    arrow_set_clip_box = False
                    if (
                        "arrowprops" in args[i]
                        and "set_clip_box" in args[i]["arrowprops"]
                    ):
                        arrow_set_clip_box = args[i]["arrowprops"]["set_clip_box"]
                        del args[i]["arrowprops"]["set_clip_box"]
                    # print('Graph plot annotate', text[i], 'args', args[i])
                    try:
                        ann = ax.annotate(text[i], **args[i])
                        if arrow_set_clip_box:
                            ann.arrow_patch.set_clip_box(ax.bbox)
                    except Exception as e:
                        msg = "Exception {} during ax.annotate: {}, {}"
                        print(msg.format(type(e), text[i], args[i]))
                        print(e)

        # legend
        if gs is None:  # create legend on current ax
            legPropUser = deepcopy(self.attr("legendproperties", default="best"))
            if not isinstance(legPropUser, dict):
                legPropUser = {"loc": legPropUser}
            if "loc" in legPropUser:
                legPropUser["loc"] = str(legPropUser["loc"]).lower()
                rep = {
                    "best": 0,
                    "ne": 1,
                    "nw": 2,
                    "sw": 3,
                    "se": 4,
                    "right": 5,
                    "w": 6,
                    "e": 7,
                    "s": 8,
                    "n": 9,
                    "center": 10,
                }
                if legPropUser["loc"] in rep:
                    legPropUser["loc"] = rep[legPropUser["loc"]]
            prop = {}
            if "fontsize" in self.graphInfo:
                prop = {"size": self.graphInfo["fontsize"]}
            if "prop" in legPropUser:
                prop.update(legPropUser["prop"])
            legPropUser["prop"] = prop
            if "fontsize" in legPropUser:
                legPropUser["prop"].update({"size": legPropUser["fontsize"]})
                del legPropUser["fontsize"]
            legLabelColor = None
            if "color" in legPropUser:
                legLabelColor = legPropUser["color"].lower()
                del legPropUser["color"]
            legProp = {
                "fancybox": True,
                "framealpha": 0.5,
                "frameon": False,
                "numpoints": 1,
                "scatterpoints": 1,
            }
            legProp.update(legPropUser)
            labels = []
            for i in range(len(handles) - 1, -1, -1):
                label = (
                    handles[i]["handle"].get_label()
                    if hasattr(handles[i]["handle"], "get_label")
                    else ""
                )  # not legend if dont know how to find it
                if isinstance(handles[i]["handle"], mpl.image.AxesImage):
                    label = None
                if label is None or len(label) == 0 or label[0] == "_":
                    del handles[i]  # delete curve handle if no label is to be shown
                else:
                    labels.append(label)
            leg = ax.legend(
                [h["handle"] for h in handles], labels[::-1], **legProp
            )  # labels is reversed by construction
            # color of legend
            if legLabelColor is not None:
                if legLabelColor == "curve":
                    # special value: same color for text and lines
                    lines, texts = leg.get_lines(), leg.get_texts()
                    if len(texts) > len(lines):  # issue with errorbar, notably
                        lines = []
                        for h in handles:
                            if isinstance(h["handle"], tuple):
                                lines.append(h["handle"][0])
                            else:
                                lines.append(h["handle"])
                    for line, text in zip(lines, texts):
                        try:
                            text.set_color(line.get_color())
                        except Exception:
                            pass
                else:
                    for text in leg.get_texts():
                        text.set_color(legLabelColor)
            # legend title
            legTitle = deepcopy(self.attr("legendtitle"))
            # legendTitle can be of the form ['some title', {'fontsize':24}]
            if legTitle != "":
                setfunc = []
                if isinstance(legTitle, list):
                    for key in ["color", "position", "fontsize", "align"]:
                        if key in legTitle[1]:
                            setfunc.append([key, legTitle[1][key]])
                            del legTitle[1][key]
                    prop.update(legTitle[1])
                    legTitle = legTitle[0]
                leg.set_title(legTitle, prop=prop)
                for setf in setfunc:
                    if setf[0] == "align":
                        leg._legend_box.align = setf[1]
                    else:
                        if hasattr(leg.get_title(), "set_" + setf[0]):
                            getattr(leg.get_title(), "set_" + setf[0])(setf[1])
                        else:
                            print(
                                "Warning GraphIO do not knwo what to do",
                                "with keyword",
                                setf[0],
                            )
            # in legend, set color of scatter correctly
            for i in range(len(handles)):
                if (
                    "setlegendcolormap" in handles[i]
                    and handles[i]["setlegendcolormap"]
                ):
                    try:
                        leg.legendHandles[i].set_color(
                            handles[i]["handle"].get_cmap()(0.5)
                        )  # plt.cm.afmhot(.5))
                    except AttributeError as e:
                        print("error setlegendcolormap", e)
                        pass

        # arbitrary functions
        if "arbitraryfunctions" in self.graphInfo:
            for fun in self.graphInfo["arbitraryfunctions"]:
                try:
                    # print ('Graph plot arbitrary func', fun, type(fun))
                    f, arg, opt = fun[0], fun[1], fun[2]
                    fsplit = f.split(".")
                    # print ('   ', ax, fsplit, len(fsplit))
                    obj = ax
                    if len(fsplit) > 0 and fsplit[0] == "fig":
                        obj = fig
                        del fsplit[0]
                    for subf in fsplit:
                        if subf.startswith("_"):
                            # to prevent possible security vulnerability.
                            # not sure it is "safe".
                            msg = "Error: arbitraryfunctions does not accept functions or attributes starting with character _. {}"
                            raise RuntimeError(msg.format(fsplit))
                        if hasattr(obj, "__call__"):
                            obj = getattr(obj(), subf)
                        else:
                            obj = getattr(obj, subf)
                    # print ('   ', obj, type(obj))
                    # handle ticks locators and formatters, take objects as arguments
                    if fsplit[-1] in [
                        "set_major_locator",
                        "set_minor_locator",
                        "set_major_formatter",
                        "set_minor_formatter",
                    ]:
                        if len(arg) > 0 and isinstance(arg[0], str):
                            try:
                                asplit = resplit("[()]", arg[0])
                                args = [strToVar(a) for a in asplit[1:] if a != ""]
                                import matplotlib.ticker as tck

                                arg = [getattr(tck, asplit[0])(*args)]
                            except Exception:
                                pass  # just continue with unmodified arg
                    # print('      ',arg)
                    obj(*arg, **opt)
                except Exception as e:
                    msg = (
                        "Exception in function Graph.plot arbitrary functions. "
                        "Exception {} {}"
                    )
                    print(msg.format(type(e), e))
                    pass

        # font sizes
        if gs is None:
            listLabels = [ax.title, ax.xaxis.label, ax.yaxis.label]
            if "ax.get_xticklabels()" not in fontsizeset:
                listLabels += ax.get_xticklabels()
            if "ax.get_yticklabels()" not in fontsizeset:
                listLabels += ax.get_yticklabels()
            # maybe an automatic detection of all the axis would be more robust
            # than trying to list all possible existing axis?
            if axTwinX is not None:
                listLabels += [axTwinX.yaxis.label] + axTwinX.get_yticklabels()
            if axTwinY is not None:
                listLabels += [axTwinY.xaxis.label] + axTwinY.get_xticklabels()
            if axTwinXY is not None:
                listLabels += (
                    [axTwinXY.xaxis.label, axTwinXY.yaxis.label]
                    + axTwinXY.get_xticklabels()
                    + axTwinXY.get_yticklabels()
                )
            for item in listLabels:
                if item not in fontsizeset:
                    item.set_fontsize(fontsize)

        # before saving image: restore self to its initial state
        self.update(restore)

        # DPI
        saveDPI = 300 if "dpi" not in self.graphInfo else self.graphInfo["dpi"]

        # save the graph if an export format was provided
        exportFormat = ""
        if imgFormat in [".txt", ".xml"]:
            exportFormat = imgFormat
            imgFormat = ""
        if ifSave:
            if len(imgFormat) == 0:
                imgFormat = self.config("save_imgformat", ".png")
            if not isinstance(imgFormat, list):
                imgFormat = [imgFormat]
            for imgForma_ in imgFormat:
                imgFormatTarget = ""
                if imgForma_ == ".emf":
                    # special: we save svg and convert into emf using inkscape
                    imgFormatTarget = ".emf"
                    imgForma_ = ".svg"

                filename_ = filename + imgForma_
                if not self.attr("saveSilent"):
                    print("Graph saved as " + filename_.replace("/", "\\"))
                # plt.savefig(filename_, transparent=True, dpi=saveDPI)
                fig.savefig(filename_, transparent=True, dpi=saveDPI)
                self.filename = filename_

                if imgFormatTarget == ".emf":
                    GraphIO.convertSVGtoEMF(self, filename_, imgFormat, imgFormatTarget)

        if ifExport:
            self.export(filesave=(filename + exportFormat))
        return [fig, axes]

    def convertSVGtoEMF(self, filename, imgFormat, imgFormatTarget):
        success = False
        inkscapepath = self.config("inkscape_path", [])
        if isinstance(inkscapepath, str):
            inkscapepath = [inkscapepath]
        inkscapepath += ["inkscape"]
        import subprocess

        for p in inkscapepath:
            if not os.path.exists(p):
                continue  # cannot find inkscape executable
            try:
                fileemf = filename[: -len(imgFormat)] + imgFormatTarget
                command = (
                    '"'
                    + p
                    + '" --without-gui --export-emf="'
                    + fileemf
                    + '" "'
                    + filename
                    + '"'
                )
                out = subprocess.call(command)
                if out == 0:
                    print("Graph saved as " + fileemf.replace("/", "\\"))
                    success = True
                    break
                else:
                    print("Graph save as .emf: likely error (return value " + out + ")")
            except Exception as e:
                print("Exception during save in .emf format:", e)
                pass
        if not success:
            print("Could not save image in .emf format. Please check the following:")
            print(" - A version of inkscape is available,")
            print(" - file config.txt in grapa directory,")
            print(
                " - in file config.txt a line exists, similar as that, and",
                "indicate a valid inkscape executable e.g.: inkscape_path",
                r'["C:\Program Files\Inkscape\inkscape.exe"]',
            )
