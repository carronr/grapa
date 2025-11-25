# -*- coding: utf-8 -*-
"""Class and functions assisting with import and export of data.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import logging
from re import split as resplit
from typing import Literal, Union, TYPE_CHECKING
from enum import Enum

from grapa import KEYWORDS_GRAPH  # , KEYWORDS_CURVE, KEYWORDS_HEADERS
from grapa.curve import Curve
from grapa.utils.parser_generic import FileParserGeneric, _identify_delimiter
from grapa.utils.parser_database import FileParserDatabase
from grapa.utils.error_management import FileNotReadError, issue_warning

if TYPE_CHECKING:
    from grapa.graph import Graph

logger = logging.getLogger(__name__)


class ReadAs(Enum):
    """for complement={"readas": "database"}, tested 'readas' in ['database', 'generic']"""

    DATABASE = "database"
    GENERIC = "generic"


def file_read_first3lines(filename: str) -> tuple:
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
        msg = "Exception read_file_first3lines"
        issue_warning(logger, msg, exc_info=True)
        return "", "", ""


class FileParserDispatcher:
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

    GRAPHTYPE_GRAPH = "Graph"
    GRAPHTYPE_UNDETERMINED = "Undetermined data type"

    @staticmethod
    def append_curve_from_datainput(graph: "Graph", data, attributes):
        """Initializes content of object using data given in constructor.

        :meta private:"""
        graph.append(Curve(data, {}))
        if not isinstance(attributes, dict):
            if attributes != "":
                msg = "append_curve_from_datainput: ignored attributes '{}'."
                issue_warning(logger, msg.format(attributes))
            attributes = {}
        graph.update(attributes, if_all=False)
        # nevertheless want to import all subclass types, if not already done
        graph.get_list_subclasses_parse()

    @classmethod
    def read(cls, graph: "Graph", complement: Union[dict, Literal[""]] = ""):
        """
        Reads a file, opens graph.filename
        graph: a graph object
        Complement: ...

        :meta private:"""
        if not isinstance(complement, dict):
            if len(complement) > 0:
                msg = "readDataFile, data loss complement {}."
                issue_warning(logger, msg.format(complement))
            complement = {}

        isfilecontent = False
        if "isfilecontent" in complement and complement["isfilecontent"]:
            isfilecontent = True

        # reads the file to identify types
        if not isfilecontent:
            filenamesplit, fileext = os.path.splitext(graph.filename)
            fileext = fileext.lower()
        else:
            filenamesplit, fileext = "", ".txt"
        filenamebase = filenamesplit.split("/")[-1].split("\\")[-1]
        attributes = {"filename": graph.filename}
        if len(complement) > 0:
            for key in complement:
                attributes.update({key.lower(): complement[key]})
            # attributes.update({"complement": complement})
        # default attributes of curve
        if "label" not in attributes:
            labeldefault = filenamesplit.split("/")[-1].split("\\")[-1]
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
        to_delete = []
        for key in complement:
            try:
                if key.lower() in KEYWORDS_GRAPH["keys"]:
                    graph.update({key: complement[key]})
                    to_delete.append(key)
            except Exception:
                pass
        for key in to_delete:
            del complement[key]

        # read first line to identify datatype
        line1, line2, line3 = "", "", ""
        if not isfilecontent:
            line1, line2, line3 = file_read_first3lines(graph.filename)
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
        readas = str(attributes["readas"]) if "readas" in attributes else ""
        if readas.lower() in [ReadAs.DATABASE.value, ReadAs.GENERIC.value]:
            msg_end = "opened using standard methods."
            isfilecontent = cls.readDataFromFileTxt(
                graph, attributes, isfilecontent=isfilecontent
            )
            loaded = True

        # if not, then try every possibility
        if not loaded:
            subclasses = graph.get_list_subclasses_parse()
            kwlines = {"line1": line1, "line2": line2, "line3": line3}
            for subclass in subclasses:
                if subclass.isFileReadable(filenamebase, fileext, **kwlines):
                    msg_end = "opened as " + subclass.FILEIO_GRAPHTYPE + "."
                    graph.update({"meastype": subclass.FILEIO_GRAPHTYPE})
                    res = subclass.readDataFromFile(
                        graph, attributes, fileName=filenamebase, **kwlines
                    )
                    if res is None or res != False:
                        loaded = True
                        break

        # if not, then try default methods
        if not loaded:
            # other not readable formats
            if fileext in [".pdf", ".png"]:
                msg_end = "do not know how to open."
            else:
                # keep the default opening mechanism in the main class
                msg_end = "opened using standard methods (default)."
                isfilecontent = cls.readDataFromFileTxt(
                    graph, attributes, isfilecontent=isfilecontent
                )

        if msg_end != "":
            if not graph.silent:
                msg = "File " + filenamesplit.split("/")[-1] + "... " + msg_end
                issue_warning(logger, msg)

    @classmethod
    def _funcReadDataFile(cls, content, attributes=None):
        """Try to guess which method is best suited to open the file"""
        if attributes is None:
            attributes = {}
        if "readas" in attributes:
            if attributes["readas"].lower() == ReadAs.DATABASE.value:
                return cls.readDataFromFileDatabase
            if attributes["readas"].lower() == ReadAs.GENERIC.value:
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
    def readDataFromFileTxt(
        cls, graph: "Graph", attributes, isfilecontent=False
    ) -> bool:
        """
        Reads content of a .txt file, and parse it as a database or as a generic file.
        """
        filename = graph.filename
        if not isfilecontent:
            try:
                with open(filename, "r", errors="backslashreplace") as file:
                    content = [resplit("[\t ,;]", line.strip(":\r\n")) for line in file]
            except FileNotFoundError:
                return False
            except (IOError, OSError, UnicodeError) as e:
                msg = "Exception readDataFromFileTxt. %s, %s."
                msa = (type(e), e)
                logger.error(msg, *msa, exc_info=True)
                raise FileNotReadError(msg % msa)
        else:
            content = filename.split("\n")
            delimiter = _identify_delimiter(content)
            content = [resplit(delimiter, line) for line in content]
            while len(content) > 0 and content[-1] == [""]:
                del content[-1]

        func = cls._funcReadDataFile(content, attributes)
        if not isfilecontent:
            # intentionally do not provide file content, function will read file
            # again with the desired data processing
            func(graph, attributes)
        else:
            graph.filename = ".txt"  # free some memory
            attributes.update({"filename": "From clipboard"})
            func(graph, attributes, fileContent=content)
            isfilecontent = False
        return isfilecontent

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
        graph.update({"meastype": cls.GRAPHTYPE_UNDETERMINED})
        return FileParserGeneric.read(
            graph,
            attributes,
            fileContent=fileContent,
            ifReplaceCommaByPoint=ifReplaceCommaByPoint,
            **kwargs
        )

    @classmethod
    def readDataFromFileDatabase(
        cls, graph: "Graph", attributes, fileContent=None, **kwargs
    ):
        """
        fileContent: list of lists containing file content (imagine csv file)
        kwargs: useless here
        """
        return FileParserDatabase.read(
            graph, attributes, filecontent=fileContent, **kwargs
        )
