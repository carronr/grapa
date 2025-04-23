# -*- coding: utf-8 -*-
"""
Grapa

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""


__version__ = "0.7.0.0"
__author__ = "Romain Carron"

import os
import sys
import json
import logging

logger = logging.getLogger(__name__)
_logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grapalog.log")
logging.basicConfig(
    filename=_logfile,
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger_handler = logging.StreamHandler(sys.stdout)
logger_handler.setLevel(logging.WARNING)
logger_handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
logger.addHandler(logger_handler)
# logging.basicConfig(format='%(asctime)s  %(name)s - %(levelname)s - %(message)s')


def _load_kw(filename):
    # data contained in json file structured as:
    # [
    #     ["== Figure ==", "", []],
    #     ["figsize", "Figure size (inch).\nExample:", [[6.0, 4.0]]],
    #     ...
    # ]
    out = {}
    # open file with json
    with open(filename, "r", encoding="utf-8") as file:
        datalist = json.load(file)
    # process data
    out["keys"] = [line[0] for line in datalist]
    # textual help, concatenate with examples if appropriate
    out["guitexts"] = []
    for line in datalist:
        out["guitexts"].append(line[1])
        if line[1].endswith("Example:") or line[1].endswith("Examples:"):
            aux = [
                str(li) if not isinstance(li, str) else '"' + li + '"' for li in line[2]
            ]
            out["guitexts"][-1] += " " + ", ".join(aux)
    # lists of examples - NOT cast into str()
    out["guiexamples"] = [line[2] for line in datalist]
    # # test if needed
    # for i in range(len(out["keys"])):
    #     print(out["keys"][i])
    #     print(out["guitexts"][i])
    #     for example in out["guiexamples"][i]:
    #         print("   ", example)
    return out


_folder = os.path.dirname(os.path.realpath(__file__))
KEYWORDS_GRAPH = _load_kw(os.path.join(_folder, "keywordsdata_graph.txt"))
KEYWORDS_CURVE = _load_kw(os.path.join(_folder, "keywordsdata_curve.txt"))
KEYWORDS_HEADERS = _load_kw(os.path.join(_folder, "keywordsdata_headers.txt"))


# Import selected Curve subtypes to make sure they will be registered when calling
# __subclasses__. The other subtypes in .datatypes will be imported by Graph.
from grapa.curve_inset import Curve_Inset  # make sure ready when call __subclasses__
from grapa.curve_subplot import Curve_Subplot  # ake sure ready when call __subclasses__
from grapa.curve_image import Curve_Image  # make sure ready when call __subclasses__


# imports for convenience to packages outside the scope of grapa
from grapa.graph import Graph
from grapa.curve import Curve
from grapa.utils.string_manipulations import strToVar


def grapa():
    """Start grapa GUI"""
    from grapa.GUI import build_ui

    build_ui()
