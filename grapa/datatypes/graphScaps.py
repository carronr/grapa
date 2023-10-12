# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.datatypes.curveJV import CurveJV
from grapa.datatypes.curveCV import CurveCV
from grapa.datatypes.curveCf import CurveCf
from grapa.datatypes.curveEQE import CurveEQE


class GraphScaps:
    """
    Parse the output of SCAPS files
    """

    FILEIO_GRAPHTYPE = "SCAPS output data"

    @classmethod
    def isFileReadable(cls, fileName, fileExt, line1="", line2="", line3="", **kwargs):
        if (
            fileExt in [".iv", ".cv", ".cf", ".qe"]
            and line1.startswith("SCAPS ")
            and "ELIS-UGent: Version scaps" in line1
        ):
            return True
        return False

    def readDataFromFile(self, attributes, **kwargs):
        # rely on code below in the file to parse the content of SCAPS files
        simulations = Simulations()
        simulations.parse_file(self.filename)
        graph = simulations.as_graph()
        graph[0].update(attributes)
        self.merge(graph)


class Simulations:
    """Intermediate data conteiner useful while parsing the scaps file"""

    def __init__(self):
        self.filename = ""
        self.simulations = []

    def new_simulation(self):
        # print('NEW SIMULATION DETECTED IN FILE')
        self.simulations.append({"data": [[], []], "attrs": {}})
        return True

    def parse_file(self, filename):
        """
        Parse the content of a scaps output file, populate the Simulations object
        """
        self.filename = filename
        with open(filename) as file:
            counter = 0
            endoffile = False
            while not endoffile:
                lineheader = self.parse_paragraph(file)
                # if len(self.simulations) > 1:
                #    break
                if lineheader is False:
                    counter += 1  # means line was empty
                else:
                    counter = 0
                    if lineheader is not True:
                        self.parse_data(file, lineheader)
                if counter > 10:
                    # several times consecutive empty lines -> likely end of file
                    endoffile = True
        return True

    def parse_paragraph(self, file):
        """continue to parse the file that was opened elsewhere (file = open(...) )"""
        line = file.readline()
        if line.startswith("  "):
            # we arrive to a region of file storing data
            return line
        line = line.strip()
        if len(line) == 0:
            return False

        data, attrs = self.current_data_attrs()  # will work directly on these objects
        # extract category from first line in paragraph
        split = line.split(":")
        category = split[0].strip().replace("**", "")
        if category.startswith("SCAPS ") and not self.current_is_empty():
            # is a new simulation
            self.new_simulation()
            data, attrs = self.current_data_attrs()
        num = 0
        value = ":".join(split[1:]).strip() if ":" in line else line
        attrs.update({category: value})
        # proceed with next lines
        while True:
            line = file.readline().strip()
            if len(line) <= 1:
                break
            # separate key from value
            sep = ":"
            split = [e.strip() for e in line.split(sep)]
            if len(split) == 1 and "\t" in line:
                sep = "\t"
                split = [e.strip() for e in line.split(sep)]
            # process data
            # print(split)
            if "Batch parameters" in category:
                key, value = category + " " + str(num) + " key", " ".join(split[0:-1])
                attrs.update({key: value})
                ke1, valu1 = category + " " + str(num) + " value", float(split[-1])
                attrs.update({ke1: valu1})
                attrs.update({value: valu1})
            else:
                if len(split) == 0:
                    key, value = category + " " + str(num), line
                else:
                    key = category + " " + split[0]
                    value = sep.join(split[1:]).replace("\t", " ")
                    if key.endswith(" ="):
                        key = key[:-2]
                    if (
                        category
                        == "solar cell parameters deduced from calculated IV-curve"
                        and " " in value
                    ):
                        splittmp = value.split(" ")
                        if splittmp[1] == "%":
                            value = float(splittmp[0]) * 0.01
                        else:
                            key = key + " (" + splittmp[1] + ")"
                            value = float(splittmp[0])
                attrs.update({key: value})
            num += 1
        if num == 0:
            return False  # only empty line
        return True  # some content was parsed

    def parse_data(self, file, lineheader):
        """continue to parse the file that was opened elsewhere (file = open(...) )"""
        data, attrs = self.current_data_attrs()
        # parse only first 2 columns
        split = lineheader.split("\t")
        attrs.update({split[0].strip(): split[1].strip()})
        attrs.update({"labels": [split[0].strip(), split[1].strip()]})
        line = file.readline()  # expect 1 empty line
        while True:
            line = file.readline()
            if len(line) <= 1:
                break
            split = line.split("\t")
            x, y = float(split[0].strip()), float(split[1].strip())
            data[0].append(x)
            data[1].append(y)
        return True

    def current_data_attrs(self):
        if len(self.simulations) == 0:
            self.new_simulation()
        return self.simulations[-1]["data"], self.simulations[-1]["attrs"]

    def current_is_empty(self, index=-1):
        sim = self.simulations[index]
        if (
            len(sim["data"][0]) == 0
            and len(sim["data"][1]) == 0
            and len(sim["attrs"].keys()) == 0
        ):
            return True
        return False

    def as_graph(self):
        """returns a Graph with all JV curves"""

        # splitext = os.path.splitext(self.filename)[0]
        dataext = str(os.path.splitext(self.filename)[1]).replace(".", "").upper()
        curvetypes = {"IV": CurveJV, "CV": CurveCV, "CF": CurveCf, "QE": CurveEQE}
        curveclass, curvekwargs = Curve, {}
        if dataext in curvetypes:
            curveclass = curvetypes[dataext]
            if dataext == "IV":
                curvekwargs.update({"silent": True})

        graph = Graph()
        for si in range(len(self.simulations)):
            sim = self.simulations[si]
            # print('GRAPH', sim['data'])
            # print(sim['attrs'])
            if len(sim["data"][0]) > 0:
                graph.append(curveclass(sim["data"], sim["attrs"], **curvekwargs))
                pkeys, pvals = [], []
                i = 0
                flag = True
                while flag:
                    key = graph[-1].attr("Batch parameters " + str(i) + " key")
                    val = graph[-1].attr("Batch parameters " + str(i) + " value")
                    if len(key) > 0:
                        pkeys.append(key)
                        pvals.append(val)
                    else:
                        flag = False
                    i += 1
                graph.update({"legendtitle": "\n".join(pkeys)})
                graph[-1].update({"label": "; ".join([str(v) for v in pvals])})
        labels = graph[-1].attr("labels")
        graph.update(
            {"xlabel": labels[0], "ylabel": labels[1], "filename": self.filename}
        )
        if dataext == "IV":
            graph.update({"axhline": [0, {"linewidth": 0.5}]})
        return graph
