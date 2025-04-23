# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 22:22:45 2017

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import os
import xml.etree.ElementTree as ElementTree

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.utils.curve_subclasses_utils import FileLoaderOnDemand


class GraphXML(Graph):
    """
    Can read xml files.
    Can be used from outside grapa with class methods: ::

        content = GraphXML.parsexml_to_dict(filename, preset)
        datasets = GraphXML.split_format_datasets(content, filename, preset)

    Intepret xml hierarchy as keys of a dictionary
    A fundamental problem is that xml format accepts several children to have same names
    etc., while python dictionary only accept uique keys.
    The following behaviors are implemented:

    - increment key values: ``<text>value 1</text><text>value 2</text>``
      --> ``{"text#0": "value1", "text#1": "value2"}``

    - attribs_append_to_tag: ``<length unit="m">123</length>``
      with ``[["unit", "()", "delete"]]``  -->  ``{"length(m)": 123}``

    - attribs_as_metadata: process attrib as if it were a subelement.
      ``<item id="abc"><sub>content</sub></item>``
      with ``[["id", "delete"]]``  -->  ``{"item_id": "abc", "item_sub": "content"}``

    See file XML_fitparameters.txt for more details.
    """

    FILEIO_GRAPHTYPE = "Undetermined measurement type"

    PATHXMLFILE = os.path.dirname(os.path.abspath(__file__))

    DATAFORMATS_FILE = "XML_dataformats.txt"
    DATAFORMATS = FileLoaderOnDemand(os.path.join(PATHXMLFILE, DATAFORMATS_FILE), 0)

    @classmethod
    def isFileReadable(cls, _fname, fileext, line1="", line2="", line3="", **_kwargs):
        for preset in cls.DATAFORMATS:
            if fileext[1:] == preset.attr("extension"):
                if (
                    line1.startswith(preset.attr("line1"))
                    and line2.startswith(preset.attr("line2"))
                    and line3.startswith(preset.attr("line3"))
                ):
                    return True
        return False

    def readDataFromFile(self, attributes, line1="", line2="", line3="", **_kwargs):
        # NOTE: self is a Graph object, but not a GraphXML
        # Therefore, calls, to GraphXML.function
        filename = attributes["filename"]
        _, fileext = os.path.splitext(filename)
        presetc = None
        for pres in GraphXML.DATAFORMATS:
            if fileext[1:] == pres.attr("extension"):
                if (
                    line1.startswith(pres.attr("line1"))
                    and line2.startswith(pres.attr("line2"))
                    and line3.startswith(pres.attr("line3"))
                ):
                    presetc = pres
        if presetc is None:
            raise RuntimeError("GraphXML thought could open the file, actually not?!?")
        preset = presetc.get_attributes()

        # read the file
        content = GraphXML.parsexml_to_dict(filename, preset)
        datasets = GraphXML.split_format_datasets(content, filename, preset)
        # format the Graph, create the Curves
        keys = list(datasets.keys())
        keys.sort()
        for key in keys:
            dataset = datasets[key]
            metadata = dataset["metadata"]
            data = dataset["data"]
            self.append(Curve(data, attributes))
            self[-1].update(metadata)
            # move required attributes from Curve into Graph
            tograph = ["xlabel", "ylabel", "title"]
            for attr in tograph:
                val = self[-1].attr(attr)
                if val != "":
                    self.update({attr: val})
                    self[-1].update({attr: ""})
            if preset["cast"] != "":
                self.castCurve(preset["cast"], -1, silentSuccess=True)
        # cosmetics, last changes
        val = presetc.attr("_meastype")
        if len(val) > 0:
            self.update({"meastype": val})

    @staticmethod
    def _datastr_to_listfloat(val):
        separator = " "
        if " " in val:
            separator = " "
        elif ";" in val:
            separator = ";"
        elif "," in val:
            separator = ","
        else:
            msg = "Don't know how to interpret data string: {}..."
            print(msg.format(val[:50]))
        valsfloat = [float(v) for v in val.split(separator)]
        return valsfloat

    @staticmethod
    def _issue_log(type_, key_, value_, issues_container):
        # issues_container is a dict containing lists
        if type_ not in issues_container:
            issues_container[type_] = {}
        if key_ not in issues_container[type_]:
            issues_container[type_].update({key_: []})
        issues_container[type_][key_].append(value_)

    @classmethod
    def _output_add(cls, key_, value_, key_replace, output, issues_container):
        # modifies output nd possibly issues_container
        # user-requested modifications of key
        for repl in key_replace:
            if repl[2] == "equal":
                if key_ == repl[0]:
                    key_ = repl[1]
            else:
                key_ = key_.replace(repl[0], repl[1])
        # insert into output dict, assuming key does not already exist
        if key_ == "":
            cls._issue_log("key empty", key_, value_, issues_container)
            return
        if key_ not in output:
            output[key_] = str(value_)
        else:
            cls._issue_log("key repeat", key_, value_, issues_container)

    @classmethod
    def _walk_element(cls, element, taghierarchy, elementtag=None, **kwargs):
        # tree = kwargs["tree"]
        issues = kwargs["issues"]
        output = kwargs["output"]
        preset = kwargs["preset"]
        argso = [preset["key_replace"], output, issues]
        sep = preset["separator"]
        # retrieve information of element
        attrib = dict(element.attrib)
        if elementtag is None:
            elementtag = element.tag
        if "roottag_strip" in kwargs and kwargs["roottag_strip"] is not None:
            elementtag = elementtag.replace(kwargs["roottag_strip"], "")

        # file-format-specific: modification of element tag
        # to clarify output and lower risk of confusion with multiple elements
        # with same tags
        for instr in preset["attribs_append_to_tag"]:
            # input format eg. {"attribs_append_to_tag": [["axis", "-", "delete"]]}
            # tag becomes elementtag + "-" + attrib['axis'],
            # then delete (forgets) attrib['axis']
            if instr[0] in attrib:
                if instr[1] == "()":  # e.g. quantity(unit)
                    elementtag = "{}({})".format(elementtag, attrib[instr[0]])
                else:  # e.g. quantity-axisname
                    elementtag = "{}{}{}".format(elementtag, instr[1], attrib[instr[0]])
                if instr[2] == "delete":
                    del attrib[instr[0]]

        # build on the hierarchy line - actually here only a list of branches
        taghierarchy = list(taghierarchy) + [elementtag]
        key = sep.join([t for t in taghierarchy])
        text = element.text.strip() if element.text is not None else ""

        # file-format-specific: user-requested metadata generation from attrib
        for keyadd in preset["attribs_as_metadata"]:
            if isinstance(keyadd, list):  # e.g. ["id", "delete"]
                if keyadd[0] in attrib:
                    cls._output_add(key + sep + keyadd[0], attrib[keyadd[0]], *argso)
                    if keyadd[1] == "delete":
                        del attrib[keyadd[0]]
            else:  # e.g. "id"
                if keyadd in attrib:
                    cls._output_add(key + sep + keyadd, attrib[keyadd], *argso)

        # retrieve metadata
        if len(attrib) == 0 and len(text) == 0:
            # may result in data loss if attribs_append_to_tag removes all content
            pass
        elif len(attrib) == 0 and len(text) > 0:
            cls._output_add(key, text, *argso)
        elif len(attrib) == 1 and len(text) == 0:
            k = list(attrib.keys())[0]
            key += sep + k
            cls._output_add(key, attrib[k], *argso)
        elif len(attrib) > 1 and len(text) == 0:
            cls._output_add(key, str(attrib), *argso)
        elif len(attrib) > 0 and len(text) > 0:
            cls._output_add(key, str(attrib) + "; " + text, *argso)
        else:  # should not happen thanks to cases above
            cls._issue_log("not parsed", key, str(attrib) + "; " + text, issues)

        # loop over children - modify child tags to prevent duplicates
        children, childtags, duplicates = [], [], {}
        for idx, child in enumerate(element, 1):
            children.append(child)
            childtag = child.tag
            if childtag not in childtags:
                childtags.append(childtag)
            else:
                if childtag not in duplicates:
                    duplicates[childtag] = 1
                childtags.append(childtag + "#" + str(duplicates[childtag]))
                duplicates[childtag] += 1
        for key in duplicates:  # also modify the first one "#0"
            childtags[childtags.index(key)] = key + "#0"
        for i in range(len(children)):
            cls._walk_element(
                children[i], taghierarchy, elementtag=childtags[i], **kwargs
            )

    # end of iteration

    @classmethod
    def parsexml_to_dict(cls, filename, preset, silent=False):
        """in principle, a generic xml parser, agnostic of file content.
        As few as possible file-format-specific modifications
        Some custom behavios configured in preset

        - "attribs_append_to_tag": []

          ``[["unit", "()", "delete"]]``: parses ``<length unit="m">123</length>``
          --> ``{"length(m)": 123}``

          ``[["axis", "-", "delete"]]``: parses
          ``<position axis="AXIS_1">123</position>`` --> ``{"position-AXIS_1": 123}``

          Special value[2]: "delete" will delete the attrib[key] after append.
          Purposes: clear output, compact and self-sufficient key-value pairs

        - "attribs_as_metadata": to extract attrb as if it were a subelement.

          With ``[["id", "delete"]]``: ``<item id="abc"><sub>content</sub></item>``
          --> ``"item_id": "abc", "item_sub": "content"``

        - "separator": "_",  # could be "_" or "/"

        - "key_replace": format metadata key to shorten metadata keys, e.g.
          ``[["xrdMeasurement", "xrd", "replace"], ["xrds_xrd_", "xrd_", "replace"],
          ["xrds_xrd", "xrd", "equal"]]``

          list element [2] can be "equal" (replace only if str are equal), or
          "replace" (apply str.replace() function)

        - "roottag_strip": to remove excess information from the tag. "{}" to remove
          '{url}' found in root tag
        """
        # make sure user preferences are fill, complete if needed
        preset_default = {
            "attribs_append_to_tag": [["unit", "()", "delete"]],
            "attribs_as_metadata": [],  # ["appendNumber"],
            "separator": "_",
            "key_replace": [],
            "roottag_strip": "{}",
        }
        for key in preset_default:
            if key not in preset:
                preset[key] = preset_default[key]

        # parsing of  XML file
        output = {}  # will be modified during file parsing and walkthrough
        issues = {}  # will be modified during file parsing and walkthrough
        tree = ElementTree.parse(filename)
        # roottag_strip: to clean up tags from "excess" information
        tagroot = tree.getroot().tag
        roottag_strip = str(preset["roottag_strip"])
        if roottag_strip == "{}":
            if "{" in tagroot and "}" in tagroot:
                roottag_strip = tagroot.split("}")[0] + "}"
            else:
                roottag_strip = None
                print("Could not find {...} in root.tag", tagroot)
        # walk iteratively through elements
        kwwalk = {
            # "tree": tree,
            "preset": preset,
            "roottag_strip": roottag_strip,  # modified from preset
            "issues": issues,  # container to store issues encountered issues
            "output": output,  # container to store output
        }  # kwwalk carried along through iterations
        cls._walk_element(tree.getroot(), [], **kwwalk)

        # report issues during parsing, if any
        if len(issues) > 0 and not silent:
            print("Issues were encountered during file parsing:")
            for issuetype in issues:
                for key, value in issues[issuetype].items():
                    msg = "- {} with key '{}', values lost: {}"
                    print(msg.format(issuetype, key, value))
        # return
        return output

    @classmethod
    def split_format_datasets(cls, datainput, filename, preset=None):
        """retrieve data from within metadata
        Splits content according to possibly several datasets.
        Datasets consist in metadata, and 2 1-D vectors of data: datax, datay
        Returns a dict {idx1: {"data": [datax, datay], "metadata": {...}}, idx2: ...}"""
        # file-format-specific configuration
        preset_default = {
            "_xlabel": "Position",  # underscore bc technical reason FileLoaderOnDemand
            "_ylabel": "Counts",  # underscore bc technical reason FileLoaderOnDemand
            "keyend_valuesy": "(counts)",
            "keyend_valuesx": "",  # empty string: does not look for that
            # obviously file-format-specific, at least useful to know what to write that
            "keyend_valuesx_start": "2Theta(deg)_startPosition",
            "keyend_valuesx_end": "2Theta(deg)_endPosition",
            "separator": "_",  # separator in key hierarchy. Could be "_", or "/"
            "datasetsplit": "",  # Dataset split.
            # Example: "" (if only one dataset expected), or "scan#", with last
            # character assumed to be a separator
            "filename_to_label_replace": [  # order matters
                ["- ", "_"],
                [" ", ""],
                [";", "_"],
                ["_", " "],
            ],
        }

        # sanity checks
        preset = dict(preset)
        for key in preset_default:  # set to empty if not present
            if key not in preset:
                preset[key] = preset_default[key]
        keyend_valuesx = preset["keyend_valuesx"]
        keyend_valuesy = preset["keyend_valuesy"]
        sep = preset["separator"]
        dssplit = preset["datasetsplit"]
        dssplit_ = sep + dssplit
        dssplitsep = dssplit[-1] if len(dssplit) > 0 else "#"
        # prepare output
        metadata_all = {}
        datasets = {}
        # add graph formatting metadata - may be overridden by input content
        metadata_all["filename_initial"] = filename
        metadata_all["xlabel"] = preset["_xlabel"]
        metadata_all["ylabel"] = preset["_ylabel"]

        # populate content into data structure, one main, and one for each dataset
        keys = list(datainput.keys())
        for key in keys:
            value = datainput[key]
            # file-format-specific identification of dataset information, dispatch data
            if dssplit_ not in key or dssplit == "":
                metadata_all[key] = value
            else:
                tmp = key.split(sep)
                for t in tmp:
                    if t.startswith(dssplit):
                        idx = dssplitsep.join(t.split(dssplitsep)[1:])
                        if idx not in datasets:
                            datasets[idx] = {"metadata": {}, "data": {}}
                        if len(keyend_valuesx) > 0 and key.endswith(keyend_valuesx):
                            valx = cls._datastr_to_listfloat(value)
                            datasets[idx]["data"]["valuesx"] = valx
                        elif len(keyend_valuesy) > 0 and key.endswith(keyend_valuesy):
                            valy = cls._datastr_to_listfloat(value)
                            datasets[idx]["data"]["valuesy"] = valy
                        else:  # is not the data, just metadata
                            key_ = key.replace(
                                "{}{}{}".format(dssplit_, idx, sep),
                                "{}{}".format(dssplit_[:-1], sep),
                            )
                            if key.endswith("{}{}".format(dssplit_, idx)):
                                key_ = key.replace(
                                    "{}{}".format(dssplit_, idx), dssplit_[:-1]
                                )
                            datasets[idx]["metadata"][key_] = value
                        break
        # in case could not identify specific datasets - maybe not a xrdml file,
        # yet want to export something
        if len(datasets) == 0:
            idx = ""
            datasets[idx] = {"metadata": {}, "data": {}}
            keys = list(metadata_all.keys())
            for key in keys:
                if key.endswith(preset["keyend_valuesy"]):
                    print("... retrieved ydata from key '{}'".format(key))
                    valy = cls._datastr_to_listfloat(metadata_all[key])
                    datasets[idx]["data"]["valuesy"] = valy
                    del metadata_all[key]
                    break

        # loop over the datasets to create x etc.
        out = {}
        for dskey, content in datasets.items():
            # add a label
            fname = os.path.basename(filename)
            label = ".".join(fname.split(".")[:-1])
            for repl in preset["filename_to_label_replace"]:
                label = label.replace(repl[0], repl[1])
            if len(datasets) > 1:
                label += " {}{}".format(dssplit_, dskey)
            content["metadata"]["label"] = label
            # metadata
            metadata = dict(metadata_all)
            metadata.update(content["metadata"])
            # datay
            if "valuesy" not in content["data"]:
                msg = "WARNING: Could dot find data within metadata"
                print(msg)
                content["data"]["valuesy"] = [0]
                # TODO MAYBE: some aggressive data retrieval?
            y = content["data"]["valuesy"]
            if len(y) == 0:
                y = [0]
            # datax
            if "valuesx" not in content["data"]:
                # must retrieve/generate somehow x data
                if len(keyend_valuesx) > 0:
                    msg = "WARNING datax not found, expected to find a match with '{}'"
                    print(msg.format(keyend_valuesx))
                # file-format-specific: create x from start and end values
                start, end, num_points = None, None, len(y)
                for key, value in metadata.items():
                    if key.endswith(preset["keyend_valuesx_start"]):
                        start = float(value)
                    if key.endswith(preset["keyend_valuesx_end"]):
                        end = float(value)
                if start is None or end is None:
                    msg = "WARNING scan#{}: unable to compute xdata (start {} stop {})"
                    print(msg.format(dskey, start, end))
                    x = list(range(len(y)))
                else:
                    if num_points == 1:
                        x = [0]
                    else:
                        step_size = (end - start) / (num_points - 1)
                        x = [start + i * step_size for i in range(num_points)]
                if len(x) != len(y):
                    msg = "WARNING len(xdata) not the same as len(ydata) ({}, {})"
                    print(msg.format(len(x), len(y)))
                datasets[dskey]["data"]["valuesx"] = x
            x = datasets[dskey]["data"]["valuesx"]
            # format output
            try:
                dskeyint = int(dskey)
                if dskeyint not in out:
                    dskey = dskeyint
            except ValueError:
                pass
            out[dskey] = {"metadata": metadata, "data": [x, y]}
            # print("Dataset", dskey)
            # for key, value in metadata:
            #    print(" ", key, value)
        return out

    @classmethod
    def convert_file(cls, filename, preset="auto"):
        """
        # parse an XML file and reformat the content into a grapa-style output file
        # Maybe useful to debug and try converg to suitable values to input in
        # XML_dataformats.txt input
        # not tested actually, more to store this bit of code somewhere
        """
        if preset == "auto":
            preset = {
                "roottag_strip": "{}",
                "key_replace": [
                    ["xrdMeasurement", "xrd", "replace"],
                    ["xrds_xrd_", "xrd_", "replace"],
                    ["xrds_xrd", "xrd", "equal"],
                ],
                "attribs_append_to_tag": [
                    ["axis", "-axis", "delete"],
                    ["unit", "()", "delete"],
                ],
                "attribs_as_metadata": [["id", "delete"], ["name", "delete"]],
                "separator": "_",  # could be "_" or "/"
                # for formatting
                "xlabel": "2Theta (deg)",
                "ylabel": "Counts",
                "keyend_valuesx": "",
                "keyend_valuesx_start": "2Theta(deg)_startPosition",
                "keyend_valuesx_end": "2Theta(deg)_endPosition",
                "keyend_valuesy": "_dataPoints_counts(counts)",
                "datasetsplit": "scan#",  # last character assumed to be a separator
                "filename_to_label_replace": [
                    ["- ", "_"],
                    [" ", ""],
                    [";", "_"],
                    ["_", " "],
                ],  # order matters
            }

        def export_file(filenameout, data, metadata):
            keys = list(metadata.keys())
            keys.sort()
            # write file
            with open(filenameout, "w") as f:
                for key_ in keys:
                    f.write("{}\t{}\n".format(key_, metadata[key_]))
                for i in range(len(data[0])):
                    f.write("{}\t{}\n".format(data[0][i], data[1][i]))
            print("File written with success: {}".format(filenameout))

        print("Processing file: {}".format(filename))
        content = cls.parsexml_to_dict(filename, preset)
        datasetdict = cls.split_format_datasets(content, filename, preset)
        for key, content in datasetdict.items():
            file_out = ".".join(filename.split(".")[:-1]) + "_{}_export.txt"
            export_file(file_out.format(key), content["data"], content["metadata"])
