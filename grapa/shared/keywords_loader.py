"""Lazy loader for keyword metadata used by graph/curve parsing and GUI hints."""

import os
from typing import Dict
import json

_folder = os.path.dirname(os.path.realpath(__file__))
_KW_CACHE: Dict[str, Dict] = {}


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


def _get_kw(relpath: str):
    if relpath not in _KW_CACHE:
        _KW_CACHE[relpath] = _load_kw(os.path.join(_folder, relpath))
    return _KW_CACHE[relpath]


def keywords_graph():
    """Return graph keyword metadata (keys, GUI texts, examples). Cached after load."""
    return _get_kw("keywordsdata_graph.txt")


def keywords_curve():
    """Return curve keyword metadata (keys, GUI texts, examples). Cached after load."""
    return _get_kw("keywordsdata_curve.txt")


def keywords_headers() -> Dict:
    """Return headers keyword headers (keys, GUI texts, examples). Cached after load."""
    return _get_kw("keywordsdata_headers.txt")
