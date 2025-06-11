# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import ast
import codecs
from string import Formatter, Template
import logging

import numpy as np

logger = logging.getLogger(__name__)


class TemplateCustom(Template):
    """Custom string Template to be able to specify key together with formatting"""

    # braceidpattern = "(?a:[_a-z][_a-z0-9 \[\]]*)"  # added: whitespace, [, ]
    # added: whitespace, [, ]; also, formatter indication as ${key, :.0f}
    # braceidpattern = r"(?a:[_a-z][_a-z0-9 \[\]]*(,[ ]*[:.0-9\-\+eEfFgG%]+)?)"
    # added: whitespace, [, ]; also, formatter indication as ${key:.0f}
    braceidpattern = r"(?a:[_a-z][_a-z0-9 \[\]]*(:[.0-9\-\+eEfFgG%]+)?)"


def format_string_curveattr(curve, formatter: str):
    """
    Format a string according to formatting using python string Template,
    with curve attributes as variables. Uase: case: curve.label_auto()
    - curve: a Curve object
    - formatter: e.g. "${sample} ${_simselement}"
    """
    # if modify implementation: beware GraphSIMS, label_auto(...)
    t = TemplateCustom(formatter)
    try:
        identifiers = t.get_identifiers()
    except AttributeError:  # python < 3.11
        # identifiers must be surrounded by {}
        # identifiers = [ele[1] for ele in Formatter().parse(formatter) if ele[1]]
        identifiers = []
        for ele in Formatter().parse(formatter):
            if ele[1]:
                identifiers.append(ele[1])
                if ele[2]:
                    identifiers[-1] = identifiers[-1] + ":" + ele[2]
    attrs = {}
    for key in identifiers:
        attrs[key] = str(curve.attr(key))
        if ":" in key:  # to supper e.g. "${temperature, :.0f}"
            split = key.split(":")
            ke = split[0].strip()
            fm = split[1].strip()
            form = "{:" + fm + "}"
            if len(fm) > 0 and curve.has_attr(ke) > 0:
                val = curve.attr(ke)
                try:
                    attrs[key] = form.format(val)
                except ValueError:
                    try:
                        attrs[key] = form.format(float(val))
                    except Exception:
                        msg = "format_string_curveattr: format '{}', {}, {}, {}"
                        msgargs = [form, type(val), val, curve.attr("filename")]
                        logger.warning(msg.format(*msgargs), exc_info=True)
                        attrs[key] = str(curve.attr(ke))
            else:
                attrs[key] = str(curve.attr(ke))
    string = t.safe_substitute(attrs)
    return string


MOJIBAKE_WINDOWS = ["\xc2", "\xc3"]
ESCAPE_WARNING = {"a": "\a", "b": "\b", "r": "\r", "f": "\f"}


def stringToVariable(val):
    """Deprecated."""
    print("WARNING: stringToVariable, use strToVar instead (arg", val, ")")
    return strToVar(val)


def strToVar(val):
    """Converts a string into variable, based on ast.literal_eval, but trying otherwise
    if fails"""
    # val = codecs.getdecoder("unicode_escape")(val)[0]
    flagast = False
    try:
        val = float(val)
    except ValueError:
        try:
            val = ast.literal_eval(val)
            flagast = True
        except Exception:
            if isinstance(val, str) and len(val) > 1:
                if (val[0] == "[" and val[-1] == "]") or (
                    val[0] == "(" and val[-1] == ")"
                ):
                    try:
                        val = [
                            float(v)
                            for v in val[1:-1]
                            .replace(" ", "")
                            .replace("np.inf", "inf")
                            .split(",")
                        ]
                    except ValueError:
                        pass
    # print('strToVar', val, flagast)
    if not flagast:  # not needed when created through ast.literal_eval
        val = strUnescapeIter(val)
    # print('   ', val)
    return val


def strUnescapeIter(var):
    """TRies its best to unescape special characters. Used when converting str to
    variable, if ast.literal_eval failed and another parsing approach was tried.
    Iterates over the elements of lists and dicts"""
    # likely, implementation not robust and could be much more elegant.
    if isinstance(var, list):
        for i in range(len(var)):
            var[i] = strUnescapeIter(var[i])
    elif isinstance(var, dict):
        for key in var:
            var[key] = strUnescapeIter(var[key])
    elif isinstance(var, str):
        try:
            var = codecs.getdecoder("unicode_escape")(var)[0]
        except Exception:
            msg = "strUnescapeIter: Exception in codecs.getdecoder, input ({})."
            logger.error(msg.format(var))
        try:
            # handling of special characters transformed into mojibake
            # The 2-pass code below can clean inputs with special characters
            # encoded in:
            # ANSI (e.g. Windows-1252), UTF-8, UTF-8 escaped (eg. \xb5 for mu)
            # First pass
            for char in MOJIBAKE_WINDOWS:
                if char in var:
                    # print('Suspicion mojibake, latin-1-utf-8 conversion')
                    var = var.encode("latin-1").decode("utf-8")
                    break
            # Second pass. Appears necessary with some charset input
            for char in MOJIBAKE_WINDOWS:
                if char in var:
                    # print('Suspicion mojibake (2), latin-1-utf-8 conversion')
                    var = var.encode("latin-1").decode("utf-8")
                    break
            for key in ESCAPE_WARNING:
                if ESCAPE_WARNING[key] in var:
                    msg = (
                        "strUnescapeIter: escape character detected (\\{}). "
                        "Consider double backslash (\\\\{}) for Latex code."
                    )
                    logger.warning(msg.format(key, key))
        except Exception:
            msg = (
                "strUnescapeIter (mojibake). Possibly, mix of special characters "
                "and escape sequences in same input ({})."
            )
            logger.error(msg.format(var), exc_info=True)
            # keep current out value. Likely to fail
    return var


def varToStr(val):
    """Converts a variable into str, ready to be displayed by the GUI"""
    # return codecs.getencoder("unicode_escape")(out)[0].decode('utf-8')
    try:
        out = repr(val).strip("'")
    except Exception:
        msg = "varToStr Exception, input {}."
        logger.error(msg.format(val), exc_info=True)
        out = ""
    return out


def listToString(val):
    """Converts a list to a string with suitable formatting"""
    return (
        "["
        + ", ".join(
            [str(el) if not isinstance(el, str) else "'" + el + "'" for el in val]
        )
        + "]"
    )


def restructuredtext_to_text(string: str, nparammax: int = -1) -> list:
    """Converts a rest restructured text (e.g. docstring) into text, to be e.g. printed
    into the grapa console.
    NB: not a "clean" implementation.

    :param string: e.g. a docstring
    :param nparammax: only consider the first nparammax parameters of the docstring,
           the later ones are ignored
    :return: a list of str (lines)"""
    if string is None:
        string = ""
    out = []
    lines = string.split("\n")

    trailing_all = [len(a) - len(a.lstrip(" ")) for a in lines]
    trailing = [a for a in trailing_all if a > 0]
    n_trailing = 0
    if len(trailing) > 1:
        n_trailing = np.min(trailing[1:])
    elif len(trailing) == 1:
        n_trailing = np.min(trailing)
    trailing_str = " " * n_trailing

    nparam = 0
    section = 0
    for i, lineraw in enumerate(lines):
        if len(lineraw) == 0:
            continue
        line = lineraw[n_trailing:] if lineraw.startswith(trailing_str) else lineraw
        if len(line) == 0:
            continue
        # do not want to keep the reStructuredText formatting in printout
        if line.startswith(":param "):
            section = 1
            if 0 <= nparammax <= nparam:
                continue  # skip the line
            nparam += 1
            line = "-" + line[6:]
        if line.startswith(":return:"):
            section = 2
            line = "Returns:" + line[8:]
        if line.startswith(":returns:"):
            section = 2
            line = "Returns:" + line[9:]
        if line.startswith("  "):
            line = "  " + line.lstrip(" ")
        if section == 1 and line[0] not in [" ", "-"]:
            line = "  " + line
        out.append(line)
        if (
            not line.endswith(":")
            and len(lines) > i + 2
            and len(lines[i + 1].strip(" ")) == 0
            and lines[i + 2].lstrip(" ").startswith(":param ")
        ):
            out.append("Arguments:")
    return out


class TextHandler:
    """text are draw onto plot as Annotations"""

    @classmethod
    def check_valid(cls, graph):
        """validates"""
        text = graph.attr("text", None)
        texy = graph.attr("textxy", "")
        targ = graph.attr("textargs", "")
        if text is None:
            graph.update({"textxy": "", "textargs": ""})
            return "", "", ""

        onlyfirst = False if isinstance(text, list) else True
        # transform everything into lists
        text, texy, targ = cls._sanitize_text_input(text, texy, targ)
        if onlyfirst:
            text, texy, targ = text[0], texy[0], targ[0]
        if text != graph.attr("text"):
            msg = "Corrected attribute text {} (former {})"
            logger.warning(msg.format(text, graph.attr("text")))
        if texy != graph.attr("textxy") and graph.attr("textxy", None) is not None:
            msg = "Corrected attribute textxy {} (former {})."
            logger.warning(msg.format(texy, graph.attr("textxy")))
        if targ != graph.attr("textargs"):
            msg = "Corrected attribute textargs {} (former {})."
            logger.warning(msg.format(targ, graph.attr("textargs")))
        graph.update({"text": text, "textxy": texy, "textargs": targ})
        return text, texy, targ

    @staticmethod
    def _sanitize_text_input(text, texy, targ):
        if not isinstance(text, list):
            text = [text]
        if not isinstance(texy, list):
            texy = [texy]
        if not isinstance(targ, list):
            targ = [targ]
        if (
            len(texy) == 2
            and not isinstance(texy[0], (list, tuple))
            and texy[0] != ""
            and texy[1] != ""
        ):
            texy = [texy]  # if texy was like (0.5,0.8)
        for i in range(len(targ)):
            if not isinstance(targ[i], dict):
                targ[i] = {}
        for i in range(len(texy)):
            if not isinstance(texy[i], (tuple, list)):
                texy[i] = ""
        while len(texy) < len(text):
            texy.append(texy[-1])
        while len(targ) < len(text):
            targ.append(targ[-1])
        while len(texy) > len(text):
            texy.pop()
        while len(targ) > len(text):
            targ.pop()
        return text, texy, targ

    @classmethod
    def add(cls, graph, text, textxy, textargs=None):
        """
        Adds a text to be annotated in the plot, handling the not-so-nice
        internal implementation
        text, textxy, textargs: as single elements, or as lists (1 item per annotation)
        """
        if textargs is None:
            textargs = []
        if not isinstance(textargs, list):
            textargs = [textargs]
        restore = [
            {
                "text": graph.attr("text"),
                "textxy": graph.attr("textxy"),
                "textargs": graph.attr("textargs"),
            }
        ]
        text, textxy, textargs = cls._sanitize_text_input(text, textxy, textargs)
        attrs = {"text": text, "textxy": textxy, "textargs": textargs}
        if graph.attr("text", None) is None:
            graph.update(attrs)
        else:  # need to merge existing with new
            cls.check_valid(graph)  # makes sure existing info are as lists
            for key, value in attrs.items():
                graph.update({key: graph.attr(key) + value})
        cls.check_valid(graph)
        return restore

    @classmethod
    def remove(cls, graph, by_id=-1):
        """
        By default, removes the last text annotation in the list (pop)

        :param graph: the Graph object to act on
        :param by_id: index of the annotation to remove
        :returns: a dict with initial text values, prior to removal (use case: restore)
        """
        restore = [
            {
                "text": graph.attr("text"),
                "textxy": graph.attr("textxy"),
                "textargs": graph.attr("textargs"),
            }
        ]
        attrs = ["text", "textxy", "textargs"]
        cls.check_valid(graph)
        if graph.attr("text", None) is None:
            pass
        else:
            for key in attrs:
                tmp = graph.attr(key)
                del tmp[by_id]
                graph.update({key: tmp})
        return restore
