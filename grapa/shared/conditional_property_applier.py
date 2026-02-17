# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import logging

from grapa.graph import Graph
from grapa.shared.error_management import issue_warning

logger = logging.getLogger(__name__)


class ConditionalPropertyApplier:
    """
    Apply changes to attributes of Curves within a Graph, for the Curves that satisfy a
    test
    """

    MODES_BYINPUTTYPE = {
        "any": ["==", "!=", ">", ">=", "<", "<="],
        "str": [
            "startswith",
            "endswith",
            "contains",
            "not startswith",
            "not endswith",
            "not contains",
        ],
    }
    # flatten in a single list
    MODES_VALUES = [x for _, xs in MODES_BYINPUTTYPE.items() for x in xs]

    @staticmethod
    def _evaluate_values(valref, valtest, mode):
        if mode == "==":
            return valref == valtest
        elif mode == "!=":
            return valref != valtest
        elif mode == ">":
            return valref > valtest
        elif mode == ">=":
            return valref >= valtest
        elif mode == "<":
            return valref < valtest
        elif mode == "<=":
            return valref <= valtest
        elif mode == "startswith":
            return str(valref).startswith(str(valtest))
        elif mode == "endswith":
            return str(valref).endswith(str(valtest))
        elif mode == "contains":
            return str(valtest) in str(valref)
        elif mode == "not startswith":
            return not str(valref).startswith(str(valtest))
        elif mode == "not endswith":
            return not str(valref).endswith(str(valtest))
        elif mode == "not contains":
            return str(valtest) not in str(valref)
        msg = (
            "GraphConditionalPropertyApplier._evaluate_values: unsupported mode "
            "{}, return False. Input values {}, {}."
        )
        issue_warning(logger, msg.format(mode, valref, valtest))
        return False

    @classmethod
    def _coerce_mode(cls, mode):
        if mode in cls.MODES_VALUES:
            return mode
        new = "=="
        msg = (
            "GraphConditionalProperty: unsupported mode, changed '{}' for '{}'. "
            "Possible values: {}."
        )
        issue_warning(logger, msg.format(mode, new, cls.MODES_VALUES))
        return new

    @classmethod
    def apply(
        cls, graphorcurve, test_prop, test_mode, test_value, apply_prop, apply_value
    ):
        """Changes values of property to a given value, for all curves satisfying
        test condition."""
        mode = cls._coerce_mode(test_mode)
        # if Graph: loop and execute over the curves
        if isinstance(graphorcurve, Graph):
            for curve in graphorcurve:
                cls.apply(curve, test_prop, mode, test_value, apply_prop, apply_value)
            return
        # supposedly a Curve object: execute behavior
        val = graphorcurve.attr(test_prop)
        try:
            test = cls._evaluate_values(val, test_value, mode)
        except (AttributeError, TypeError) as e:
            msg = (
                "GraphConditionalProperty.apply: Error during evaluation: {}. "
                "Comparison: {} ({}), {}, {}."
            )
            issue_warning(logger, msg.format(e, val, test_prop, mode, test_value))
        else:
            if test:
                graphorcurve.update({apply_prop: apply_value})
