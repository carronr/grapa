# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import logging
from typing import Optional, Union, TYPE_CHECKING

from grapa.shared.error_management import issue_warning, IncorrectInputError

if TYPE_CHECKING:
    from grapa.graph import Graph
    from grapa.curve import Curve

logger = logging.getLogger(__name__)


def update_values_keys(
    graph_or_curve: Union["Graph", "Curve"], *values, keys: Optional[list] = None
):
    """
    Performs update({key1: value1, key2: value2, ...}).

    :param curve: Curve object
    :param values: value1, value2, ...
    :param keys: ['key1', 'key2', ...]. list[str]
    :returns: True if success, False otherwise
    """
    if keys is None:
        keys = []
    if not isinstance(keys, list) or len(keys) != len(values):
        msg = (
            "update_values_keys: 'list_keys' must be a list, same len as "
            "*values. Provided ({}), len {}, expected len {}."
        )
        issue_warning(logger, msg.format(keys), len(keys), len(values))
        return False

    if len(keys) != len(values):
        msg = (
            "update_values_keys: len of keyword argument list_keys ({}) must match"
            " the number of provided values as values ({}). Stop at minimal len."
        )
        issue_warning(logger, msg.format(len(keys), len(values)))
    for i in range(min(len(keys), len(values))):
        graph_or_curve.update({keys[i]: values[i]})
    return True


def update_graph_values_keys_condition(
    graph: "Graph", *values, keys: Optional[list] = None, also_attr=None, also_vals=None
) -> bool:
    """Similar as update_values_keys, but for all curves inside the provided graph,
    and not on the Graph itself.

    :param graph: Graph object, will apply to all Curves within
    :param args: value1, value2, ...
    :param keys: list of keys ['key1', 'key2', ...]. list[str]
    :param also_attr: list of attribute keys
    :param also_vals: list, same len as also_attr. Test each curve in graph, only perfom
           modifications if curve.attr(also_attr[i]) == also_vals[i], for all i in range
    """
    if keys is None:
        keys = []
    if not isinstance(keys, list) or len(keys) != len(values):
        msg = (
            "update_graph_values_keys_conditional: 'keys' must be a list, "
            "same len as *values. Provided (%s), len %s, expected len %s. Abort."
        )
        raise IncorrectInputError(msg % (keys, len(keys), len(values)))
    for curve in graph:
        flag = True
        if also_attr is not None and also_vals is not None:
            for key, value in zip(also_attr, also_vals):
                if not curve.attr(key) == value:
                    flag = False
        if flag:
            update_values_keys(curve, *values, keys=keys)
    return True
