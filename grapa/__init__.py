# -*- coding: utf-8 -*-
"""
Grapa

@author: Romain Carron
Copyright (c) 2026 Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""


__version__ = "0.8.0.2rc1"
__author__ = "Romain Carron"

import logging

logger = logging.getLogger(__name__)


# Import selected Curve subtypes to make sure they will be registered when calling
# __subclasses__. The other subtypes in .datatypes will be imported by Graph.
from grapa.curve_inset import Curve_Inset  # make sure ready when call __subclasses__
from grapa.curve_subplot import Curve_Subplot  # ake sure ready when call __subclasses__
from grapa.curve_image import Curve_Image  # make sure ready when call __subclasses__


# imports for convenience to packages outside the scope of grapa
from grapa.graph import Graph
from grapa.curve import Curve
from grapa.shared.string_manipulations import strToVar


def grapa():
    """Start grapa GUI"""
    from grapa.GUI import build_ui

    build_ui()
