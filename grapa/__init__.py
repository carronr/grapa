# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:50:26 2017

@author: Romain Carron
Copyright (c) 2024, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

__version__ = "0.6.4.0"

from grapa.graph import Graph
from grapa.curve import Curve


def grapa():
    from grapa.GUI import buildUI

    buildUI()
