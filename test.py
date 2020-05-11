# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:17:03 2018

@author: Romain
"""
import sys
import os
import numpy as np
from copy import deepcopy

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if path not in sys.path:
    sys.path.append(path)
from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import roundSignificantRange
from grapa.datatypes.curveJV import CurveJV


file = 'test_ex.txt'
graph = Graph(file)
graph.plot()