# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:17:03 2018

@author: Romain
"""
import sys
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if path not in sys.path:
    sys.path.append(path)
from grapa.graph import Graph


graph = Graph('examples/PL/rem_Sample1_564_run2_i_TR.Sample.asc')
graph.plot()

