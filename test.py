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


file = r'C:\Users\Romain\Desktop\TOIMPLEMENT\2__Constant Current 20190918 115606.csv'
#file = r'C:\Users\Romain\Desktop\TOIMPLEMENT\3_LCO OCP-EIS-CV0_1-EIS-OCV_Cyclic Voltammetry Start 20190911 101750.csv'
#file = r'C:\Users\Romain\Desktop\TOIMPLEMENT\7_LCO OCP-EIS-CV0_1-EIS-OCV_Potentiostatic EIS 20190912 104832.csv'
#file = r'C:\Users\Romain\Desktop\TOIMPLEMENT\8_LCO OCP-EIS-CV0_1-EIS-OCV_Open Circuit Potential 20190912 104933.csv'
graph = Graph(file)
graph.plot()

"""
def testCurve(curve):
    print('=== Test Curve')
    print(curve.diodeFit_BestGuess())
    fit = curve.CurveJVFromFit_print([-0.5, np.inf], 5)
    graph.append(fit)
    print('=== End Curve')


graph = Graph('examples/JV/SAMPLE_A/I-V_SAMPLE_A_a2_01.txt')
curve = graph.curve(0)
#print(curve.diodeFit_BestGuess())
testCurve(curve)


curve = CurveJV([curve.x()*1000, curve.y()], deepcopy(curve.getAttributes()))
curve.update({'label': 'mult', 'linewidth':5})
graph.append(curve)
testCurve(curve)


graph.update({'ylim':[-50, 50]})
graph.plot()
graph.update({'alter':['', 'log10abs']})
graph.plot()
"""