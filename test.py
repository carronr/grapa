# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:17:03 2018

@author: Romain
"""
import sys
import os
import numpy as np

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if path not in sys.path:
    sys.path.append(path)
from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import roundSignificantRange





graph = Graph('examples/EQE/SAMPLE_A_d1_1.sr')
res = graph.curve(0).CurveEQE_bandgapLog_print([30.0, 88.0])
nm = graph.curve(0).x()


bandgap = [1.1811625274956468, 43.884021117356845]
from grapa.datatypes.curveEQE import CurveEQE


z = [bandgap[1], -bandgap[0]*bandgap[1]]
p = np.poly1d(z)
fit = p(Curve.NMTOEV / nm)
print(list(fit))
for i in range(len(fit)):
    if fit[i] >=0:
        fit[i] = 1 - np.exp(- np.sqrt(fit[i]) / (Curve.NMTOEV/nm[i]))
    else:
        fit[i] = np.nan
print(list(fit))


graph.append(res)
print(res.getData())

graph.update({'xlim':'', 'ylim':''})
graph.plot()
