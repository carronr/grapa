# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:08:04 2016

@author: car
Copyright (c) 2018, Empa, Romain Carron
"""

#print ('matplotlib 4')
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

folder = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if folder not in sys.path:
    sys.path.append(folder)
from grapa.graph import Graph



def isSimilar(a, b, tolRel=1e-5):
    if np.max(np.abs((np.array(a) - np.array(b)) / np.array(b))) < tolRel:
        return True
    print('Relative difference', (np.array(a) - np.array(b))/np.array(b))
    return False
def test(result, expected, tolRel, description):
    if isSimilar(result, expected, tolRel=tolRel):
        print('    OK', description)
        return True
    print('    ERROR', description, result, 'expected', expected)
    return False

        
def testJscVoc(graphs=True):
    ret = True
    from grapa.datatypes.curveJscVoc import GraphJscVoc
    graph = Graph('JscVoc/JscVoc_SAMPLE_a3_Values.txt', silent=True)
    out = GraphJscVoc.CurvesJscVocSplitTemperature(graph, threshold=3, curve=graph.curve(0))
    for c in out:
        c.update({'linespec': '.'})
    graph.append(out)
    
    ret *= test(graph.curve(-1).fit_nJ0(), [4.55823, 0.002558908], tolRel=1e-4, description='JscVoc fit single T (1)')
    
    out = GraphJscVoc.CurveJscVoc_fitNJ0(graph, curve=graph.curve(-1), silent=True, Jsclim=[0.5,np.inf])
    ret *= test(out[0].getAttribute('_popt'), [1.745988, 4.0931023e-8], tolRel=1e-4, description='JscVoc fit single T (2)')
    
    out = GraphJscVoc.CurveJscVoc_fitNJ0(graph, Jsclim=[0.5,np.inf], curve=graph.curve(0), graphsnJ0=False, silent=True)
    ret *= test(out[-1].getAttribute('_popt'), [1.745988, 4.0931023e-8], tolRel=1e-4, description='JscVoc fit multiple T')
    
    out = GraphJscVoc.CurveJscVoc_fitNJ0(graph, Jsclim=[1,np.inf], curve=graph.curve(0), graphsnJ0=True, silent=True)
    graph.append(out)
    if graphs:
        graph.plot(ifSave=False, ifExport=False)
    for c in graph.iterCurves():
        c.swapShowHide()
    graph.curve(-2).swapShowHide()
    graph.curve(-3).swapShowHide()
    graph.update({'typeplot': ''})
    graph.update({'alter': ['CurveArrhenius.x_1000overK', 'CurveArrhenius.y_Arrhenius']})
    #graph.update({'xlim': [np.inf, 350], 'ylim': [-30,10]})
    out = graph.curve(-1).CurveArrhenius_fit([-np.inf, np.inf], silent=True)
    graph.append(out)
    ret *= test(out.getAttribute('_popt'), [1.4993444970660568, 656784.49095749843], tolRel=1e-4, description='JscVoc J0 Arrhenius')
    if graphs:
        graph.plot(ifSave=False, ifExport=False)
    return ret
    

def testEQEAdd(graphs=True):
    ret = True
    graph = Graph('./EQE/SAMPLE_A_d1_1.sr', silent=True)
    target = 32.9679429444
    ret *= test(graph.curve(0).currentCalc(silent=True), target, tolRel=1e-4, description='EQE current')
    if graphs:
        graph.plot(ifSave=False, ifExport=False)
    return ret

    




def testAll():
    graphs = False
    
    # run various tests
    out = testJscVoc(graphs=graphs)
    print('test JscVoc: ', 'passed' if out else 'FAILED')

    out = testEQEAdd(graphs=graphs)
    print('test EQE: ', 'passed' if out else 'FAILED')

    
    # all scripts can be check independently
    # -> JV, CV, Cf, EQE ok
    # TODO: Spectrum, TRPL, SIMS, MCA
    # ARRHENIUS done in JscVoc
    
    plt.show()
    
    
if __name__ == "__main__":
    testAll()
