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
#from grapa.mathModule import roundSignificantRange, strToVar, varToStr
from grapa.datatypes.curveJV import CurveJV

import ast
import codecs


# MOJIBAKE_WINDOWS = [u'\xc2\xb5', u'\xc2\xb1', u'\xc2\xb0', u'\xc3\xa9', u'\xc3\xb6']
MOJIBAKE_WINDOWS = [u'\xc2', u'\xc3']


"""
def strToVar(val):
    # val = codecs.getdecoder("unicode_escape")(val)[0]
    flagast = False
    try:
        val = float(val)
    except Exception:
        try:
            val = ast.literal_eval(val)
            flagast = True
        except Exception:
            if isinstance(val, str) and len(val) > 1:
                if ((val[0] == '[' and val[-1] == ']')
                        or (val[0] == '(' and val[-1] == ')')):
                    try:
                        val = [float(v) for v in val[1:-1].replace(' ', '').replace('np.inf', 'inf').split(',')]
                    except Exception:
                        pass
    # print('strToVar', val, flagast)
    if not flagast:  # not needed when created through ast.literal_eval
        val = strUnescapeIter(val)
    # print('   ', val)
    return val


def strUnescapeIter(var):
    if isinstance(var, list):
        for i in range(len(var)):
            var[i] = strUnescapeIter(var[i])
    elif isinstance(var, dict):
        for key in var:
            var[key] = strUnescapeIter(var[key])
    elif isinstance(var, str):
        varold = var
        var = codecs.getdecoder("unicode_escape")(var)[0]
        for char in ESCAPEWARNING:
            if char in var:
                if len(char) == 1 or '$' in var:
                    print('WARNING strToVar: possible missing escape \'\\\'',
                          'character, detected '+repr(char)+' ('+str(char)+')',
                          'in input \''+varold+'\'.')
    return var

def varToStr(val):
    # return codecs.getencoder("unicode_escape")(out)[0].decode('utf-8')
    out = repr(val).strip("'")
    return out
"""

"""
flag = False
for char in MOJIBAKE_WINDOWS:
    if char in val2:
        print('Relevant mojibake detected, convert from latin-1', char)
        flag = True
if flag and True:
    val2 = val2.encode('latin-1').decode('utf-8')
    #val2 = val2.encode('latin-1')
    #val2 = codecs.getdecoder("unicode_escape")(val.encode('latin-1'))[0]

for v in val2:
    print(codecs.getencoder("unicode_escape")(v)[0])

flag = False
for char in MOJIBAKE_WINDOWS:
    if char in val2:
        print('round 2, Relevant mojibake detected, convert from latin-1', char)
        flag = True
if flag and True:
    val2 = val2.encode('latin-1').decode('utf-8')
"""
# val2 = val.replace('\\n', chr(92))
# val2 = codecs.getdecoder("unicode_escape")(val)[0]
# val2 = codecs.getdecoder("unicode_escape")(val)[0]
# val2 = ast.literal_eval(val)
# val2 = val.encode('utf-8').decode('unicode-escape')
# val2 = val.decode('unicode-escape')
# val2 = val.encode("ascii", "ignore").decode("utf-8")
# val = val.encode('latin1').decode('utf8').encode('latin1').decode('utf8')


def strToVar(val):
    # validates inputs such as '$\\alpha$\nµ' ($\ alpha$ carriage return mu)
    try:
        out = codecs.getdecoder("unicode_escape")(val)[0]
    except Exception as e:
        print('Exception in function strToVar', type(e), e)
        print('Input:', val)
        out = val
    try:
        # handling of special characters transformed into mojibake
        # The 2-pass code below can clean inputs with special characters
        # encoded in:
        # ANSI (e.g. Windows-1252), UTF-8, UTF-8 escaped (eg. \xb5 for mu)
        # First pass
        for char in MOJIBAKE_WINDOWS:
            if char in out:
                print('Suspicion of mojibake (1), latin-1 - utf-8 conversion')
                out = out.encode('latin-1').decode('utf-8')
                break
        # Second pass. Appears necessary with some charset input
        for char in MOJIBAKE_WINDOWS:
            if char in out:
                print('Suspicion of mojibake (2), latin-1 - utf-8 conversion')
                out = out.encode('latin-1').decode('utf-8')
                break
    except Exception as e:
        print('Exception in function strToVar (mojibake)', type(e), e)
        print('Possibly, mix of special characters and escape sequences in',
              'same input')
        print('Input:', val)
        # keep current out value
    return out

def varToStr(val):
    try:
        out = repr(val).strip("'")
    except Exception as e:
        print('varToStr Exception', type(e), e)
        print('Input:', val)
        out = ''
    return out

        # val = codecs.getencoder("unicode_escape")(val)[0].decode('utf-8')
        # val = codecs.getencoder("unicode_escape")(val)[0].decode('utf-8').encode('latin-1').decode('utf-8')
        # val = val.encode('unicode-escape').decode('utf-8')
        # val = val.encode('utf-8')

with open("test2_input.txt", "r") as f:
    lines = f.readlines()
text = '   '.join(lines)

print(text)
text = strToVar(text)
text = varToStr(text)
print(text)
text = strToVar(text)
text = varToStr(text)
print(text)

import matplotlib.pyplot as plt
graph = Graph()
graph.append(Curve([[0,1], [2,3]], {'label': ''}))
graph.update({'xlabel': strToVar(text), 'subplots_adjust': [0.45, 0.45, 0.95, 0.95]})
graph.update({'ylabel': strToVar('   '.join(lines)), 'subplots_adjust': [0.45, 0.45, 0.95, 0.95]})
graph.plot()

with open("test2_output.txt", "w") as f:
    f.write(text)
print('end')

plt.show()

# #text = r'-$\\alpha\alpha$-\n-$\\omega$-$\omega$-\t-$\\theta$-$\theta$-\n-$\\nu$-$\nu$-' # NOT OK
# text = r"['-$\\alpha\alpha$-\n-$\\omega$-$\omega$-\t-$\\theta$-$\theta$', '$\\nu$-$\nu$-', 1]"  # NOT OK
# text = r"['a$\alpha$', 1]"  # NOT OK
# # print(type(text), text)
# # text = strToVar(text)
# # print(type(text), text)
# # text = varToStr(text)
# # print(type(text), text)
#
# text = ['a$\\alpha\n\t\omega\\nu$', 1]
# print(type(text), text)
# text = repr(text)
# print(type(text), text)
# text = strToVar(text)
# print(type(text), text)
