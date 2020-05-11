# -*- coding: utf-8 -*-

"""
Created on Fri Jul 15 15:46:13 2016

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np
import warnings
import inspect

from grapa.mathModule import is_number


class Curve:
    
    CURVE = 'Curve'

    NMTOEV = 1239.5
    LINESTYLEHIDE = ['none', 'None', None]

    def __init__ (self, Curve, attributes, silent=True):
        self.silent = silent
        self.data = np.array([])
        self.attributes = {}
        # add data
        if isinstance(Curve, (np.ndarray, np.generic)):
            self.data = Curve
        else:
            self.data = np.array(Curve)
        if self.data.shape[0] < 2 :
            self.data = self.data.reshape((2, len(self.data)))
            print ('WARNING class Curve: data Curve need to contain at least 2 columns')
            print ('   Shape of Curve:', self.data.shape)
        # clean nan pairs at end of data
        iMax = self.data.shape[1]
        for i in range(self.data.shape[1]-1, 0, -1): # loop down to 1 only, not 0
            if np.isnan(self.data[:,i]).all():
                iMax -= 1
            else:
                break
        if iMax < self.data.shape[1]:
            self.data = self.data[:,:iMax]
        # add attributes
        if isinstance(attributes, dict):
            self.update(attributes)
        else:
            print ('ERROR class Curve: attributes need to be of class dict.')

    
    # methods related to GUI
    @classmethod
    def classNameGUI(cls): # can be overridden, see CurveArrhenius
        return cls.CURVE
    
    def funcListGUI(self, **kwargs):
        """
        Fills in the Curve actions specific to the Curve type. Syntax:
        [func,
         'Button text',
         ['label 1', 'label 2', ...],
         ['value 1', 'value 2', ...],
         {'hiddenvar1': 'value1', ...}, (optional)
         [dictFieldAttributes, {}, ...]] (optional)
        By default returns quick modifs for offset and muloffset (if already
        set), and a help for some plot types (errorbar, scatter)
        In principle the current Graph and the position of self in the Graph
        can be accessed through kwargs['graph'] and kwargs['graph_i']
        """
        out = []
        # by default the offset and muloffset can be directly accessed if are already set
        typeplot = self.getAttribute('type')
        try:
            graph = kwargs['graph']
            c = kwargs['graph_i']
            if typeplot == 'errorbar': # assist functions for type errorbar
                out += self.funcListGUI_errorbar(graph, c)
            elif typeplot == 'scatter': # assist function for type scatter
                out += self.funcListGUI_scatter(graph, c)
            elif typeplot.startswith('fill'): # assist function for type fill
                out += self.funcListGUI_fill(graph, c)
            elif typeplot.startswith('boxplot'): # assist function for type boxplot
                out += self.funcListGUI_boxplot(graph, c)
        except Exception as e:
            print('Exception', type(e), 'in Curve.funcListGUI')
        if self.getAttribute('offset', None) is not None or self.getAttribute('muloffset', None) is not None:
            from grapa.graph import Graph
            at = ['offset', 'muloffset']
            values = {'offset': [], 'muloffset': []}
            for key in at:
                for i in range(len(Graph.dataInfoKeysGraphData)):
                    if Graph.dataInfoKeysGraph[i] == key:
                        values.update({key: Graph.dataInfoKeysGraphExalist[i]})
                        break
            out.append([self.updateValuesDictkeys, 'Modify screen offsets',
                        at,
                        [self.getAttribute(a) for a in at],
                        {'keys': at},
                        [{'field':'Combobox','values':values[a]} for a in at]])
        return out
    
    def funcListGUI_errorbar(self, graph, c):
        out = []
        types = []
        if c+1 < graph.length():
            types.append(graph.curve(c+1).getAttribute('type'))
        if c+2 < graph.length():
            types.append(graph.curve(c+2).getAttribute('type'))
        xyerrOR = ''
        if len(types) > 0:
            choices = ['', 'errorbar_xerr', 'errorbar_yerr']
            labels = ['next Curves type: ('+str(c+1)+')']+['('+str(i)+')' for i in range(c+2, c+1+len(types))]
            fieldprops = [{'field': 'Combobox', 'values': choices}] * len(labels)
            out.append([self.updateNextCurvesScatter, 'Save', labels, types, {'graph': graph, 'graph_i': c}, fieldprops])
            xyerrOR = 'OR '
        out.append([self.updateValuesDictkeys, 'Save', [xyerrOR+'error values x', 'y'], [self.getAttribute('xerr'), self.getAttribute('yerr')], {'keys': ['xerr', 'yerr']}])
        at = ['capsize', 'ecolor']
        out.append([self.updateValuesDictkeys, 'Save', at, [self.getAttribute(a) for a in at], {'keys': at}])
        return out
        
    def funcListGUI_scatter(self, graph, c):
        out = []
        types = []
        if c+1 < graph.length():
            types.append(graph.curve(c+1).getAttribute('type'))
        if c+2 < graph.length():
            types.append(graph.curve(c+2).getAttribute('type'))
        if len(types) > 0:
            choices = ['', 'scatter_c', 'scatter_s']
            labels = ['next Curves type: ('+str(c+1)+')']+['('+str(i)+')' for i in range(c+2, c+1+len(types))]
            fieldprops = [{'field':'Combobox', 'values':choices}] * len(labels)
            out.append([self.updateNextCurvesScatter, 'Save', labels, types, {'graph': graph, 'graph_i': c}, fieldprops])
        keys = ['cmap', 'vminmax']
        out.append([self.updateValuesDictkeys, 'Save', keys, [self.getAttribute(k) for k in keys], {'keys': keys}])
        return out
        
    def funcListGUI_fill(self, graph, c):
        out = []
        at = ['fill', 'hatch', 'fill_padto0']
        at2 = list(at)
        at2[2] = 'pad to 0'
        out.append([self.updateValuesDictkeys, 'Save', at2,
                    [self.getAttribute(a) for a in at], {'keys': at},
                    [{'field': 'Combobox', 'values': ['True', 'False']},
                     {'field': 'Combobox', 'values': ['', '.', '+', '/', r'\\'], 'width': 7},
                     {'field': 'Combobox', 'values': ['', 'True', 'False']}]])
        return out
        
    def funcListGUI_boxplot(self, graph, c):
        out = []
        at = ['boxplot_position']
        at2 = ['position']
        out.append([self.updateValuesDictkeys, 'Save', at2,
                    [self.getAttribute(a) for a in at], {'keys': at},
                    [{}]])
        at =  ['widths', 'notch', 'vert']
        out.append([self.updateValuesDictkeysGraph, 'Save', at,
                    [self.getAttribute(a) for a in at],
                    {'keys': at, 'graph': graph, 'alsoAttr': ['type'], 'alsoVals': ['boxplot']},
                    [{},
                     {'field': 'Combobox', 'values': ['True', 'False']},
                     {'field': 'Combobox', 'values': ['True', 'False']}]])
        return out
        
    def updateNextCurvesScatter(self, *values, **kwargs):
        try:
            graph = kwargs['graph']
            c = kwargs['graph_i']
        except KeyError:
            print('KeyError in Curve.updateNextCurvesScatter: "graph" or "graph_i" not provided in kwargs')
        for i in range(len(values)):
            if c+i < graph.length():
                graph.curve(c+1+i).update({'type': values[i]})
        return True
        
    
    def alterListGUI(self):
        """
        Determines the possible curve visualisations. Syntax:
        ['GUI label', ['alter_x', 'alter_y'], 'semilogx']
        By default only Linear (ie. raw data) is provided
        """
        out = []
        out.append(['Linear', ['',''], ''])
        return out
    
    
    # Function related to Fits
    def updateFitParam(self, *param):
        f = self.getAttribute('_fitFunc', '')
        #print ('Curve update', param, 'f', f)
        if f != '' and not self.attributeEqual('_popt'):
            if hasattr(self, f):
                self.setY(getattr(self, f)(self.x(), *param))
                self.update({'_popt': np.array(param)})
                return True
            return 'ERROR Update fit parameter: No such fit function (',f,').'
        return 'ERROR Update fit parameter: Empty parameter (_fitFunc:',f,', _popt:',self.getAttribute('_popt'),').'
    


    # more classical class methods
    def getData(self) :
        return self.data
    
    def shape (self, idx=':') :
        if idx == ':' :
            return self.data.shape
        return self.data.shape[idx]
    
    def _fractionToFloat(self, frac_str):
        try:
            return float(frac_str)
        except ValueError:
            num, denom = frac_str.split('/')
            return float(num) / float(denom)

    def x_offsets(self, **kwargs):
        """
        Same as x(), including the effect of offset and muloffset on output.
        """
        reserved = ['minmax', '0max']
        special = None
        x = self.x(**kwargs)
        offset = self.getAttribute('offset', None)
        if offset is not None:
            o = offset[0] if isinstance(offset, list) else 0
            if isinstance(o, str):
                if o in reserved:
                    special = o
                    o = 0
                else:
                    o = self._fractionToFloat(o)
            x = x + o
        muloffset = self.getAttribute('muloffset', None)
        if muloffset is not None:
            o = muloffset[0] if isinstance(muloffset, list) else 1
            if isinstance(o, str):
                if o.replace(' ','') in reserved:
                    special = o
                    o = 1
                else:
                    o = self._fractionToFloat(o)
            x = x * o
        if special is not None:
            m, M = np.min(x), np.max(x)
            if special == 'minmax':
                x = (x - m) / (M - m)
            elif special == '0max':
                x = x / M
        return x
    
    def x (self, index=np.nan, alter='', xyValue=None, errorIfxyMix=False, neutral=False):
        """
        Returns the x data.
        index: range to the data.
        alter: possible alteration to the data (i.e. '', 'nmeV')
        xyValue: provide [x, y] value pair to be alter-ed. x, y can be np.array.
        errorIfxyMix: throw ValueError exception if alter value calculation requires both x and y components. Useful for transforming xlim and ylim, where x do not know y in advance.
        neutral: no use yet, introduced to keep same call arguments as self.y()
        """
        if alter != '':
            if alter == 'nmeV':
                if xyValue is not None: # invert order of xyValue, to help for graph xlim and ylim
                    xyValue = np.array(xyValue)
                    if len(xyValue.shape) > 1:
                        xyValue = [xyValue[0,::-1], xyValue[1,::-1]]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    return Curve.NMTOEV / self.x(index, xyValue=xyValue)
            elif alter == 'nmcm-1':
                if xyValue is not None: # invert order of xyValue, to help for graph xlim and ylim
                    xyValue = np.array(xyValue)
                    if len(xyValue.shape) > 1:
                        xyValue = [xyValue[0,::-1], xyValue[1,::-1]]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    return 1e7 / self.x(index, xyValue=xyValue)
            elif alter == 'MCAkeV':
                offset = self.getAttribute('_MCA_CtokeV_offset', default=0)
                mult   = self.getAttribute('_MCA_CtokeV_mult',   default=1)
                return mult * (self.x(index, xyValue=xyValue) + offset)
            elif alter == 'SIMSdepth':
                offset = self.getAttribute('_SIMSdepth_offset', default=0)
                mult   = self.getAttribute('_SIMSdepth_mult',   default=1)
                return mult * (self.x(index, xyValue=xyValue) + offset)
            elif alter == 'y':
                try:
                    xyValue = xyValue[::-1]
                except TypeError:
                    pass
                # print('x xyValue', xyValue, self.y(index, xyValue=xyValue))
                return self.y(index, xyValue=xyValue)
            elif alter != '':
                split = alter.split('.')
                if len(split) == 2:
                    module_name = 'grapa.datatypes.' + split[0][0].lower() + split[0][1:]
                    import importlib
                    try:
                        mod = importlib.import_module(module_name)
                        met = getattr(getattr(mod, split[0]), split[1])
                        return met(self, index=index, xyValue=xyValue)
                    except ImportError as e:
                        print('ERROR Curve.x Exception raised during module import', module_name+':')
                        print(e)
                else:
                    print('Error Curve.x: cannot identify alter keyword ('+alter+').')
            
        # alter might be used by subclasses
        val = self.data if xyValue is None else np.array(xyValue)
        if len(val.shape) > 1 :
            if np.isnan(index).any() :
                return val[0,:]
            return val[0,index]
        return val[0]
    
    def y_offsets(self, **kwargs):
        """
        Same as y(), including the effect of offset and muloffset on output.
        """
        reserved = ['minmax', '0max']
        special = None
        y = self.y(**kwargs)
        offset = self.getAttribute('offset', None)
        if offset is not None:
            o = offset[1] if isinstance(offset, list) else offset
            if isinstance(o, str):
                if o in reserved:
                    special = o
                    o = 0
                else:
                    o = self._fractionToFloat(o)
            y = y + o
        muloffset = self.getAttribute('muloffset', None)
        if muloffset is not None:
            o = muloffset[1] if isinstance(muloffset, list) else muloffset
            if isinstance(o, str):
                if o.replace(' ','') in reserved:
                    special = o
                    o = 1
                else:
                    o = self._fractionToFloat(o)
            y = y * o
        if special is not None:
            m, M = np.min(y), np.max(y)
            if special == 'minmax':
                y = (y - m) / (M - m)
            elif special == '0max':
                y = y / M
        return y
        
    def y (self, index=np.nan, alter='', xyValue=None, errorIfxyMix=False, neutral=False) :
        """
        Returns the y data.
        index: range to the data.
        alter: possible alteration to the data (i.e. '', 'log10abs', 'tauc')
        xyValue: provide [x, y] value pair to be alter-ed. x, y can be np.array.
        errorIfxyMix: throw ValueError exception if alter value calculation requires both x and y components. Useful for transforming xlim and ylim, where x do not know y in advance.
        neutral: additional parameter. If True prevent Jsc substraction for CurveJV.
        """
        # if some alteration required: make operation on data requested by itself, without operation
        if alter != '':
            if alter == 'log10abs':
                jsc = self.getAttribute ('Jsc') if self.getAttribute ('Jsc') != '' and not neutral else 0
                return np.log10(np.abs(self.y(index, xyValue=xyValue) + jsc))
            elif alter == 'abs':
                jsc = self.getAttribute ('Jsc') if self.getAttribute ('Jsc') != '' and not neutral else 0
                return np.abs(self.y(index, xyValue=xyValue) + jsc)
            elif alter == 'tauc':
                if errorIfxyMix:
                    return ValueError
                return np.power(self.y(index, xyValue=xyValue) * self.x(index, alter='nmeV', xyValue=xyValue), 2)
            elif alter == 'taucln1-eqe':
                if errorIfxyMix:
                    return ValueError
                return np.power(np.log(1-self.y(index, xyValue=xyValue)) * self.x(index, alter='nmeV', xyValue=xyValue), 2)
            elif alter == 'normalized':
                out = self.y(index, xyValue=xyValue)
                return out / np.max(out)
            elif alter == 'x':
                #print('y xyValue', xyValue[::-1], self.x(index, xyValue=xyValue))
                try:
                    xyValue = xyValue[::-1]
                except TypeError:
                    pass
                return self.x(index, xyValue=xyValue)
            elif alter != '':
                split = alter.split('.')
                if len(split) == 2:
                    import importlib
                    module_name = 'grapa.datatypes.' + split[0][0].lower() + split[0][1:]
                    try:
                        mod = importlib.import_module(module_name)
                        met = getattr(getattr(mod, split[0]), split[1])
                        return met(self, index=index, xyValue=xyValue)
                    except ImportError as e:
                        print('ERROR Curve.y Exception raised during module import', module_name+':')
                        print(e)
                else:
                    if alter != 'idle':
                        print('Error Curve.y: cannot identify alter keyword ('+alter+').')
#        elif alter == '-dCdlnf':
#            from mathModule import derivative
#            return - derivative(np.log(self.x(index, xyValue=xyValue)), self.y(index, xyValue=xyValue))

        # return (subset of) data
        val = self.data if xyValue is None else np.array(xyValue)
        if len(val.shape) > 1 :
            if np.isnan(index).any() :
                return val[1,:]
            return val[1,index]
        return val[1]

    def setX (self, x, index=np.nan) :
        """ Set new value for the x data. Index can be provided (self.data[0,index] = x). """
        if np.isnan(index) :
            self.data[0,:] = x
        else :
            self.data[0,index] = x
            
    def setY (self, y, index=np.nan):
        """ Set new value for the y data. Index can be provided (self.data[1,index] = y). """
        if len(self.shape()) > 1 :
            if np.isnan(index) :
                self.data[1,:] = y
            else :
                self.data[1,index] = y
        else :
            self.data[1] = y
    
    def appendPoints(self, xSeries, ySeries):
        xSeries, ySeries = np.array(xSeries), np.array(ySeries)
        if len(xSeries) == 0:
            print('Curve appendPoints: empty xSeries provided.')
            return False
        if len(ySeries) != len(xSeries):
            print('Curve appendPoints: cannot handle different data series lengths.')
            return False
        self.data = np.append(self.data, np.array([xSeries, ySeries]), axis=1)
        return True
        
    def attr(self, key, default=''):
        """ Shorter alias to getAttribute. """
        return self.getAttribute(key, default=default)
    def getAttribute (self, key, default='') :
        """ Getter of attribute """
        if key.lower() in self.attributes :
            return self.attributes[key.lower()]
        return default
    
    def getAttributes(self, keysList=None):
        """ Returns all attributes, or a subset with keys given in keyList """
        if isinstance(keysList, list):
            out = {}
            for key in keysList:
                out.update({key: self.getAttribute(key)})
            return out
        return self.attributes
        
    def attributeEqual(self, key, value=''): # value='' must be same as the default parameter of method getAttribute
        """
        Check if attribute 'key' has a certain value
        The value can be left empty: check if getAttribute(key) returns the
        default value.
        """
        val = self.getAttribute(key)
        if isinstance(val, type(value)) and val == value:
            return True
        return False
        
    def update(self, attributes):
        for key in attributes :
            k = key.lower()
            if not isinstance(attributes[key], str) or attributes[key] != '':
                self.attributes.update ({k.strip(' =:\t\n').replace('ï»¿',''): attributes[key]})
            elif k in self.attributes :
                del self.attributes[k]

    def updateValuesDictkeys(self, *args, **kwargs):
        """
        Performs update({key1: value1, key2: value2, ...}) with
        arguments value1, value2, ... , (*args) and
        kwargument keys=['key1', 'key2', ...]
        """
        if 'keys' not in kwargs:
            print('Error Curve updateValuesDictkeys: "keys" key must be provided, must be a list of keys corresponding to the values provided in *args.')
            return False
        if len(kwargs['keys']) != len(args):
            print('WARNING Curve updateValuesDictkeys: len of list "keys" argument must match the number of provided values (',len(args),' args provided, ',len(kwargs['keys']),' keys).')
        lenmax = min(len(kwargs['keys']), len(args))
        for i in range(lenmax):
            self.update({kwargs['keys'][i]: args[i]})
        return True

    def updateValuesDictkeysGraph(self, *args, keys=None, graph=None, alsoAttr=None, alsoVals=None, **kwargs):
        """
        similar as updateValuesDictkeys, for all curves inside a given graph
        """
        from grapa.graph import Graph
        if not isinstance(keys, list):
            print('Error Curve updateValuesDictkeysGraph: "keys" key must be provided, must be a list of keys corresponding to the values provided in *args.')
            return False
        if not isinstance(graph, Graph):
            print('Error Curve updateValuesDictkeysGraph: "graph" key must be provided, must be a Graph object.')
            return False
        if not isinstance(alsoAttr, list):
            print('Error Curve updateValuesDictkeysGraph: "alsoAttr" key must be provided, must be a list of attributes.')
            return False
        if not isinstance(alsoVals, list):
            print('Error Curve updateValuesDictkeysGraph: "alsoVals" key must be provided, must be a list of values matching alsoAttr.')
            return False
        while len(alsoVals) < len(alsoAttr):
            alsoVals.append('')
        for curve in graph.iterCurves():
            flag = True
            for i in range(len(alsoAttr)):
                if not curve.getAttribute(alsoAttr[i]) == alsoVals[i]:
                    flag = False
            if flag:
                curve.updateValuesDictkeys(*args, keys=keys)
        return True
            

        
        
    def delete (self, key) :
        out = {}
        if key.lower() in self.attributes :
            out.update({key: self.getAttribute('key')})
            del self.attributes[key.lower()]
        return out
    
    def swapShowHide(self):
        self.update({'linestyle': '' if self.isHidden() else 'none'})
        return True
    def isHidden(self):
        return True if self.getAttribute('linestyle') in Curve.LINESTYLEHIDE else False
        


    def castCurveListGUI(self, onlyDifferent=True):
        curveTypes = [] if onlyDifferent else [Curve]
        curveTypes += Curve.__subclasses__()
        out = []
        for c in curveTypes:
            if not onlyDifferent or not isinstance(self, c):
                name = c.classNameGUI()
                out.append([name, c.__name__, c])
        key = [x[0] for x in out]
        for i in range(len(key)-2, -1, -1):
            if key[i] in key[:i]:
                del key[i]
                del out[i]
        out = [x for (y,x) in sorted(zip(key, out), key=lambda pair: pair[0])] # sort list by name
#        out.append()
        return out

    def castCurve(self, newTypeGUI):
        if newTypeGUI == 'Curve':
            newTypeGUI = 'Curve'
        if type(self).__name__ == newTypeGUI:
            return self
        curveTypes = self.castCurveListGUI()
        for typ_ in curveTypes:
            if newTypeGUI == 'Curve':
                return Curve(self.data, attributes=self.getAttributes(), silent=True)
                break
            if newTypeGUI == typ_[0] or newTypeGUI == typ_[1] or newTypeGUI.replace(' ','') == typ_[1]:
                if isinstance(self, typ_[2]):
                    print ('Curve.castCurve: already that type!')
                else:
                    return typ_[2](self.data, attributes=self.getAttributes(), silent=True)
                break
        print('Curve.castCurve: Cannot understand new type:', newTypeGUI, '(possible types:', [key for key in curveTypes],')')
        return False
        
        
    def selectData(self, xlim=None, ylim=None, alter='', offsets=False, data=None):
        """
        Returns slices of the data self.x(), self.y() which satisfy
        xlim[0] <= x <= xlim[1], and ylim[0] <= y <= ylim[1]
        alter: '', or ['', 'abs']. Affect the test AND the output.
        offset: if True, calls x and y are affected by offset and muloffset
        data [xSeries, ySeries] can be provided. Helps the code of calling
            methods being cleaner. If provided, alter and offsets are ignored.
        """
        if isinstance(alter, str):
            alter = ['', alter]
        # retrieves entry data
        if data is None:
            if offsets:
                x, y = self.x_offsets(alter=alter[0]), self.y_offsets(alter=alter[1])
            else:
                x, y = self.x(alter=alter[0]), self.y(alter=alter[1])
        else:
            x, y = np.array(data[0]), np.array(data[1])
        # identify boundaries
        if xlim is None:
            xlim = [min(x), max(x)]
        if ylim is None:
            ylim = [min(y), max(y)]
        # construct a data mask
        mask = np.ones(len(x), dtype=bool)
        for i in range(len(mask)):
            if x[i] < xlim[0] or x[i] > xlim[1]:
                mask[i] = False
            if y[i] < ylim[0] or y[i] > ylim[1]:
                mask[i] = False
        return x[mask], y[mask]
        
    
    def getPointClosestToXY(self, x, y, alter='', offsets=False):
        """
        Return the data point closest to the x,y values.
        Priority on x, compares y only if equal distance on x
        """
        if isinstance(alter, str):
            alter = ['', alter]
        # select most suitable point based on x
        datax = self.x_offsets(alter=alter[0])
        absm = np.abs(datax - x)
        idx = np.where(absm == np.min(absm))
        if len(idx) == 0:
            idx = np.argmin(absm)
        elif len(idx) == 1:
            idx = idx[0]
        else:
            # len(idx) > 1: select most suitable point based on y
            datay = self.y_offsets(index=idx, alter=alter[1])
            absM = np.abs(datay - y)
            idX = np.where(absM == np.min(absM))
            if len(idX) == 0:
                idx = idx[0]
            elif len(idX) == 1:
                idx = idx[idX[0]]
            else: # equally close in x and y -> returns first datapoint found
                idx = idx[idX[0]]
        idxOut = idx if len(idx) <= 1 else idx[0]
        if offsets:
            return self.x_offsets(index=idx)[0], self.y_offsets(index=idx)[0], idxOut # no alter, but offset for the return value
        return self.x(index=idx)[0], self.y(index=idx)[0], idxOut # no alter, no offsets for the return value
        
    def getDataCustomPickerXY(self, idx, alter='', strDescription=False):
        """
        Given an index in the Curve data, returns a modified data which has
        some sense to the user. Is overloaded in some child Curve classes, see
        for example CurveCf
        If strDescription is True, then returns a string which describes what
        this method is doing.
        """
        if strDescription:
            return '(x,y) data with offset and transform'
        if isinstance(alter, str):
            alter = ['', alter]
        attr = {}
        return self.x_offsets(index=idx, alter=alter[0]), self.y_offsets(index=idx, alter=alter[1]), attr
 
    
    def printHelpFunc(self, func, leadingstrings=None):
        """ prints the docstring of a function """
        if leadingstrings is None:
            leadingstrings = ['- ', '  ']
        a, idx = 0, None
        for line in func.__doc__.split('\n'):
            if len(line) == 0:
                continue
            if idx is None:
                idx = len(line) - len(line.lstrip(' '))
            if len(line) == idx:
                continue
            print(leadingstrings[a] + line[idx:])
            a = 1
        
        
        
    # some arithmetics
    def __add__(self, other, sub=False, interpolate=-1, offsets=False, operator='add'):
        """
        Addition operation, element-wise or interpolated along x data
        interpolate: interpolate, or not the data on the x axis
            0: no interpolation, only consider y() values
            1: interpolation, x are the points contained in self.x() or in other.x()
            2: interpolation, x are the points of self.x(), restricted to min&max of other.x()
            -1: 0 if same y values (gain time), otherwise 1
        With sub=True, performs substraction
        offsets: if True, adds Curves after computing offsets and muloffsets.
            If not adds on the raw data.
        operator: 'add', 'sub', 'mul', 'div'. sub=True is a shortcut for operatore='sub'. Overriden by sub=True argument.
        """
        def op(x, y, operator):
            if operator == 'sub':
                return x - y
            if operator == 'mul':
                return x * y
            if operator == 'div':
                return x / y
            if operator == 'pow':
                return x ** y
            if operator != 'add':
                print('WARNING Curve.__add__: unexpected operator argument(' + operator + ').')
            return x + y
        
        if sub == True:
            operator = 'sub'
        selfx = self.x_offsets if offsets else self.x
        selfy = self.y_offsets if offsets else self.y
        if not isinstance(other, Curve): # add someting/number to a Curve
            out = Curve([selfx(), op(selfy(), other, operator)], self.getAttributes())
            if offsets: # remove offset information if use it during calculation
                out.update({'offset': '', 'muloffset': ''})
            out = out.castCurve(self.classNameGUI())
            return out # cast type
        # curve1 is a Curve
        otherx = other.x_offsets if offsets else other.x
        othery = other.y_offsets if offsets else other.y
        # default mode -1: check if can gain time (avoid interpolating)
        r = range(0, min(len(selfy()), len(othery())))
        if interpolate == -1:
            interpolate = 0 if np.array_equal(selfx(index=r), otherx(index=r)) else 1
        # avoiding interpolation
        if not interpolate:
            le = min(len(selfy()), len(othery()))
            r = range(0, le)
            if le < len(selfy()):
                print('WARNING Curve __add__: Curves not same lengths, clipped result to shortest (',len(selfy()),',',len(othery()),')')
            if not np.array_equal(selfx(index=r), otherx(index=r)):
                print('WARNING Curve __add__ ('+operator+'): Curves not same x axis values. Consider interpolation (interpolate=1).')
            out = Curve([selfx(index=r), op(selfy(index=r), othery(index=r), operator)], other.getAttributes())
            out.update(self.getAttributes())
            if offsets: # remove offset information if use it during calculation
                out.update({'offset': '', 'muloffset': ''})
            out = out.castCurve(self.classNameGUI())
            return out
        else: # not elementwise : interpolate
            from scipy.interpolate import interp1d
            # construct new x -> all x which are in the range of the other curve
            datax = list(selfx())
            if interpolate == 1: # x from both self and other
                xmin = max(min(selfx()), min(otherx()))
                xmax = min(max(selfx()), max(otherx()))
                datax += [x for x in otherx() if x not in datax] # no duplicates
            else: # interpolate 2: copy x from self, restrict to min&max of other
                xmin, xmax = min(otherx()), max(otherx())
            datax = [x for x in datax if x <= xmax and x >= xmin]
            reverse = (selfx(index=0) > selfx(index=1)) if len(selfx()) > 1 else False
            datax.sort(reverse=reverse)
            f0 = interp1d( selfx(),  selfy(), kind=1)
            f1 = interp1d(otherx(), othery(), kind=1)
            datay = [op(f0(x), f1(x), operator) for x in datax]
            out = Curve([datax, datay], other.getAttributes())
            out.update(self.getAttributes())
            if offsets: # remove offset information if use it during calculation
                out.update({'offset': '', 'muloffset': ''})
            out = out.castCurve(self.classNameGUI())
            return out
            
    def __radd__(self, other, **kwargs):
        return self.__add__(other, **kwargs)
    def __sub__(self, other, **kwargs):
        """ substract operation, element-wise or interpolated """
        kwargs.update({'sub': True})
        return self.__add__(other, **kwargs)
    def __rsub__(self, other, **kwargs):
        """ reversed substract operation, element-wise or interpolated """
        kwargs.update({'sub': False, 'operator': 'add'})
        return Curve.__add__(self.__neg__(), other, **kwargs)
    def __mul__(self, other, **kwargs):
        """ multiplication operation, element-wise or interpolated """
        kwargs.update({'operator': 'mul'})
        return self.__add__(other, **kwargs)
    def __rmul__(self, other, **kwargs):
        kwargs.update({'operator': 'mul'})
        return self.__add__(other, **kwargs)
    def __div__(self, other, **kwargs):
        """ division operation, element-wise or interpolated """
        kwargs.update({'operator': 'div'})
        return self.__add__(other, **kwargs)
    def __rdiv__(self, other, **kwargs):
        """ division operation, element-wise or interpolated """
        kwargs.update({'operator': 'mul'})
        return Curve.__add__(self.__invertArithmetic__(), other, **kwargs)
    def __truediv__(self, other, **kwargs):
        """ division operation, element-wise or interpolated """
        kwargs.update({'operator': 'div'})
        return self.__add__(other, **kwargs)
    def __rtruediv__(self, other, **kwargs):
        """ division operation, element-wise or interpolated """
        kwargs.update({'operator': 'mul'})
        return Curve.__add__(self.__invertArithmetic__(), other, **kwargs)
    def __pow__(self, other, **kwargs):
        """ power operation, element-wise or interpolated """
        kwargs.update({'operator': 'pow'})
        return self.__add__(other, **kwargs)

    def __neg__(self, **kwargs):
        out = Curve([self.x(), -self.y()], self.getAttributes())
        out = out.castCurve(self.classNameGUI())
        return out
    def __invertArithmetic__(self, **kwargs):
        out = Curve([self.x(), 1/self.y()], self.getAttributes())
        out = out.castCurve(self.classNameGUI())
        return out
        
        


    def plot(self, ax, graph=None, graph_i=None, type_plot='', ignoreNext=0, boxplot=None, violinplot=None, violinplotkwargs={}):
        """
        plot a Curve on some axis
        graph, graph_i: a graph instance, such that self==graph.curve(graph_i)
            required to properly plot scatter with scatter_c, etc.
        alter: '', or ['nmeV', 'abs']
        type_plot: 'semilogy'
        ignoreNext: int, counter to decide whether the next curves shall not be
            plotted (multi-Curve plotting such as scatter)
        boxplot:
        violinplot:
        violinplotkwargs:
        """
        from grapa.graph import Graph
        handle = None
        # check default arguments
        if boxplot is None:
            boxplot    = {'y':[], 'positions':[], 'labels':[], 'color':[], 'i':0}
        if violinplot is None:
            violinplot = {'y':[], 'positions':[], 'labels':[], 'color':[]}
        if graph is None:
            graph_i = None
        else:
            if graph.curve(graph_i) != self:
                graph_i = None
            if graph_i is None:
                for c in range(graph.length()):
                    if graph.curve(c) == self:
                        graph_i = c
                        break
            if graph_i is None:
                graph = None # self was not found in graph
                print('Warning Curve.plot: Curve not found in provided Graph')
                
                
        # retrieve basic information
        alter = graph._getAlter() if graph is not None else ['', '']
        attr = self.getAttributes()
        linespec = self.getAttribute('linespec')
        # construct dict of keywords based on curves attributes, in a very restrictive way
        # some attributes are commands for plotting, some are just related to the sample, and no obvious way to discriminate between the 2
        fmt = {}
        for key in attr:
            if not isinstance(key, str):
                print(type(key), key, attr[key])
            if (not isinstance(attr[key], str) or attr[key] != '') and key in Graph.dataInfoKeysGraph and key not in ['plot', 'linespec', 'type', 'ax_twinx', 'ax_twiny', 'offset', 'muloffset', 'labelhide', 'colorbar', 'xerr', 'yerr']:
                fmt[key] = attr[key]
        # do not plot curve if was asked not to display it.
        if 'linestyle' in fmt and fmt['linestyle'] in Curve.LINESTYLEHIDE:
            return None, ignoreNext
        # some renaming of kewords, etc
        if 'legend' in fmt:
            fmt['label'] = fmt['legend']
            del fmt['legend']
        if 'cmap' in fmt and not isinstance(fmt['cmap'], str):
             # convert Colorscale into matplotlib cmap
            from grapa.colorscale import Colorscale
            fmt['cmap'] = Colorscale(fmt['cmap']).cmap()
        if 'vminmax' in fmt:
            if isinstance(fmt['vminmax'], list) and len(fmt['vminmax']) > 1:
                if fmt['vminmax'][0] != '' and not np.isnan(fmt['vminmax'][0]) and not np.isinf(fmt['vminmax'][0]):
                    fmt.update({'vmin': fmt['vminmax'][0]})
                if fmt['vminmax'][1] != '' and not np.isnan(fmt['vminmax'][1]) and not np.isinf(fmt['vminmax'][1]):
                    fmt.update({'vmax': fmt['vminmax'][1]})
            del fmt['vminmax']

        # start plotting
        # retrieve data after transform, including of offset and muloffset
        x = self.x_offsets(alter=alter[0])
        y = self.y_offsets(alter=alter[1])
        type_graph = self.getAttribute('type', 'plot')
        if type_plot.endswith(' norm.'):
            type_graph = type_plot[:-6]
            y = y / max(y)
        
        # add keyword arguments which are in the plot method prototypes
        try:
            sig = inspect.signature(getattr(ax, type_graph))
            for key in sig.parameters:
                if key in attr and key not in fmt:
                    fmt.update({key: attr[key]})
        except AttributeError:
            print('Curve.plot: desired plotting method not found ('+type_graph+'). Going for default.')
            pass # for xample 'errorbar_yerr' after suppression of previous Curve 'errorbar'. Will be 'plot' anyway.
        except Exception as e:
            print('Exception in Curve.plot while identifying keyword arguments:')
            print(type(e), e)
        
        if 'labelhide' in attr and attr['labelhide']:
            if 'label' in fmt:
                del fmt['label']

        # No support for the following methods (either 2D data, or complicated to implement):
        #    hlines, vlines, broken_barh, contour, contourf, polar,
        #    pcolor, pcolormesh, streamplot, tricontour, tricontourf,
        #    tripcolor
        # Partial support for:
        #    imgshow
        attrIgnore = ['label', 'plot', 'linespec', 'type', 'ax_twinx', 'ax_twiny', 'offset', 'muloffset', 'labelhide', 'colorbar']
         
        # "simple" plotting methods, with prototype similar to plot()
        if type_graph in ['semilogx', 'semilogy', 'loglog', 'plot_date', 'stem', 'step', 'triplot']:
            handle = getattr(ax, type_graph)(x, y, linespec, **fmt)
        elif type_graph in ['fill']:
            if self.getAttribute('fill_padto0', False):
                handle = ax.fill([x[0]]+list(x)+[x[-1]], [0]+list(y)+[0], linespec, **fmt)
            else:
                handle = ax.fill(x, y, linespec, **fmt)
        # plotting methods not accepting formatting string as 3rd argument
        elif type_graph in ['bar', 'barbs', 'barh', 'cohere', 'csd', 'fill_between', 'fill_betweenx', 'hexbin', 'hist2d', 'quiver', 'xcorr']:
            handle = getattr(ax, type_graph)(x, y, **fmt)
        #  plotting of single vector data
        elif type_graph in ['acorr', 'angle_spectrum', 'eventplot', 'hist', 'magnitude_spectrum', 'phase_spectrum', 'pie', 'psd', 'specgram']:
            # careful with eventplot, the Curve data are modified
            handle = getattr(ax, type_graph)(y, **fmt)
        # a more peculiar plotting
        elif type_graph in ['spy']:
            handle = getattr(ax, type_graph)([x, y], **fmt)
        elif type_graph == 'stackplot':
            # look for next Curves with type == 'stackplot', and same x
            nexty = []
            fmt['labels'], fmt['colors'] = [''], ['']
            if 'label' in fmt:
                fmt['labels'] = ['' if self.getAttribute('labelhide') else fmt['label']]
                del fmt['label']
            if 'color' in fmt:
                fmt['colors'] = [fmt['color']]
                del fmt['color']
            attrIgnore.append('color')
            if graph is not None:
                for j in range(graph_i+1, graph.length()):
                    if graph.curve(j).getAttribute('type') == type_graph and np.array_equal(x, graph.curve(j).x_offsets(alter=alter[0])):
                        ignoreNext += 1
                        if not graph.curve(j).isHidden():
                            nexty.append(graph.curve(j).y_offsets(alter=alter[1]))
                            lbl = graph.curve(j).getAttribute('label')
                            fmt['labels'].append('' if graph.curve(j).getAttribute('labelhide') else lbl)  
                            fmt['colors'].append(graph.curve(j).getAttribute('color'))
                            continue
                    else:
                        break
            if np.all([(c=='') for c in fmt['colors']]):
                del fmt['colors']
            handle = getattr(ax, type_graph)(x, y, *nexty, **fmt)
        elif type_graph == 'errorbar':
            # look for next Curves, maybe xerr/yerr was provided
            if 'xerr' in attr:
                fmt.update({'yerr': attr['xerr']})
            if 'yerr' in attr:
                fmt.update({'yerr': attr['yerr']})
            if graph is not None:
                for j in range(graph_i+1, min(graph_i+3, graph.length())):
                    if len(graph.curve(j).y()) == len(y):
                        typenext = graph.curve(j).getAttribute('type')
                        if typenext not in ['errorbar_xerr', 'errorbar_yerr']:
                            break
                        if typenext == 'errorbar_xerr':
                            fmt.update({'xerr': graph.curve(j).y_offsets()})
                            ignoreNext += 1
                            continue
                        if typenext == 'errorbar_yerr':
                            fmt.update({'yerr': graph.curve(j).y_offsets()})
                            ignoreNext += 1
                            continue
                    break
            handle = ax.errorbar(x, y, fmt=linespec, **fmt)
        elif type_graph == 'scatter':
            convert = {'markersize': 's', 'markeredgewidth': 'linewidths'}
            for key in convert:
                if key in fmt:
                    fmt.update({convert[key]: fmt[key]})
                    del fmt[key]
            try:
                if graph is not None:
                    for j in range(graph_i+1, min(graph_i+3, graph.length())):
                        typenext = graph.curve(j).getAttribute('type')
                        if typenext not in ['scatter_c', 'scatter_s']:
                            break
                        if 's' not in fmt and typenext == 'scatter_s':
                            fmt.update({'s': graph.curve(j).y_offsets(alter=alter[1])})
                            ignoreNext += 1
                            continue
                        elif 'c' not in fmt and (typenext == 'scatter_c' or np.array_equal(x, graph.curve(j).x_offsets(alter=alter[0]))):
                            fmt.update({'c': graph.curve(j).y_offsets(alter=alter[1])})
                            ignoreNext += 1
                            if 'color' in fmt:
                                del fmt['color'] # there cannot be both c and color keywords
                            continue
                        else:
                            break
                handle = ax.scatter(x, y, **fmt)
            except Exception as e:
                print('ERROR! Exception occured in Curve.plot function during scatter.')
                print(type(e), e)
        elif type_graph == 'boxplot':
            if len(y) > 0 and not np.isnan(y).all():
                bxpltpos = self.getAttribute('boxplot_position', None)
                boxplot['y'].append(y[~np.isnan(y)])
                boxplot['positions'].append(boxplot['i'] if bxpltpos is None else bxpltpos)
                boxplot['labels'].append(fmt['label'] if 'label' in fmt else '')
                boxplot['color'].append (fmt['color'] if 'color' in fmt else '')
                for key in ['widths', 'notch', 'vert']:
                    if self.getAttribute(key, None) is not None:
                        boxplot.update({key: self.getAttribute(key)})
                boxplot['i'] += 1
        elif type_graph == 'violinplot':
            if len(y) > 0 and not np.isnan(y).all():
                bxpltpos = self.getAttribute('boxplot_position', None)
                violinplot['y'].append(y[~np.isnan(y)])
                violinplot['positions'].append(boxplot['i'] if bxpltpos is None else bxpltpos)
                violinplot['labels'].append(fmt['label'] if 'label' in fmt else '')
                violinplot['color'].append (fmt['color'] if 'color' in fmt else '')
                if 'showmeans' in attr:
                    violinplotkwargs.update({'showmeans':   attr['showmeans']})
                if 'showmedians' in attr:
                    violinplotkwargs.update({'showmedians': attr['showmedians']})
                if 'showextrema' in attr:
                    violinplotkwargs.update({'showextrema': attr['showextrema']})
                boxplot['i'] += 1
        elif type_graph in ['imshow', 'contour', 'contourf']:
            from grapa.curve_image import Curve_Image
            img, ignoreNext, X, Y = Curve_Image.getImageData(self, graph, graph_i, alter, ignoreNext)
            if 'label' in fmt:
                del fmt['label']
            if type_graph in ['contour', 'contourf']:
                for key in ['corner_mask', 'colors', 'alpha', 'cmap', 'norm', 'vmin', 'vmax', 'levels', 'origin', 'extent', 'locator', 'extend', 'xunits', 'yunits', 'antialiased', 'nchunk', 'linewidths', 'linestyles', 'hatches']:
                    if key in attr and key not in fmt:
                        fmt.update({key: attr[key]})
                # TODO: remove linewidths, linestyles for contourf, hatches for contour
            args = [img]
            if X is not None and Y is not None and type_graph in ['contour', 'contourf']:
                args = [X, Y] +  args
            try:
                handle = getattr(ax, type_graph)(*args, **fmt)
            except Exception as e:
                print('Curve plot', type_graph, 'Exception')
                print(type(e), e)
        else: # default is plot (lin-lin) # also valid if no information is stored, aka returned ''
            handle = ax.plot(x, y, linespec, **fmt)
    
        handles = handle if isinstance(handle, list) else [handle]
        for key in attr:
            if key not in fmt and key not in attrIgnore:
                for h in handles:
                    if hasattr(h, 'set_'+key):
                        try:
                            getattr(h, 'set_'+key)(attr[key])
                        except Exception as e:
                            print('GraphIO Exception during plot kwargs adjustment for key', key, ':', type(e))
                            print(e)
        
        return handle, ignoreNext
        
        