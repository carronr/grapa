# -*- coding: utf-8 -*-
"""
@author: Romain
"""

import numpy as np
from grapa.mathModule import listToString


class FuncGUI():
    """
    A class to rationalize the following:
    out.append([self.currentCalc, 'EQE current',
        ['ROI', 'interpolate', 'spectrum'],
        [[min(self.x()), max(self.x())], 'linear', fnames[0]],
        {},
        [{'width': 15},
         {'width': 8, 'field': 'Combobox', 'values': ['linear', 'quadratic']},
         {'width': 8, 'field': 'Combobox', 'values': fnames}]])
    Makes it:
    line = FuncGUI(self.currentCalc, 'EQE current')
    line.append('ROI', [min(self.x()), max(self.x())], {'width': 15})
    line.append('interpolate', 'linear', {'width': 8, 'field': 'Combobox',
                                          'values': ['linear', 'quadratic']})
    line.append('spectrum', fnames[0], {'width': 8, 'field': 'Combobox',
                                        'values': fnames})
    out.append(line)
    """

    def __init__(self, func, textsave, hiddenvars=None):
        """
        In principe, create FuncGUI with func, textsave, hiddenvars,
        then create fields with .append() one after the otherwise
        may also do FuncGUI(None, None).initLegacy(oldformat)
        """
        # store information
        self.func = func
        self.textsave = textsave
        self.hiddenvars = {}
        if isinstance(hiddenvars, dict):
            self.hiddenvars = hiddenvars
        elif hiddenvars is not None:
            print('WARNING FuncGUIParams, hiddenvars must be a dict, will be ',
                  'ignored,', hiddenvars)
        self.fields = []

    def initLegacy(self, input):
        self.func = input[0]
        self.textsave = input[1]
        if len(input) > 4 and isinstance(input[4], dict):
            self.hiddenvars = input[4]
        for i in range(len(input[2])):
            label = input[2][i]
            value = input[3][i] if len(input) > 3 else ''
            if isinstance(value, (list, np.ndarray)):
                value = listToString(value)
            options = input[5][i] if len(input) > 5 else {}
            self.append(label, value, options=options)
        return self

    def append(self, label, value, widgetclass='Entry', bind=None, keyword=None, options=None):
        """
        Example
        label: 'my choice'
        value: 'b'
        widgetclass: 'Entry', 'Combobox'... a tk widget (or other library if
            implemented)
        keyword: for the parameter to be submitted as keyword variable
        bind: a (validation) function on the widget value
        options: {'width': 8, 'values': ['a', 'b', 'c']}
        """
        if options is None:
            options = {}
        options = dict(options)  # work on a copy as would modify the content
        # _propsFromOptions can modify options
        cls, bind, kw = self._optsToProps(options, widgetclass, bind, keyword)
        self.fields.append({'label': label, 'value': value, 'options': options,
                            'widgetclass': cls, 'bind': bind, 'keyword': kw})

    def _optsToProps(self, options, widgetclass, bind, keyword):
        # to be called as
        # widgetclass, bind, keyword = self._optsToProps(options, widgetclass,
        #                                                bind, keyword)
        if 'field' in options:  # overrides provided value
            widgetclass = options['field']
            del options['field']
        if 'bind' in options:
            bind = options['bind']
            del options['bind']
        if 'keyword' in options:
            keyword = options['keyword']
            del options['keyword']
        # some futerh checks
        if widgetclass == 'Combobox':
            if 'width' not in options:
                width = int(1.1 * max([len(v) for v in options['values']]))
                options.update({'width': width})
        return widgetclass, bind, keyword

    def isSimilar(self, other):
        """
        Determines if 2 FuncGUI objects are similar
        Checks for function name, textsave, hidden vars, fields labels
        Does NOT check for field values - user force same value for all
        """
        if (isinstance(other, FuncGUI)
                and self.func.__name__ == other.func.__name__
                and self.textsave == other.textsave
                and len(self.fields) == len(other.fields)
                and len(self.hiddenvars) == len(other.hiddenvars)):
            for i in range(len(self.fields)):
                if self.fields[i]['label'] != other.fields[i]['label']:
                    return False
            return True
        return False

    def create_widgets(self, frame, callback, callbackarg):
        """
        Creates a frame and widgets inside. Returns:
        callinfo: [func, StringVar1, StringVar2, ..., {hiddenvars}]
        widgets: [innerFrame, widget1, widget2, ...]
        frame: where the widgets are created. substructure will be provided
        callback: function to call when user validates his input
        callbackarg: index of guiAction. callback(callbackarg)
        """
        import tkinter as tk
        from tkinter import ttk
        callinfo = {'func': self.func, 'args': [],
                    'kwargs': dict(self.hiddenvars)}
        widgets = []  # list of widgets, to be able to destroy later
        # create inner frame
        fr = tk.Frame(frame)
        fr.pack(side='top', anchor='w', fill=tk.X)
        widgets.append(fr)
        # validation button
        if self.func is None:
            widget = tk.Label(fr, text=self.textsave)
        else:
            widget = tk.Button(fr, text=self.textsave,
                               command=lambda j_=callbackarg: callback(j_))
        widget.pack(side='left', anchor='w')
        widgets.append(widget)
        # list of widgets
        for field in self.fields:
            bind = field['bind']
            options = dict(field['options'])
            widgetclass = field['widgetclass']
            # widgetname: tranform into reference to class
            try:
                if widgetclass in ['Combobox']:  # Combobox
                    widgetclass = getattr(ttk, widgetclass)
                else:
                    widgetclass = getattr(tk, widgetclass)
            except Exception as e:
                print('ERROR FuncGUI.create_widgets, cannot create widget of',
                      'class', widgetclass, 'Exception', type(e), e)
                continue
            # Frame: interpreted as to create a new line
            if widgetclass == tk.Frame:
                fr = tk.Frame(frame)
                fr.pack(side='top', anchor='w', fill=tk.X)
                widgets.append(fr)  # inner Frame
                continue  # stop there, go to next widget
            # first, a Label widget for to help the user
            widget = tk.Label(fr, text=field['label'])
            widget.pack(side='left', anchor='w')
            widgets.append(widget)
            # create stringvar
            stringvar = tk.StringVar()
            stringvar.set(str(field['value']))
            if field['keyword'] is None:
                callinfo['args'].append(stringvar)
            else:
                callinfo['kwargs'].update({field['keyword']: stringvar})
            # default size if widgetclass is Entry
            if widgetclass == tk.Entry and 'width' not in options:
                width = int(max(8, (40 - len(field['label'])/3 - len(self.textsave)/2)/len(self.fields)))
                if len(stringvar.get()) < 2:
                    width = int(0.3 * width + 0.7)
                options.update({'width': width})
            # link to StringVar
            if widgetclass == tk.Checkbutton:
                options.update({'variable': stringvar})
            else:
                options.update({'textvariable': stringvar})
            # create widget
            try:
                widget = widgetclass(fr, **options)
            except Exception as e:
                print('Exception', type(e), e)
                print('Could not create widget', field['label'],
                      widgetclass.__name__, options)
                continue
            widget.pack(side='left', anchor='w')
            widgets.append(widget)
            # bind
            bind = field['bind']
            if bind is not None:
                if bind == 'beforespace':
                    bind = lambda event: event.widget.set(event.widget.get().split(' ')[0])
                widget.bind('<<ComboboxSelected>>', bind)
            widget.bind('<Return>', lambda event, j_=callbackarg: callback(j_))
        # end of loop, return
        return callinfo, widgets
