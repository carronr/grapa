# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:30:21 2017

@author: Romain Carron
Copyright (c) 2023, Empa, Laboratory for Thin Films and Photovoltaics, Romain
Carron
"""

# from abc import ABCMeta, abstractmethod


class Observable(object):

    def __init__(self):
        self.observers = []

    def register(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def unregister(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def unregister_all(self):
        if self.observers:
            del self.observers[:]

    def update_observers(self, *args, **kwargs):
        for observer in self.observers:
            if hasattr(observer, 'update'):
                observer.update(*args, **kwargs)
            elif callable(observer):
                observer(*args, **kwargs)
            else:
                print('WARNING Observable.update_observers, dont know what to',
                      'do', observer, 'args:', *args, **kwargs)


# class Observer(object):
#     __metaclass__ = ABCMeta
#
#     @abstractmethod
#     def update(self, *args, **kwargs):
#         pass
#
#
# class ObserverStringVar(Observer):
#     """
#     subclass of an Observer, which contains a StringVar (or Intvar) given as an
#     argument. Can set and get the contact of the Var.
#     """
#
#     def __init__(self, var, valueInit):
#         """ var is a stringvar (or a intvar, not tested yet) """
#         self._var = var
#         self._var.set(valueInit)
#         self._valueInit = valueInit
#
#     def get(self):
#         return self._var.get()
#
#     def set(self, value):
#         return self._var.set(value)
#
#     def update(self, *args, **kwargs):
#         """ simply set the value to that of the first provided arg """
#         for arg in args:
#             self._var.set(arg)
#             return True
#
#     def var(self):
#         return self._var
#
#
# class ObserverStringVarMethodOrKey(ObserverStringVar):
#     def __init__(self, var, valueInit, methodOrKey, valuesDict=None, methodArgs=None, methodKwargs={}):
#         """
#         methodOrKey: a method to call, or an attribute of the object feeded
#             to the updater
#         methodArgs: a list of argument to be feeded to method methodOrKey
#         methodKwargs: a dict of argument to be feeded to method methodOrKey
#         valuesDict: a dict, which keys correspond to the values of the
#             methodOrKey attribute or result of method call.
#         """
#         ObserverStringVar.__init__(self, var, valueInit)
#         self.methodOrKey = methodOrKey
#         self.valuesDict = valuesDict  # expected ['valueIfTrue', valueIfFalse']
#         self.methodArgs = methodArgs if methodArgs is not None else []
#         self.methodKwargs = methodKwargs
#
#     def update(self, *args, **kwargs):
#         """
#         Subclass of an observer, which contains a tkinter variable var
#         By default sets the value of var to valueInit.
#         The update process is as follows.
#         It receives a list of arg. For each arg, it does the following:
#             if arg.methodOrKey() exists, then
#                 value = arg.methodOrKey(*methodArgs, **methodKwargs)
#             else if arg as attribute value, sets value to arg.value
#             esle sets value to valueInit
#
#             if valuesDict is set and valuesDict[value] exists, put that in var
#             otherwise sets the variable to value
#         """
#         for arg in args:
#             # retrieve the value of interest
#             if hasattr(arg, self.methodOrKey) and callable(getattr(arg, self.methodOrKey)):
#                 # if is method of arg
#                 fun = getattr(arg, self.methodOrKey)
#                 try:
#                     value = fun(*self.methodArgs, **self.methodKwargs)
#                 except Exception as e:
#                     print('Exception', type(e), 'when updating ObserverStringVarMethodOrKey: ',
#                           self.methodOrKey,'(', self.methodArgs, self.methodKwargs, ')')
#                     print(e)
#                     value = self._valueInit
#             elif hasattr(arg, 'attr'):
#                 # if can getAttribute(methodOrKey)
#                 value = arg.attr(self.methodOrKey)
#             else:  # else back to default
#                 value = self._valueInit
#             # formatting
#             if isinstance(value, tuple):
#                 value = list(value)
#             if isinstance(value, list):
#                 value = '[' + ', '.join([str(elem) if not isinstance(elem, str) else '\''+elem+'\'' for elem in value]) + ']'
#             if isinstance(value, str):
#                 value = value.replace('\n', '\\n')
#             # update internal stringvar, possibly via a key of valuesDict
#             if isinstance(self.valuesDict, dict) and value in self.valuesDict:
#                 self.set(self.valuesDict[value])
#             else:
#                 self.set(value)
#             return True
