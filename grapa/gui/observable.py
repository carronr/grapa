# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:30:21 2017

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
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
            if hasattr(observer, "update"):
                observer.update(*args, **kwargs)
            elif callable(observer):
                observer(*args, **kwargs)
            else:
                print(
                    "WARNING Observable.update_observers, dont know what to",
                    "do",
                    observer,
                    "args:",
                    *args,
                    **kwargs
                )
