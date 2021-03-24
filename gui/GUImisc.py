# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:00:36 2017

@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""


from tkinter import ttk
import tkinter as tk
from tkinter import X, BOTH


def bind_tree(widget, event, callback, add=''):
    "Binds an event to a widget and all its descendants."
    widget.bind(event, callback, add)
    for child in widget.children.values():
        bind_tree(child, event, callback) # , replace_callback

        
class OptionMenuVar(tk.OptionMenu):
    """ replacement for tk.OptionMenu, with embedded tk.Stringvar """
    def __init__(self, frame, values, default='', func=None, width=None):
        self.values = values
        self.var = tk.StringVar()
        tk.OptionMenu.__init__(self, frame, self.var, self.values)
        self.resetValues(self.values, default=default, func=func)
        if width is not None:
            self.configure(width=width)
    def get(self):
        return self.var.get()
    def set(self, val):
        if val in self.values:
            self.var.set(val)
    def resetValues(self, values, func=None, default=''):
        self.values = values
        self['menu'].delete(0, 'end')
        for val in values:
            if func is not None:
                self['menu'].add_command(label=val, command=lambda v=val: func(v))
            else:
                self['menu'].add_command(label=val, command=tk._setit(self.var, val))
        self.var.set(default if default in self.values else self.values[0])


class EntryVar(tk.Entry):
    """ replacement for tk.Entry, with embedded tk.Stringvar """
    def __init__(self, frame, value, **kwargs):
        self.var = tk.StringVar()
        self.var.set(value)
        tk.Entry.__init__(self, frame, textvariable=self.var, **kwargs)
    def get(self):
        return self.var.get()
    def set(self, val):
        self.var.set(val)

        
class LabelVar(tk.Label):
    """ replacement for tk.Label, with embedded tk.Stringvar """
    def __init__(self, frame, value, **kwargs):
        self.var = tk.StringVar()
        self.var.set(value)
        tk.Label.__init__(self, frame, textvariable=self.var, **kwargs)
    def get(self):
        return self.var.get()
    def set(self, val):
        self.var.set(val)

        
class ComboboxVar(ttk.Combobox):
    """ replacement for tk.Combobbox, with embedded tk.Stringvar """
    def __init__(self, frame, values, default='', **kwargs):
        self.values = values
        self.var = tk.StringVar()
        self.var.set(default)
        ttk.Combobox.__init__(self, frame, values=self.values, textvariable=self.var, **kwargs)
    def get(self):
        return self.var.get()
    def set(self, val):
        self.var.set(val)



class CheckbuttonVar(tk.Checkbutton):
    """ replacement for tk.Checkbutton, with embedded tk.BooleanVar """
    def __init__(self, frame, text, default, **kwargs):
        self.var = tk.BooleanVar()
        self.var.set(default)
        tk.Checkbutton.__init__(self, frame, text=text, variable=self.var, **kwargs)
    def get(self):
        return self.var.get()
    def set(self, value):
        self.var.set(value)

        
        
class ButtonSmall(tk.Frame):
    """ fabricate a Button with a tunable size """
    def __init__(self, frame, text, command, width=16, height=16, **kwargs):
        tk.Frame.__init__(self, frame, width=width, height=height)
        self.propagate(0)
        self._button = tk.Button(self, text=text, command=command, **kwargs)
        self._button.pack(fill=tk.BOTH, expand=True)
        
        
        
        
class FrameTitleContentHide(tk.Frame):
    """ A class to show and hide some content in the user-interface """
    def __init__(self, master, funcFillTitle, funcFillContent,
                 contentkwargs={}, default='show', layout='vertical',
                 showHideTitle=False, 
                 **kwargs):
        self.setButtonLabels()
        tk.Frame.__init__(self, master, **kwargs)
        self._visiblecontent = True
        self.showHideTitle = showHideTitle
        self._layout = layout
        self._contentkwargs = contentkwargs
        # define elements inside
        self._title = None
        self.createWidgets()
        # fill title and content
        if funcFillTitle is not None and self._title is not None:
            funcFillTitle(self._title)
        funcFillContent(self._content)
        if self._title is not None:
            bind_tree(self._title, '<Button-1>', self.showHide)
        if default == 'hide':
            self.showHide()
    @classmethod
    def frameHline(cls, frame):
        return tk.Frame(frame, height=3, background='gainsboro')
    def setButtonLabels(self):
        self.btnlbl_in = u"\u25B3"
        self.btnlbl_out = u"\u25BC"
    def createWidgets(self):
        up = tk.Frame(self)
        dwn= tk.Frame(self)
        up.pack (side='top', anchor='w', fill=X)
        dwn.pack(side='top', anchor='w', fill=X)
        self._title   = tk.Frame(up)
        self._buttonFrame = tk.Frame(up, width=20, height=20)
        self._buttonFrame.propagate(0)
        self._button  = tk.Button(self._buttonFrame, text=self.btnlbl_in,
                                  command=self.showHide)
        self._horizLineFrame = self.frameHline(up)
        self._dummy = tk.Frame(dwn)
        self._content = tk.Frame(dwn, **self._contentkwargs)
        self._title.pack(side='left', anchor='n')
        self._buttonFrame.pack(side='left', anchor='center', padx=5)
        self._horizLineFrame.pack(side='left', anchor='center', fill=X, expand=1, padx=5)
        self._button.pack(side='left', anchor='n', fill=BOTH, expand=1)
        self._dummy.pack(side='top', anchor='w', fill=X)
        self._content.pack(side='top', anchor='w', fill=X)
    def getFrameTitle(self):
        return self._title
    def getFrameContent(self):
        return self._content
    def getButton(self):
        return self._button
    def isvisible(self):
        return self._visiblecontent
    def showHide(self, *args):
        side = 'left' if self._layout == 'horizontal' else 'top'
        anchor = 'center' if self._layout == 'horizontal' else 'w'
        if self._visiblecontent:
            self._content.pack_forget()
            if self.showHideTitle:
                self._title.pack(side=side, anchor=anchor, fill=X)
            self._button.configure({'text': self.btnlbl_out})
        else:
            self._content.pack(side=side, anchor=anchor, fill=X)
            if self.showHideTitle:
                self._title.pack_forget()
            self._button.configure({'text': self.btnlbl_in})
        self._visiblecontent = not self._visiblecontent
            
class FrameTitleContentHideHorizontal(FrameTitleContentHide):
    def setButtonLabels(self):
        self.btnlbl_in  = u"\u25C1"
        self.btnlbl_out = u"\u25B6"
    def createWidgets(self):
        left = tk.Frame(self, height=60, width=30)
        righ = tk.Frame(self)
        left.pack(side='left', anchor='w')
        righ.pack(side='left', anchor='w', fill=X)
        self._buttonFrame = tk.Frame(left, width=20, height=20)
        self._buttonFrame.pack(side='left', anchor='center', padx=5)
        self._buttonFrame.propagate(0)
        left.propagate(0)
        self._title   = tk.Frame(righ)
        self._title.pack(side='left', anchor='center')
        self._dummy = tk.Frame(righ)
        self._dummy.pack(side='left', anchor='n', fill=X)
        self._button  = tk.Button(self._buttonFrame, text=u"\u25B3",
                                  command=self.showHide)
        self._button.pack(side='left', anchor='n', fill=BOTH, expand=1)
        self._content = tk.Frame(righ, **self._contentkwargs)
        self._content.pack(side='top', anchor='w', fill=X)

  

        
def imageToClipboard(graph):
    """ copy the image output of a Graph to the clipboard - Windows only """
    # save image, because we don't have pixel map at the moment
    print('Copying graph image to clipboard')
    selffilename = graph.filename if hasattr(graph, 'filename') else None
    fileClipboard = '_grapatoclipboard'
    tmp = graph.getAttribute('saveSilent')
    graph.update({'saveSilent': True})
    graph.plot(ifSave=True, ifExport=False, filesave=fileClipboard)
    graph.update({'saveSilent': tmp})
    if selffilename is not None: # restore self.filename
        graph.filename = selffilename
    
    from io import BytesIO
    from PIL import Image
    try:
        import win32clipboard
    except ImportError as e:
        print('Module win32clipboard not found, cannot copy to clipboard. Image was created.')
        print(e)
        return False
    def send_to_clipboard(clip_type, data):
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(clip_type, data)
        win32clipboard.CloseClipboard()
    # read image -> get pixel map, convert into clipbaord-readable format
    output = BytesIO()
    img = Image.open(fileClipboard+'.png')
    try:
        img = img.convert('RGBA')
    except KeyError:
        print("Copy to clipboard: RGBA not found. More recent version PIL versions should be able though. Transparency won't be handled correctly.")
        img = img.convert('RGB')
    try:
        img.save(output, 'BMP')
    except (IOError, OSError):
        print("Copy to clipboard: cannot write image as BMP. More recent version PIL versions should be able though. Transparency won't be handled correctly.")
        img = img.convert('RGB')
        img.save(output, 'BMP') # second try
    data = output.getvalue()[14:]
    output.close()
    send_to_clipboard(win32clipboard.CF_DIB, data)
    return True



