# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2018, Empa, Laboratory for Thin Films and Photovoltaics,
Romain Carron
"""


from tkinter import ttk
import tkinter as tk
from tkinter import X, BOTH


def bind_tree(widget, event, callback, add=""):
    "Binds an event to a widget and all its descendants."
    widget.bind(event, callback, add)
    for child in widget.children.values():
        bind_tree(child, event, callback)  # , replace_callback


class TextWriteable(tk.Text):
    """
    New texyt field class, with possibility for write (print in this field)
    Used for the console output
    """

    def write(self, string):
        self.insert(tk.END, string)
        self.see(tk.END)


class OptionMenuVar(tk.OptionMenu):
    """
    Replacement for tk.OptionMenu, with embedded tk.Stringvar
    func: function that will be called as command()
    varType: alternative to tk.Stringvar
    """

    def __init__(self, frame, values, default="", func=None, width=None, varType=None):
        self.values = values
        if varType is None:
            self.var = tk.StringVar()
        else:
            self.var = varType()
        tk.OptionMenu.__init__(self, frame, self.var, self.values)
        self.resetValues(self.values, default=default, func=func)
        if width is not None:
            self.configure(width=width)

    def get(self):
        return self.var.get()

    def set(self, val, force=False):
        if val in self.values or force:
            self.var.set(val)

    def resetValues(self, values, labels=None, func=None, default=""):
        """
        - labels: if labels different from values
        """
        self.values = values
        if labels is None or len(values) != len(labels):
            if labels is not None and len(labels) != len(values):
                print("OptionMenuVar resetValues wrong len", values, labels)
            labels = values
        self["menu"].delete(0, "end")
        for i in range(len(values)):
            val, lbl = values[i], labels[i]
            if func is not None:
                self["menu"].add_command(label=lbl, command=lambda v=val: func(v))
            else:
                self["menu"].add_command(label=lbl, command=tk._setit(self.var, val))
        if len(values) > 0:
            # does not change value if default '' and '' not in values
            if default != "" or (default == "" and default in values):
                self.var.set(default if default in values else values[0])


class EntryVar(tk.Entry):
    """replacement for tk.Entry, with embedded tk.Stringvar"""

    def __init__(self, frame, value, varType=None, **kwargs):
        if varType is None:
            self.var = tk.StringVar()
        else:
            self.var = varType()
        self.var.set(value)
        tk.Entry.__init__(self, frame, textvariable=self.var, **kwargs)

    def get(self):
        return self.var.get()

    def set(self, val):
        self.var.set(val)


class LabelVar(tk.Label):
    """replacement for tk.Label, with embedded tk.Stringvar"""

    def __init__(self, frame, value, **kwargs):
        self.var = tk.StringVar()
        self.var.set(value)
        tk.Label.__init__(self, frame, textvariable=self.var, **kwargs)

    def get(self):
        return self.var.get()

    def set(self, val):
        self.var.set(val)


class ComboboxVar(ttk.Combobox):
    """replacement for tk.Combobbox, with embedded tk.Stringvar"""

    def __init__(self, frame, values, default="", **kwargs):
        self.values = values
        self.var = tk.StringVar()
        self.var.set(default)
        ttk.Combobox.__init__(
            self, frame, values=self.values, textvariable=self.var, **kwargs
        )

    def get(self):
        return self.var.get()

    def set(self, val):
        self.var.set(val)


class CheckbuttonVar(tk.Checkbutton):
    """replacement for tk.Checkbutton, with embedded tk.BooleanVar"""

    def __init__(self, frame, text, default, **kwargs):
        self.var = tk.BooleanVar()
        self.var.set(default)
        tk.Checkbutton.__init__(self, frame, text=text, variable=self.var, **kwargs)

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)


class ButtonSmall(tk.Frame):
    """fabricate a Button with a tunable size"""

    def __init__(self, frame, text, command, width=16, height=16, **kwargs):
        tk.Frame.__init__(self, frame, width=width, height=height)
        self.propagate(0)
        self._button = tk.Button(self, text=text, command=command, **kwargs)
        self._button.pack(fill=tk.BOTH, expand=True)


class FrameScrollable(tk.Frame):
    """
    A Frame with a vertical scrollbar on the right
    Widgets must be placed into .child
    Special methods:
    - update_idletasks()
    """

    def __init__(self, parent, **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)
        # elements
        self.scrollbary = tk.Scrollbar(self, command=self.scrolly, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(self)
        self.child = tk.Frame(self)
        # geometry
        self.scrollbary.grid(row=0, column=1, sticky=tk.N + tk.S + tk.E)
        self.canvas.grid(row=0, column=0, sticky=tk.N + tk.S + tk.W)
        self.canvas.create_window(0, 0, window=self.child, anchor="nw")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        # behavior
        self.canvas.configure(yscrollcommand=self.scrollbary.set)
        self.canvas.bind("<Configure>", self.on_configure)
        self.child.bind("<MouseWheel>", self.on_mousewheel)
        self.bind("<MouseWheel>", self.on_mousewheel)
        # assign ref to be able to uncind if necessary
        self._upd_idle = self.child.bind("<Configure>", self.update_idletasks)

    def update_idletasks(self, event=None):
        """
        Updates the child Frame and resizes the canvas accordingly.
        Normally handled by event self.child <Configure>
        """
        self.child.update_idletasks()
        w = self.child.winfo_width()
        h = self.child.winfo_height()
        self.canvas.config(width=w, height=h)

    def on_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def scrolly(self, *args):
        self.canvas.yview(*args)

    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class FrameTitleContentHide(tk.Frame):
    """
    A class to show and hide some content in the user-interface

    - createButtons: if False, does not create button AND does not fill Frames
      _title and _content. Up to the user to do that. Buttons can be created
      by calling createButtonShow() and createButtonHide()
    """

    # some hack. ideally would need to write a simpler version of
    # FrameTitleContentHide without buttons and make other versions derive
    # from it

    def __init__(
        self,
        master,
        funcFillTitle,
        funcFillContent,
        contentkwargs={},
        default="show",
        layout="horizontal",
        showHideTitle=False,
        createButtons=True,
        horizLineFrame=True,
        **kwargs
    ):
        tk.Frame.__init__(self, master, **kwargs)
        self.setButtonLabels()
        self._visiblecontent = True
        self.showHideTitle = showHideTitle
        self._layout = layout
        self._contentkwargs = contentkwargs
        self.createButtons = createButtons
        self.horizLineFrame = horizLineFrame
        # define elements inside
        self._title = None
        self.createWidgets()
        # fill title and content
        if self.createButtons:
            if funcFillTitle is not None and self._title is not None:
                funcFillTitle(self._title)
            if funcFillContent is not None:
                funcFillContent(self._content)
        if self._title is not None:
            bind_tree(self._title, "<Button-1>", self.showHide)
        if default == "hide":
            self.showHide()

    @classmethod
    def frameHline(cls, frame):
        return tk.Frame(frame, height=3, background="gainsboro")

    def setButtonLabels(self):
        self.btnlbl_in = "\u25B3"
        self.btnlbl_out = "\u25BC"

    def createButton(self, frame, symbol="auto", size=None):
        """
        size: None: return Button. 'auto': [20, 20]. Or size [x, y]
        """
        if symbol == "auto":
            symbol = self.btnlbl_in
        if size == "auto":
            size = [20, 20]
        if size is None:
            return tk.Button(frame, text=symbol, command=self.showHide)
        else:
            fr = tk.Frame(frame, width=size[0], height=size[1])
            fr.propagate(0)
            btn = tk.Button(fr, text=symbol, command=self.showHide)
            btn.pack(side="left", anchor="n", fill=tk.BOTH, expand=1)
            return fr, btn

    def createWidgets(self):
        side, anchor, fill = self.sideAnchorFill()
        self._up = tk.Frame(self)
        dwn = tk.Frame(self)
        self._up.pack(side="top", anchor="w", fill=X)
        dwn.pack(side="top", anchor="w", fill=X)
        self._title = tk.Frame(self._up)
        self._title.pack(side=side, anchor=anchor, fill=fill)
        if self.createButtons:
            self._buttonFrame, self._button = self.createButton(self._up, size="auto")
            self._buttonFrame.pack(side="left", anchor="center", padx=5)
        if self.horizLineFrame:
            self._horizLineFrame = self.frameHline(self._up)
            self._horizLineFrame.pack(
                side="left", anchor="center", fill=X, expand=1, padx=5
            )
        self._content = tk.Frame(dwn, **self._contentkwargs)
        self._content.pack(side=side, anchor=anchor, fill=fill)
        self._dummy = tk.Frame(dwn)
        self._dummy.pack(side=side, anchor=anchor, fill=fill)

    def getFrameTitle(self):
        return self._title

    def getFrameContent(self):
        return self._content

    def getButton(self):
        return self._button

    def isvisible(self):
        return self._visiblecontent

    def sideAnchorFill(self):
        side = "left" if self._layout == "horizontal" else "top"
        anchor = "center" if self._layout == "horizontal" else "w"
        fill = tk.X if self._layout == "horizontal" else tk.Y
        return side, anchor, fill

    def showHide(self, *args):
        side, anchor, fill = self.sideAnchorFill()
        if self._visiblecontent:
            if self.showHideTitle:
                self._title.pack(side=side, anchor=anchor, fill=fill)
                if not self.createButtons and not self.horizLineFrame:
                    self._up.pack(side=side, anchor=anchor, fill=fill)
            self._content.pack_forget()
            if self.createButtons:
                self._button.configure({"text": self.btnlbl_out})
        else:
            self._content.pack(side=side, anchor=anchor, fill=fill)
            if self.showHideTitle:
                self._title.pack_forget()
                if not self.createButtons and not self.horizLineFrame:
                    self._up.pack_forget()
            if self.createButtons:
                self._button.configure({"text": self.btnlbl_in})
        self._visiblecontent = not self._visiblecontent


class FrameTitleContentHideHorizontal(FrameTitleContentHide):
    def setButtonLabels(self):
        self.btnlbl_in = "\u25C1"
        self.btnlbl_out = "\u25B6"

    def createWidgets(self):
        if self.createButtons:
            left = tk.Frame(self, width=30)  # height=60,
            left.pack(side="left", anchor="w")
            self._buttonFrame, self._button = self.createButton(left, size="auto")
            self._buttonFrame.pack(side="left", anchor="center", padx=5)
            # left.propagate(0)  # to reserve space also when hidden
        righ = tk.Frame(self)
        righ.pack(side="left", anchor="w", fill=X)
        self._title = tk.Frame(righ)
        self._title.pack(side="left", anchor="center")
        self._dummy = tk.Frame(righ)
        self._dummy.pack(side="left", anchor="n", fill=X)
        self._content = tk.Frame(righ, **self._contentkwargs)
        self._content.pack(side="top", anchor="w", fill=X)


def imageToClipboard(graph):
    """copy the image output of a Graph to the clipboard - Windows only"""
    # save image, because we don't have pixel map at the moment
    print("Copying graph image to clipboard")
    selffilename = graph.filename if hasattr(graph, "filename") else None
    fileClipboard = "_grapatoclipboard"
    tmp = graph.getAttribute("saveSilent")
    graph.update({"saveSilent": True})
    graph.plot(ifSave=True, ifExport=False, filesave=fileClipboard)
    graph.update({"saveSilent": tmp})
    if selffilename is not None:  # restore self.filename
        graph.filename = selffilename
    from io import BytesIO
    from PIL import Image

    try:
        import win32clipboard
    except ImportError as e:
        print(
            "Module win32clipboard not found, cannot copy to clipboard.",
            "Image was created.",
        )
        print(e)
        return False

    def send_to_clipboard(clip_type, content):
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(clip_type, content)
        win32clipboard.CloseClipboard()

    # read image -> get pixel map, convert into clipbaord-readable format
    output = BytesIO()
    img = Image.open(fileClipboard + ".png")

    # new version, try to copy png into clipboard.
    # Maybe people would complain, then revert to legacy
    img.save(output, "PNG")
    data = output.getvalue()
    output.close()
    send_to_clipboard(win32clipboard.RegisterClipboardFormat("PNG"), data)
    return True

    """
    # LEGACY. Through BMP format, transparency was lost. Keep the code in case of
    try:
        img = img.convert('RGBA')
    except KeyError:
        print('Copy to clipboard: RGBA not found. More recent version PIL',
              "versions should be able though. Transparency won't be handled",
              'correctly.')
        img = img.convert('RGB')
    try:
        img.save(output, 'BMP')
    except (IOError, OSError):
        print("Copy to clipboard: cannot write image as BMP. More recent"
              "version PIL versions should be able though. Transparency won't",
              "be handled correctly.")
        img = img.convert('RGB')
        img.save(output, 'BMP')  # second try
    data = output.getvalue()[14:]
    output.close()
    send_to_clipboard(win32clipboard.CF_DIB, data)
    return True
    """
