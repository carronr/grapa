# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics,
Romain Carron
"""

import logging
from tkinter import ttk
import tkinter as tk
from tkinter import X

from grapa.utils.error_management import issue_warning

logger = logging.getLogger(__name__)


def bind_tree(widget, event, callback, add=""):
    """Binds an event to a widget and all its descendants."""
    widget.bind(event, callback, add)
    for child in widget.children.values():
        bind_tree(child, event, callback)  # , replace_callback


class TextWriteable(tk.Text):
    """
    New texyt field class, with possibility for write (print in this field)
        Used for the console output
    """

    def write(self, string):
        """called by StreamHandler setStream, as this widget is the console"""
        self.insert(tk.END, string)
        self.see(tk.END)

    def flush(self):
        """called by StreamHandler setStream, as this widget is the console"""
        self.update_idletasks()


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
        self.reset_values(self.values, default=default, func=func)
        if width is not None:
            self.configure(width=width)

    def get(self):
        """Returns the current value of the OptionMenuVar."""
        return self.var.get()

    def set(self, val, force=False):
        """Sets the current value of the OptionMenuVar."""
        if val in self.values or force:
            self.var.set(val)

    def reset_values(self, values, labels=None, func=None, default=""):
        """
        - labels: if labels different from values
        """
        self.values = values
        if labels is None or len(values) != len(labels):
            if labels is not None and len(labels) != len(values):
                msg = "OptionMenuVar resetValues wrong len. {}, {}."
                issue_warning(logger, msg.format(values, labels))
            labels = values
        self["menu"].delete(0, "end")
        for val, lbl in zip(values, labels):
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
        """Returns the current value of the EntryVar."""
        return self.var.get()

    def set(self, val):
        """Sets the current value of the EntryVar."""
        self.var.set(val)


class LabelVar(tk.Label):
    """replacement for tk.Label, with embedded tk.Stringvar"""

    def __init__(self, frame, value, **kwargs):
        self.var = tk.StringVar()
        self.var.set(value)
        tk.Label.__init__(self, frame, textvariable=self.var, **kwargs)

    def get(self):
        """Returns the current value of the LabelVar."""
        return self.var.get()

    def set(self, val):
        """Sets the current value of the LabelVar."""
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
        """Returns the current value of the ComboboxVar."""
        return self.var.get()

    def set(self, val):
        """Sets the current value of the ComboboxVar."""
        self.var.set(val)

    def reset_values(self, values):
        """Resets the possible values of the ComboboxVar."""
        self["values"] = values


class CheckbuttonVar(tk.Checkbutton):
    """replacement for tk.Checkbutton, with embedded tk.BooleanVar"""

    def __init__(self, frame, text, default, **kwargs):
        self.var = tk.BooleanVar()
        self.var.set(default)
        tk.Checkbutton.__init__(self, frame, text=text, variable=self.var, **kwargs)

    def get(self):
        """Returns the current value of the CheckbuttonVar."""
        return self.var.get()

    def set(self, value):
        """Sets the current value of the CheckbuttonVar."""
        self.var.set(value)


class ButtonSmall(tk.Frame):
    """fabricate a Button with a tunable size"""

    def __init__(self, frame, text, command, width=16, height=16, **kwargs):
        tk.Frame.__init__(self, frame, width=width, height=height)
        self.propagate(False)
        self._button = tk.Button(self, text=text, command=command, **kwargs)
        self._button.pack(fill=tk.BOTH, expand=True)


class ButtonVar(tk.Button):
    """replacement for tk.Button, with embedded tk.Stringvar for the text"""

    def __init__(self, frame, text, command, **kwargs):
        self.var = tk.StringVar()
        super().__init__(frame, textvariable=self.var, command=command, **kwargs)
        self.var.set(text)

    def get(self):
        """Returns the current value of the ButtonVar."""
        return self.var.get()

    def set(self, text):
        """Sets the current value of the ButtonVar."""
        return self.var.set(text)


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

    def update_idletasks(self, _event=None):
        """
        Updates the child Frame and resizes the canvas accordingly.
        Normally handled by event self.child <Configure>
        """
        self.child.update_idletasks()
        w = self.child.winfo_width()
        h = self.child.winfo_height()
        self.canvas.config(width=w, height=h)

    def on_configure(self, _event):
        """Called when the child frame is configured.
        Updates the scrollregion of the canvas."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def scrolly(self, *args):
        """Scroll vertically the canvas."""
        self.canvas.yview(*args)

    def on_mousewheel(self, event):
        """Scroll vertically the canvas with the mouse wheel."""
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

    BUTTON_LABEL_IN = "\u25b3"
    BUTTON_LABEL_OUT = "\u25bc"

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
        self._visiblecontent = True
        self.show_hide_title = showHideTitle
        self._layout = layout
        self._contentkwargs = contentkwargs
        self.if_create_buttons = createButtons
        self.if_horizlineframe = horizLineFrame
        self._horizlineframe: tk.Frame  # namespace placeholder
        # define elements inside
        self._title: tk.Frame  # namespace placeholder
        self._buttonframe: tk.Frame  # namespace placeholder
        self.create_widgets()
        # fill title and content
        if self.if_create_buttons:
            if funcFillTitle is not None and self._title is not None:
                funcFillTitle(self._title)
            if funcFillContent is not None:
                funcFillContent(self._content)
        if self._title is not None:
            bind_tree(self._title, "<Button-1>", self.show_hide)
        if default == "hide":
            self.show_hide()

    @classmethod
    def frame_hline(cls, frame):
        """Create a horizontal line frame"""
        return tk.Frame(frame, height=3, background="gainsboro")

    def cw_button_sized(self, frame, symbol="auto", size="auto"):
        """
        size: None:
        auto: [20, 20]. Or size [x, y]
        return Button.
        """
        if size == "auto":
            size = [20, 20]
        fr = tk.Frame(frame, width=size[0], height=size[1])
        fr.propagate(False)
        btn = self.cw_button_only(fr, symbol)
        btn.pack(side="left", anchor="n", fill=tk.BOTH, expand=1)
        return fr, btn

    def cw_button_only(self, frame, symbol):
        """Create only the button."""
        if symbol == "auto":
            symbol = self.BUTTON_LABEL_IN
        return tk.Button(frame, text=symbol, command=self.show_hide)

    def create_widgets(self):
        """Create the internal widgets."""
        side, anchor, fill = self._side_anchor_fill()
        self._up = tk.Frame(self)
        dwn = tk.Frame(self)
        self._up.pack(side="top", anchor="w", fill=X)
        dwn.pack(side="top", anchor="w", fill=X)
        self._title = tk.Frame(self._up)
        self._title.pack(side=side, anchor=anchor, fill=fill)
        if self.if_create_buttons:
            self._buttonframe, self._button = self.cw_button_sized(
                self._up, size="auto"
            )
            self._buttonframe.pack(side="left", anchor="center", padx=5)
        if self.if_horizlineframe:
            self._horizlineframe = self.frame_hline(self._up)
            self._horizlineframe.pack(
                side="left", anchor="center", fill=X, expand=1, padx=5
            )
        self._content = tk.Frame(dwn, **self._contentkwargs)
        self._content.pack(side=side, anchor=anchor, fill=fill)
        self._dummy = tk.Frame(dwn)
        self._dummy.pack(side=side, anchor=anchor, fill=fill)

    def get_frame_title(self):
        """Returns the title frame."""
        return self._title

    def get_frame_content(self):
        """Returns the content frame."""
        return self._content

    def get_button(self):
        """Returns the show/hide button."""
        return self._button

    def is_visible(self):
        """Returns True if the content is visible, False otherwise."""
        return self._visiblecontent

    def _side_anchor_fill(self):
        side = "left" if self._layout == "horizontal" else "top"
        anchor = "center" if self._layout == "horizontal" else "w"
        fill = tk.X if self._layout == "horizontal" else tk.Y
        return side, anchor, fill

    def show_hide(self, *_args):
        """Show or hide the content frame."""
        side, anchor, fill = self._side_anchor_fill()
        if self._visiblecontent:
            if self.show_hide_title:
                self._title.pack(side=side, anchor=anchor, fill=fill)
                if not self.if_create_buttons and not self.if_horizlineframe:
                    self._up.pack(side=side, anchor=anchor, fill=fill)
            self._content.pack_forget()
            if self.if_create_buttons:
                self._button.configure({"text": self.BUTTON_LABEL_OUT})
        else:
            self._content.pack(side=side, anchor=anchor, fill=fill)
            if self.show_hide_title:
                self._title.pack_forget()
                if not self.if_create_buttons and not self.if_horizlineframe:
                    self._up.pack_forget()
            if self.if_create_buttons:
                self._button.configure({"text": self.BUTTON_LABEL_IN})
        self._visiblecontent = not self._visiblecontent


class FrameTitleContentHideHorizontal(FrameTitleContentHide):
    """Frame with title and content area that can be shown/hidden horizontally."""

    BUTTON_LABEL_IN = "\u25c1"
    BUTTON_LABEL_OUT = "\u25b6"

    def create_widgets(self):
        """Create the internal widgets."""
        if self.if_create_buttons:
            left = tk.Frame(self, width=30)  # height=60,
            left.pack(side="left", anchor="w")
            self._buttonframe, self._button = self.cw_button_sized(left, size="auto")
            self._buttonframe.pack(side="left", anchor="center", padx=5)
            # left.propagate(0)  # to reserve space also when hidden
        righ = tk.Frame(self)
        righ.pack(side="left", anchor="w", fill=X)
        self._title = tk.Frame(righ)
        self._title.pack(side="left", anchor="center")
        self._dummy = tk.Frame(righ)
        self._dummy.pack(side="left", anchor="n", fill=X)
        self._content = tk.Frame(righ, **self._contentkwargs)
        self._content.pack(side="top", anchor="w", fill=X)
