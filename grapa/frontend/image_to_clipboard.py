# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import os
import logging
from io import BytesIO
from typing import TYPE_CHECKING

from grapa.shared.error_management import issue_warning

if TYPE_CHECKING:
    from grapa.graph import Graph
    
logger = logging.getLogger(__name__)


def image_to_clipboard(graph: "Graph", folder=""):
    """copy the image output of a Graph to the clipboard - Windows only"""
    # save image, because we don't have pixel map at the moment
    print("Copying graph image to clipboard")
    selffilename = graph.filename if hasattr(graph, "filename") else None
    file_clipboard = "_grapatoclipboard"
    if len(folder) > 0:
        file_clipboard = os.path.join(folder, file_clipboard)
    tmp = graph.attr("saveSilent")
    graph.update({"saveSilent": True})
    graph.plot(if_save=True, if_export=False, filesave=file_clipboard)
    graph.update({"saveSilent": tmp})
    if selffilename is not None:  # restore self.filename
        graph.filename = selffilename
    from PIL import Image

    try:
        import win32clipboard
    except ImportError:
        msg = (
            "Module win32clipboard not found, cannot copy to clipboard. Image was "
            "created: %s. Try the following:\npip install pywin32"
        )
        issue_warning(logger, msg, file_clipboard, severity_log="error", exc_info=True)
        return False

    def send_to_clipboard(clip_type, content):
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(clip_type, content)
        win32clipboard.CloseClipboard()

    # read image -> get pixel map, convert into clipboard-readable format
    output = BytesIO()
    img = Image.open(file_clipboard + ".png")

    # new version, try to copy png into clipboard.
    # Maybe people would complain, then revert to legacy
    img.save(output, "PNG")
    data = output.getvalue()
    output.close()
    send_to_clipboard(win32clipboard.RegisterClipboardFormat("PNG"), data)
    return True
