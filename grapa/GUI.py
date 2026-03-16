# -*- coding: utf-8 -*-
"""Graphical user interface GUI of grapa

@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # So this file can be used as a starting point
    path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    if path not in sys.path:
        sys.path.append(path)
    from grapa.frontend.gui_main import build_ui

    build_ui()  #  open_file=open_file, config_file=config_file)
else:
    # to place build_ui in the namespace
    from grapa.frontend.gui_main import build_ui
