# -*- coding: utf-8 -*-
"""Graphical user interface GUI of grapa

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    if path not in sys.path:
        sys.path.append(path)

    from grapa.frontend.gui_main import build_ui

    # open_file = None
    # config_file = sys.argv[1]
    build_ui()  #  open_file=open_file, config_file=config_file)
