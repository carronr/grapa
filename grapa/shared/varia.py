# -*- coding: utf-8 -*-
"""
@author: Romain Carron
Copyright (c) 2026, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from grapa.curve import Curve

logger = logging.getLogger(__name__)


def get_point_closest_to_xy(curve: "Curve", x, y, alter="", offsets=False):
    """
    Return the data point closest to the x,y values.
    Priority on x, compares y only if equal distance on x
    """
    if isinstance(alter, str):
        alter = ["", alter]
    # select most suitable point based on x
    datax = curve.x_offsets(alter=alter[0])
    absxm = np.abs(datax - x)
    idx = np.where(absxm == np.min(absxm))
    if len(idx) == 0:
        idx = np.argmin(absxm)
    elif len(idx) == 1:
        idx = idx[0]
    else:
        # len(idx) > 1: select most suitable point based on y
        datay = curve.y_offsets(index=idx, alter=alter[1])
        absym = np.abs(datay - y)
        idxy = np.where(absym == np.min(absym))
        if len(idxy) == 0:
            idx = idx[0]
        elif len(idxy) == 1:
            idx = idx[idxy[0]]
        else:  # equally close in x and y -> returns first datapoint found
            idx = idx[idxy[0]]
    idx_out = idx if len(idx) <= 1 else idx[0]
    if offsets:
        # no alter, but offset for the return value
        return curve.x_offsets(index=idx)[0], curve.y_offsets(index=idx)[0], idx_out
    # no alter, no offsets for the return value
    return curve.x(index=idx)[0], curve.y(index=idx)[0], idx_out
