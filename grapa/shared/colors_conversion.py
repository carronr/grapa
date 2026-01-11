# -*- coding: utf-8 -*-
"""Provides conversions between different color spaces. RGB, and CIE Lab and CIE LCh.
Meant as an extension to python colorsys
Includes reparametrized versions of CIE Lab and LCh, with values 0-1.

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""
import math


# CONVERSION CIUE Lab to/from RGB
def cielab_to_rgb(L: float, a: float, b: float):
    """Convert CIELAB to sRGB (D65).
    :param L: 0-100
    :param a: -127, 127
    :param b: -127, 127
    :return: (R,G,B) in 0-1 (consistency with colorsys convention).
    """
    # --- Lab → XYZ ---
    # Reference white D65
    Xn, Yn, Zn = 95.047, 100.000, 108.883

    fy = (L + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)

    def f_inv(t):
        if t**3 > 0.008856:
            return t**3
        else:
            return (t - 16 / 116) / 7.787

    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)
    # Normalize for sRGB matrix (XYZ scaled 0–1)
    X /= 100
    Y /= 100
    Z /= 100
    # --- XYZ → linear RGB ---
    r_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    # --- linear RGB → sRGB ---
    def compand(val):
        if val <= 0.0031308:
            return 12.92 * val
        else:
            return 1.055 * (val ** (1 / 2.4)) - 0.055

    r = compand(max(0, min(1, r_lin)))
    g = compand(max(0, min(1, g_lin)))
    b = compand(max(0, min(1, b_lin)))
    # Scale to 0–1
    return (r, g, b)


def rgb_to_cielab(r: float, g: float, b: float):
    """Convert an sRGB color to CIELAB (D65).
    :param r: float 0-1
    :param g: float 0-1
    :param b: float 0-1
    :return: (L* 0-100, a* ~-100-+100, b ~-100-+100)
    """

    # --- Helper functions ---
    def pivot_rgb(u: float) -> float:
        # u = u / 255.0  # if rgb 0-255
        return ((u + 0.055) / 1.055) ** 2.4 if u > 0.04045 else u / 12.92

    def pivot_xyz(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    # --- Convert RGB → linear RGB ---
    r_lin = pivot_rgb(r)
    g_lin = pivot_rgb(g)
    b_lin = pivot_rgb(b)
    # --- Linear RGB → XYZ (D65) ---
    # sRGB transform matrix
    X = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    Y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    Z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    # Normalize for D65 white point
    X /= 0.95047
    Y /= 1.00000
    Z /= 1.08883
    # --- XYZ → Lab ---
    fX = pivot_xyz(X)
    fY = pivot_xyz(Y)
    fZ = pivot_xyz(Z)
    L = (116 * fY) - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)
    return (L, a, b)


# CONVERSION CIE LCh to CIE Lab and RGB
def cielab_to_cielch(L: float, a: float, b: float):
    """Convert CIELAB to CIELCh°.
    :param L: Lightness (0-100)
    :param a: Lab chromaticity component a
    :param b: Lab chromaticity component b
    :return: (L, C, h) where C = chroma, h = hue angle in degrees (0-360)
    """
    C = math.sqrt(a * a + b * b)  # Chroma
    h = math.degrees(math.atan2(b, a))  # Hue angle in degrees
    if h < 0:
        h += 360  # Wrap into 0–360
    return (L, C, h)


def cielch_to_cielab(L: float, C: float, h: float):
    """Convert CIELCh° to CIELAB.
    :param L: Lightness (0-100)
    :param C: Chroma (>=0)
    :param h: Hue angle in degrees (0-360)
    Returns (L, a, b)
    """
    h_rad = math.radians(h)  # convert to radians
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    return (L, a, b)


def rgb_to_cielch(r, g, b):
    """(r,g,b) as floats 0-1
    :return: (L, C, h) where C = chroma, h = hue angle in degrees (0-360)
    """
    return cielab_to_cielch(*rgb_to_cielab(r, g, b))


def cielch_to_rgb(L: float, C: float, h: float):
    """Converts CIE LCh into rgb
    (L, C, h) where C = chroma, h = hue angle in degrees (0-360)
    :return: (r,g,b) as floats 0-1"""
    return cielab_to_rgb(*cielch_to_cielab(L, C, h))


# CONVERSION CIE Lab to its normalized parametrization
class CIELabNormalization:
    """Provides conversion functions for CIE Lab colorspace. The L, a, b values are
    normalized to 0-1."""

    @classmethod
    def labnorm_to_lab(cls, L_norm: float, a_norm: float, b_norm: float):
        """Value de-normalisation. See also inverse function lab_to_labnorm()"""
        return (L_norm * 100, a_norm * 256 - 128, b_norm * 256 - 128)

    @classmethod
    def lab_to_labnorm(cls, L: float, a: float, b: float):
        """Value normalisation. See also inverse function lab_to_labnorm()"""
        return (L / 100, (a + 128) / 256, (b + 128) / 256)


def cielabnorm_to_rgb(L_norm: float, a_norm: float, b_norm: float):
    """Convert CIELAB to sRGB (D65).
    All values are in 0-1 range for consistency with colorsys
    :param L: 0-1, internally mapped to 0-100
    :param a: 0-1, internally mapped to -127, 127
    :param b: 0-1, internally mapped to -127, 127
    :return: (R,G,B) in 0-1 (consistency with colorsys convention).
    """
    lab = CIELabNormalization.labnorm_to_lab(L_norm, a_norm, b_norm)
    return cielab_to_rgb(*lab)


def rgb_to_cielabnorm(r: float, g: float, b: float):
    """Convert an sRGB color to CIELAB (D65).
    :param r: float 0-1
    :param g: float 0-1
    :param b: float 0-1
    :return: L_norm, a_norm, b_norm. 0-1.
    """
    lab = rgb_to_cielab(r, g, b)
    return CIELabNormalization.lab_to_labnorm(*lab)


# CIE LCh normalized parametrization
class CIELChNormalization:
    """Provides conversion functions for CIE LCh colorspace, with parameter values
    normalized 0-1. CIE LCh is a reparametrization of CIE Lab, which parameters are
    also assumed to be normalized 0-1."""

    @staticmethod
    def lch_to_lchnorm(L: float, C: float, h: float):
        """Normalize parameter values L, a, b. See also lchnorm_to_lch()."""
        return (L / 100, C / 150, h / 360)

    @staticmethod
    def lchnorm_to_lch(L_norm: float, C_norm: float, h_norm: float):
        """De-normalize parameter values L, a, b. See also lch_to_lchnorm()."""
        return (L_norm * 100, C_norm * 150, h_norm * 360)


def cielabnorm_to_cielchnorm(L_norm: float, a_norm: float, b_norm: float):
    """Convert CIELAB to CIELCh°, all values between 0-1
    :param L: lightness (0-1)
    :param a: Lab chromaticity component b, normalized
    :param b: Lab chromaticity component b, normalized
    :return: (L, C, h) where C = chroma /150 and h = hue angle 0-1
    """
    lab = CIELabNormalization.labnorm_to_lab(L_norm, a_norm, b_norm)
    lch = cielab_to_cielch(*lab)
    return CIELChNormalization.lch_to_lchnorm(*lch)


def cielchnorm_to_cielabnorm(L_norm: float, C_norm: float, h_norm: float):
    """Convert CIELCh° to CIELAB.
    :param L: Lightness (normalized 0-1)
    :param C: Chroma (≥0, normalied /150)
    :param h: Hue angle in degrees (normalized 0-1)
    :return: (L, a, b) noramlized values 0-1
    """
    lch = CIELChNormalization.lchnorm_to_lch(L_norm, C_norm, h_norm)
    lab = cielch_to_cielab(*lch)
    return CIELabNormalization.lab_to_labnorm(*lab)


def rgb_to_cielchnorm(r, g, b):
    """(r,g,b) as floats 0-1"""
    lch = cielab_to_cielch(*rgb_to_cielab(r, g, b))
    return CIELChNormalization.lch_to_lchnorm(*lch)


def cielchnorm_to_rgb(L_norm: float, C_norm: float, h_norm: float):
    """(r,g,b) as floats 0-1"""
    lch = CIELChNormalization.lchnorm_to_lch(L_norm, C_norm, h_norm)
    return cielch_to_rgb(*lch)
