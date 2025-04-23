"""Module storing the values of physical constants, expressed in SI units."""

import numpy as np


class Constants:
    """
    A number of physical constants, expressed in SI units.

    Suggestion: import the instance CST created below, and use e.g. CST.c, CST.hbar etc.
    """

    # Implements the @property decorator for read-only values

    def __init__(self):
        """Definition of the SI unit system"""
        # unperturbed ground state hyperfine transition frequency of the caesium
        # 133 atom
        self._DeltaNu_Cs = 9192631770  # [Hz]
        self._c = 299792458  # c speed of light [m s-1]
        self._h = 6.62607015e-34  # Planck [J s]
        self._q = 1.602176634e-19  # elemental charge [C]
        self._kb = 1.380649e-23  # Boltzmann constant [J K-1]
        self._N_A = 6.02214076e23  # Avogadro constant [mol-1]
        # luminous efficacy of monochromatic radiation of frequency 540e12 Hz
        self._K_cd = 683  # [lm W-1]

        # Physical constants and other quantities, outside definition of SI unit system
        self._hbar = self._h / (2 * np.pi)  # [J s]
        self._nm_eV = self._h * self._c / self._q * 1e9  # conversion photon nm to eV
        self._mu0 = 1.25663706127e-6  # vacuum magnetic permittivity [N A-2]
        # vacuum permittivity epsilon0 # 8.8541878188e-12
        self._epsilon0 = 1 / self._mu0 / self.c**2  # [F m-1] [m-3 kg-1 s4 A2]

        self._STC_T = 273.15 + 25  # K # Temperature STC Standard Test Condition 25°C

    @property
    def DeltaNu_Cs(self):
        """..., in Hz"""
        return self._DeltaNu_Cs

    @property
    def c(self):
        """Speed of light, in m/s"""
        return self._c

    @property
    def h(self):
        """Planck constant, in J s"""
        return self._h

    @property
    def q(self):
        """Elementary charge, in C"""
        return self._q

    @property
    def kb(self):
        """Boltzmann constant, in J K-1"""
        return self._kb

    @property
    def N_A(self):
        """Avogadro constant, mol-1"""
        return self._N_A

    @property
    def K_cd(self):
        """
        Luminous efficacy of monochromatic radiation of frequency 540e12 Hz. lm W-1
        """
        return self._K_cd

    @property
    def hbar(self):
        """Reduced Planck constant, in J s"""
        return self._hbar

    @property
    def nm_eV(self):
        """To convert photon energy (eV) into wavelength (nm)"""
        return self._nm_eV

    @property
    def mu0(self):
        """vacuum magnetic permittivity, in N A-2"""
        return self._mu0

    @property
    def epsilon0(self):
        """vacuum permittivity epsilon0, in F m-1"""
        return self._epsilon0

    @property
    def STC_T(self):
        """Standard test condition STC: temperature, in K"""
        return self._STC_T


CST = Constants()
