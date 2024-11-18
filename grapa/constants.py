import numpy as np


class Constants:
    """
    Physical constants
    Implements the @property decorator for read-only values
    Please import the instance CST created below, and use e.g. CST.c, CST.hbar etc.
    """

    def __init__(self):
        """Physical constants and important quantities, expressed in SI units"""
        self._q = 1.602176634e-19  # elemental charge [C]

        self._h = 6.62606979e-34  # Planck [J s]
        self._hbar = self._h / (2 * np.pi)  # [J s]

        self._c = 299792458  # c speed of light [m s-1]
        self._kb = 1.38064852e-23  # Boltzmann constant [J K-1]

        self._epsilon0 = 8.85418782e-12  # m-3 kg-1 s4 A2 # vacuum permittivity

        self._STC_T = 273.15 + 25  # K # Temperature STC Standard Test Condition 25°C

        """ Physical constants and other quantities, outside SI unit system """
        self._nm_eV = self._h * self._c / self._q * 1e9  # conversion photon nm to eV

    @property
    def q(self):
        return self._q

    @property
    def h(self):
        return self._h

    @property
    def hbar(self):
        return self._hbar

    @property
    def c(self):
        return self._c

    @property
    def kb(self):
        return self._kb

    @property
    def epsilon0(self):
        return self._epsilon0

    @property
    def STC_T(self):
        return self._STC_T

    @property
    def nm_eV(self):
        return self._nm_eV


CST = Constants()

# Tests
if __name__ == "__main__":
    print(CST.nm_eV)
    print(CST.STC_T)
    print("The script should fail after this line - read-only value")
    CST.STC_T = "this should fail"
    print(CST.STC_T)  # should fail
