import numpy as np

from grapa.constants import CST


def test_constants():
    assert CST.kb == 1.380649e-23  # J/K
    assert CST.c == 299792458  # m s-1
    assert CST.h == 6.62607015e-34  # J Hz-1
    assert CST.hbar == 6.62607015e-34 / 2 / np.pi
    assert CST.q == 1.602176634e-19  # C
    assert CST.N_A == 6.02214076e23

    assert CST.STC_T == 298.15  # K
    assert CST.mu0 == 1.25663706127e-6
    assert np.abs(CST.epsilon0 - 8.8541878188e-12) / CST.epsilon0 < 1e-8  # F m-1

    assert np.abs(CST.nm_eV - 1239.84198) / CST.nm_eV < 1e-8



def test_constants_fail():
    try:
        CST.STC_T = "this should fail"
        assert False
    except Exception:
        pass
    try:
        CST.c = 123  # should fail
        assert False
    except Exception:
        pass
