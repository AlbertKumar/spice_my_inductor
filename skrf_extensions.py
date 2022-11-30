import math
import numpy as np
from matplotlib import pyplot as plt
import skrf as rf
from skrf.media import DefinedGammaZ0
from skrf import Network, Frequency, connect, innerconnect, mathFunctions

"""
Adds functions useful for extracting parameters from inductor scikit-RF Networks.
"""


def R11(ntwk):
    Y11= ntwk.y[:, 0, 0]
    return((1/Y11).real)


def R12(ntwk):
    Y12= ntwk.y[:, 0, 1]
    return((-1/Y12).real)


def L11(ntwk):
    Y11 = ntwk.y[:, 0, 0]
    f = ntwk.frequency.f
    return((1/Y11).imag / (2*math.pi*f))


def L12(ntwk):
    Y12 = ntwk.y[:, 0, 1]
    f = ntwk.frequency.f
    return((-1/Y12).imag / (2*math.pi*f))


def R(ntwk, p1=0, p2=1):
    if (p1==0 and p2==0) or (p1==1 and p2==1): Ypp = ntwk.y[:, p1, p2]
    else: Ypp = -ntwk.y[:, p1, p2]
    return((1/Ypp).real)


def L(ntwk, p1=0, p2=1):
    if (p1 == 0 and p2 == 0) or (p1 == 1 and p2 == 1): Ypp = -ntwk.y[:, p1, p2]
    else: Ypp = ntwk.y[:, p1, p2]
    f = ntwk.frequency.f
    return((-1/Ypp).imag / (2*math.pi*f))


def parallel(ntwk1, ntwk2):
    '''
    Returns parallel combination of two networks.
    :param ntwk1: Network object 1.
    :param ntwk2: Network object 2.
    :return: Parallel combination of two networks.
    '''
    # Check if frequencies of both networks are equal. Use allclose because of floating point rounding errors.
    # Normalize to minimum frequency so that the default reltol and abstol can be used.
    if np.allclose(ntwk1.frequency.f / ntwk1.frequency.f.min(), ntwk2.frequency.f) / ntwk2.frequency.f.min():
        raise Exception("shunt_branch: Frequencies of networks do not match.")

    freq = ntwk1.frequency
    ntwk = Network(frequency=freq, s=rf.y2s(ntwk1.y + ntwk2.y), z0=50)
    return(ntwk)
