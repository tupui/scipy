from scipy._lib._array_api import as_xparray, array_namespace
from . import _npfft as npfft
from ._npfft import _pocketfft as pocketfft
import numpy as np


def fht(a, dln, mu, offset=0.0, bias=0.0):
    return npfft.fht(a, dln, mu, offset, bias)


def ifht(a, dln, mu, offset=0.0, bias=0.0):
    return npfft.ifht(a, dln, mu, offset, bias)
