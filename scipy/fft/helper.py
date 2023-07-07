from scipy._lib._array_api import as_xparray, array_namespace
from . import _npfft as npfft
from ._npfft import _pocketfft as pocketfft
import numpy as np


def fftfreq(n, d=1.0):
    return npfft.fftfreq(n, d)


def rfftfreq(n, d=1.0):
    return npfft.rfftfreq(n, d)


def fftshift(x, axes=None):
    return npfft.fftshift(x, axes)


def ifftshift(x, axes=None):
    return npfft.ifftshift(x, axes)


def next_fast_len():
    return npfft.next_fast_len()


def set_workers(workers):
    return npfft.set_workers(workers)


def get_workers():
    return npfft.get_workers()
