from scipy._lib._array_api import as_xparray, array_namespace
from . import _npfft as npfft
from ._npfft import _pocketfft as pocketfft
import numpy as np


def set_backend(backend, coerce=False, only=False):
    return npfft.set_backend(backend, coerce, only)


def skip_backend(backend):
    return npfft.skip_backend(backend)


def set_global_backend(backend, coerce=False, only=False, try_last=False):
    return npfft.set_global_backend(backend, coerce, only, try_last)


def register_backend(backend):
    return npfft.register_backend(backend)
