from scipy._lib._array_api import as_xparray, array_namespace
from . import _npfft as npfft
from ._npfft import _pocketfft as pocketfft
import numpy as np


def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, orthogonalize=None):
    return npfft.dct(x, type, n, axis, norm, overwrite_x,
                     workers, orthogonalize)


def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    return npfft.idct(x, type, n, axis, norm, overwrite_x,
                      workers, orthogonalize)


def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, orthogonalize=None):
    return npfft.dst(x, type, n, axis, norm, overwrite_x,
                     workers, orthogonalize)


def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    return npfft.idst(x, type, n, axis, norm, overwrite_x,
                      workers, orthogonalize)


def dctn(x, type=2, s=None, axes=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    return npfft.dctn(x, type, s, axes, norm, overwrite_x,
                      workers, orthogonalize)


def idctn(x, type=2, s=None, axes=-1, norm=None, overwrite_x=False,
          workers=None, orthogonalize=None):
    return npfft.idctn(x, type, s, axes, norm, overwrite_x,
                       workers, orthogonalize)


def dstn(x, type=2, s=None, axes=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    return npfft.dstn(x, type, s, axes, norm, overwrite_x,
                      workers, orthogonalize)


def idstn(x, type=2, s=None, axes=-1, norm=None, overwrite_x=False,
          workers=None, orthogonalize=None):
    return npfft.idstn(x, type, s, axes, norm, overwrite_x,
                       workers, orthogonalize)
