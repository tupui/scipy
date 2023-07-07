from scipy._lib._array_api import as_xparray, array_namespace
from . import _npfft as npfft
from ._npfft._pocketfft import _pocketfft as pocketfft
import numpy as np


def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
        plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of fft.
    For numpy arrays, see the documentation for npfft.fft.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.fft(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.fft(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.fft(x, n, axis, norm)
    return xp.asarray(y)


def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of ifft.
    For numpy arrays, see the documentation for npfft.ifft.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.ifft(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.ifft(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.ifft(x, n, axis, norm)
    return xp.asarray(y)


def fft2(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    return npfft.fft2(x, n, axis, norm, overwrite_x, workers, plan)


def ifft2(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    return npfft.ifft2(x, n, axis, norm, overwrite_x, workers, plan)


def fftn(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of fftn.
    For numpy arrays, see the documentation for npfft.fftn.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.fftn(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.fftn(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.fftn(x, n, axis, norm)
    return xp.asarray(y)


def ifftn(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of ifftn.
    For numpy arrays, see the documentation for npfft.ifftn.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.ifftn(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.ifftn(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.ifftn(x, n, axis, norm)
    return xp.asarray(y)


def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of rfft.
    For numpy arrays, see the documentation for npfft.rfft.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.rfft(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.rfft(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.rfft(x, n, axis, norm)
    return xp.asarray(y)


def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of irfft.
    For numpy arrays, see the documentation for npfft.irfft.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.irfft(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.irfft(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.irfft(x, n, axis, norm)
    return xp.asarray(y)


def rfft2(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    return npfft.rfft2(x, n, axis, norm, overwrite_x, workers, plan)


def irfft2(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    return npfft.irfft2(x, n, axis, norm, overwrite_x, workers, plan)


def rfftn(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of rfftn.
    For numpy arrays, see the documentation for npfft.rfftn.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.rfftn(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.rfftn(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.rfftn(x, n, axis, norm)
    return xp.asarray(y)


def irfftn(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of irfftn.
    For numpy arrays, see the documentation for npfft.irfftn.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.irfftn(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.irfftn(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.irfftn(x, n, axis, norm)
    return xp.asarray(y)


def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of hfft.
    For numpy arrays, see the documentation for npfft.hfft.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.hfft(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.hfft(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.hfft(x, n, axis, norm)
    return xp.asarray(y)


def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    For non-numpy arrays, this implements the Array API specification of ihfft.
    For numpy arrays, see the documentation for npfft.ihfft.
    Note that if arguments outside of those in the Array API specification
    are provided with a non-numpy array, an exception is raised.
    """
    if isinstance(x, np.ndarray):
        return npfft.ihfft(x, n, axis, norm, overwrite_x, workers, plan)
    if overwrite_x is not False:
        Exception
    if workers is not None:
        Exception
    if plan is not None:
        Exception
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.ihfft(x, n, axis, norm)
    x = np.asarray(x)
    y = pocketfft.ihfft(x, n, axis, norm)
    return xp.asarray(y)


def hfft2(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    return npfft.hfft2(x, n, axis, norm, overwrite_x, workers, plan)


def ihfft2(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    return npfft.ihfft2(x, n, axis, norm, overwrite_x, workers, plan)


def hfftn(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    return npfft.hfftn(x, n, axis, norm, overwrite_x, workers, plan)


def ihfftn(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    return npfft.ihfftn(x, n, axis, norm, overwrite_x, workers, plan)
