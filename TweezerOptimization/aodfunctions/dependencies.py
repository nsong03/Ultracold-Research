 ## More dependencies
import scipy.fft as spfft
from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d
from scipy.ndimage import affine_transform as sp_affine_transform
from scipy.ndimage import gaussian_filter as sp_gaussian_filter
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


from scipy.ndimage import zoom


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.legendre import Legendre


cupyon=True

try:
    import cupy as cp
    import cupyx.scipy.fft as cpfft
    import cupyx.scipy.ndimage
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    from cupyx.scipy.ndimage import affine_transform as cp_affine_transform
    from cupy.lib.stride_tricks import as_strided

except ImportError:
    cp = np
    cpfft = spfft
    cp_gaussian_filter1d = sp_gaussian_filter1d
    cp_gaussian_filter = sp_gaussian_filter
    cp_affine_transform = sp_affine_transform
    cupyon = False
    from numpy.lib.stride_tricks import as_strided

    print("cupy not installed. Using numpy.")

import sys
import os 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping, minimize

import matplotlib.patches as patches
from matplotlib.pyplot import cm

# from aodfunctions.dependencies import *
# from aodfunctions.general import *
# from aodfunctions.testbed import *
# from aodfunctions.settings import *

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import ipyparallel as ipp
from scipy.special import binom, hyp2f1
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from scipy.integrate import simps
from scipy.integrate import cumtrapz
from numpy import gradient
