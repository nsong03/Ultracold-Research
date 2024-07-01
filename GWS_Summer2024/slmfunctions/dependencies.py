import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# from instrumental.drivers.cameras import uc480
import matplotlib.pyplot as plt
import math
from PIL import Image
import scipy.fftpack as sfft
import random
import sys
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label, center_of_mass
import cv2
import tifffile
from scipy.spatial import ConvexHull
from scipy.special import comb, factorial
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import sawtooth
from scipy.ndimage import rotate
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
import os
import ctypes
from ctypes import *
from scipy import misc
from time import sleep
from scipy.optimize import curve_fit
# from instrumental.drivers.cameras import uc480

## More dependencies
import scipy.fft as spfft
from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d
from scipy.ndimage import affine_transform as sp_affine_transform
from scipy.ndimage import gaussian_filter as sp_gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import pickle
import gzip
import shutil