import numpy as np
import scipy as sp
from scipy import constants
from collections import namedtuple

from iminuit import Minuit
import iminuit

Hit = namedtuple("Hit", ["x", "y", "z", "t", "x_err", "y_err", "z_err", "t_err","layer", "ind"])
Track = namedtuple("Track", ["x0", "y0", "z0", "t", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
Vertex = namedtuple("Vertex", ["x0", "y0", "z0", "t0", "cov", "chi2", "tracks"])