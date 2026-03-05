from lenstronomy.SimulationAPI.data_api import DataAPI
from lenstronomy.SimulationAPI.model_api import ModelAPI
from lenstronomy.ImSim.image_model import ImageModel
import copy
import numpy as np


def amplitude2magnitude(amps, zero_point=30.):
    cps_norm = 1
    cps = amps*cps_norm
    delta_m = -np.log10(cps) * 2.5
    magnitude = delta_m + zero_point
    return magnitude
