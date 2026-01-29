import math
import numpy as np

def circular_mean(yaws, weights):
    sin_sum = np.sum(weights * np.sin(yaws))
    cos_sum = np.sum(weights * np.cos(yaws))
    return np.arctan2(sin_sum, cos_sum)