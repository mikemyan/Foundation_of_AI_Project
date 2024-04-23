import numpy as np

def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window