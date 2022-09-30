import numpy as np
from matplotlib import pyplot as plt

def fun(x,y, a, z0):
    return a*(x+y+z0)

data = np.loadtxt('dish_zenith.txt').T

