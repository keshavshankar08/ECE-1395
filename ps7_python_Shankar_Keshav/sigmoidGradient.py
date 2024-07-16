import numpy as np
from sigmoid import *

def sigmoidGradient(z):
    g = sigmoid(z)
    g_prime = g * (1 - g)

    return g_prime