import numpy as np
from predict import *

def nnCost(Theta1, Theta2, X, y, K, lamb):
    m = len(X)

    # recode labels as vectors
    y_recoded = np.zeros((m, K))
    for i in range(m):
        y_recoded[i, y[i] - 1] = 1

    # get hypothesis for inputs
    p, h_x = predict(Theta1, Theta2, X)

    # calculate cost
    J = (-1/m * np.sum(y_recoded * np.log(h_x) + (1 - y_recoded) * np.log(1 - h_x))) + ((lamb / (2 * m)) * (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:]))))

    return J