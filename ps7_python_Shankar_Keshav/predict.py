import numpy as np
from sigmoid import *

def predict(Theta1, Theta2, X):
    # input layer
    a1 = np.hstack((np.ones((X.shape[0], 1)), X))

    # hidden layer
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))

    # output layer
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    # outputs
    h_x = a3
    p = np.argmax(h_x, axis=1) + 1

    return p, h_x