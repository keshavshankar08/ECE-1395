import numpy as np
from sigmoid import *

def gradFunction(theta, X_train, y_train):
    # get num points
    m = len(X_train)

    # calculate gradient
    h_theta = sigmoid(np.dot(X_train, theta))  
    gradient = 1.0/m * (np.dot(X_train.T, (h_theta - y_train)))

    # make 1d
    gradient = np.ndarray.flatten(gradient)
    return gradient