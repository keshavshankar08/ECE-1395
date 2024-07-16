import numpy as np
from sigmoid import *

def costFunction(theta, X_train, y_train):
    # get num points
    m = len(X_train)

    h_theta = sigmoid(np.dot(X_train, theta))
    J_theta = -1.0/m * (np.dot(y_train, np.log(h_theta + 1e-15)) + np.dot((1 - y_train), np.log(1 - h_theta + 1e-15))) # added 1e-15 so log doesnt divide by 0
    return J_theta
