import numpy as np

# Computes the cost given an estimate of the parameter vector theta
def computeCost(X, y, theta):
    # declare vars
    m = len(X)
    sum = 0

    # for number of data points
    for i in range(m):
        h_theta = 0
        # for number of features
        for j in range(len(X[0])):
            h_theta += theta[j][0]*X[i][j]
        sum += (h_theta - y[i][0])**2

    # compute cost and return
    J = 1.0/(2.0 * m) * sum
    return J