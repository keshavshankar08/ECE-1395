import numpy as np
from costFunction import *

# Computes the gradient descent solution to linear regression given 2 features
def gradientDescent(X_train, y_train, alpha, iters):
    # declare vars
    m = len(X_train)
    n = len(X_train[0])
    theta = np.random.rand(n, 1)
    cost = np.zeros(iters)

    # for each iteration
    for i in range(iters):
        summation_0 = summation_1 = summation_2 = 0

        # for number of data points
        for j in range(m):
            summation_0 += theta[0][0]*X_train[j][0] + theta[0][0]*X_train[j][1] + theta[0][0]*X_train[j][2] - y_train[j][0]
            summation_1 += (theta[1][0]*X_train[j][0] + theta[1][0]*X_train[j][1] + theta[1][0]*X_train[j][2] - y_train[j][0]) * X_train[j][1]
            summation_2 += (theta[2][0]*X_train[j][0] + theta[2][0]*X_train[j][1] + theta[2][0]*X_train[j][2] - y_train[j][0]) * X_train[j][2]

        # compute theta and cost
        theta[0][0] = theta[0][0] - ((alpha/m) * summation_0)
        theta[1][0] = theta[1][0] - ((alpha/m) * summation_1)
        theta[2][0] = theta[2][0] - ((alpha/m) * summation_2)
        cost[i] = computeCost(X_train, y_train, theta)
    return theta, cost

# Computes the gradient descent solution to linear regression given 1 feature
def gradientDescent2(X_train, y_train, alpha, iters):
    # declare vars
    m = len(X_train)
    n = len(X_train[0])
    theta = np.random.rand(n, 1)
    cost = np.zeros(iters)

    # for each iteration
    for i in range(iters):
        summation_0 = summation_1 = 0

        # for number of data points
        for j in range(m):
            summation_0 += theta[0][0]*X_train[j][0] + theta[0][0]*X_train[j][1] - y_train[j][0]
            summation_1 += (theta[1][0]*X_train[j][0] + theta[1][0]*X_train[j][1] - y_train[j][0]) * X_train[j][1]

        # compute theta and cost
        theta[0][0] = theta[0][0] - ((alpha/m) * summation_0)
        theta[1][0] = theta[1][0] - ((alpha/m) * summation_1)
        cost[i] = computeCost(X_train, y_train, theta)
    return theta, cost