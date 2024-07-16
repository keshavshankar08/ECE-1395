import numpy as np
from sigmoid import *
from sigmoidGradient import *
import matplotlib.pyplot as plt
from nnCost import *

def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lamb, alpha, MaxEpochs):
    # 4a
    Theta1 = np.random.uniform(-0.18, 0.18, (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.random.uniform(-0.18, 0.18, (num_labels, hidden_layer_size + 1))
    m = X_train.shape[0]

    costs = []

    # recode labels
    y_train_encoded = np.zeros((m, num_labels))
    for i in range(m):
        y_train_encoded[i, y_train[i] - 1] = 1

    # loop through epochs
    for epoch in range(MaxEpochs):
        print("Progress: " + str(round(epoch/MaxEpochs*100.0, 2)) + " %")
        # 4b
        for i in range(m):
            # forward pass
            a1 = np.hstack((1, X_train[i]))
            z2 = np.dot(Theta1, a1)
            a2 = sigmoid(z2)
            a2 = np.hstack((1, a2))
            z3 = np.dot(Theta2, a2)
            a3 = sigmoid(z3) # h_x

            # back propagation
            delta3 = a3 - y_train_encoded[i]
            delta2 = np.dot(Theta2.T, delta3) * sigmoidGradient(np.hstack((1, z2)))

            delta2 = delta2[1:]

            Delta2 = np.outer(delta3, a2)
            Delta1 = np.outer(delta2, a1)

            # 4c
            theta1_regularized = np.copy(Theta1)
            theta1_regularized[:, 0] = 0
            theta2_regularized = np.copy(Theta2)
            theta2_regularized[:, 0] = 0

            theta1_gradient = (Delta1 + (lamb * theta1_regularized))
            theta2_gradient = (Delta2 + (lamb * theta2_regularized))

            # 4d
            Theta1 -= alpha * theta1_gradient
            Theta2 -= alpha * theta2_gradient

            # 4e
            J = nnCost(Theta1, Theta2, X_train[i:i+1], y_train[i:i+1], num_labels, lamb)
            costs.append(J)

    '''
    plt.plot(costs)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.savefig('output/ps7-e-1.png')
    plt.close()
    '''

    return Theta1, Theta2