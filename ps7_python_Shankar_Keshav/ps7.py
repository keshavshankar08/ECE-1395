import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import random
from predict import *
from nnCost import *
from sigmoidGradient import *
from sGD import *
import time

# ----- Problem 0 -----
print("\n----- Problem 0 -----")

# 0a
vehicle_data = loadmat("input/HW7_Data2_full.mat")
X = vehicle_data["X"]
y_labels = vehicle_data["y_labels"]
random_indices = np.random.choice(X.shape[0], 16, replace=False)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    image = X[random_indices[i]].reshape(32, 32).T
    ax.imshow(image, cmap='gray')
    if(y_labels[random_indices[i]][0] == 1):
        ax.set_title("Airplane")
    elif(y_labels[random_indices[i]][0] == 2):
        ax.set_title("Automobile")
    else:
        ax.set_title("Truck")
    ax.axis('off')
plt.savefig("output/ps7-0-a-1.png")
plt.close()

# 0b
indices = np.arange(len(X))
np.random.shuffle(indices)
X_shuffled = X[indices]
y_labels_shuffled = y_labels[indices]
train_split = 13000
test_split = 2000
X_train = X_shuffled[:train_split, :]
X_test = X_shuffled[-test_split:, :]
y_labels_train = y_labels_shuffled[:train_split, :]
y_labels_test = y_labels_shuffled[-test_split:, :]


# ----- Problem 1 -----
print("\n----- Problem 1 -----")

# 1b
parameters = loadmat("input/HW7_weights_3_full.mat")
theta1 = parameters["Theta1"]
theta2 = parameters["Theta2"]
p, h_x = predict(theta1, theta2, X_train)
accuracy = np.mean(p == y_labels_train.flatten()) * 100
print("Training accuracy: " + str(round(accuracy, 2)) + "%")


# ----- Problem 2 -----
print("\n----- Problem 2 -----")

# 2b
lambdas = [0.1, 1, 2]
for lamb in lambdas:
    J = nnCost(theta1, theta2, X, y_labels, 3, lamb)
    print("Cost when lambda = " + str(lamb) + ": " + str(round(J, 2)))


# ----- Problem 3 -----
print("\n----- Problem 3 -----")

z = np.array([-10, 0, 10])
g_z_prime = sigmoidGradient(z)
print("Sigmoid gradient: " + str(g_z_prime))


# ----- Problem 4 -----
print("\n----- Problem 4 -----")
theta1, theta2 = sGD(1024, 50, 3, X_train, y_labels_train, 0.1, 0.01, 1)


# ----- Problem 5 -----
print("\n----- Problem 5 -----")
lambdas = [0.1, 1, 2]
epochs = [50, 300]
for epoch in epochs:
    for lamb in lambdas:
        start_time = time.time()
        print("\nEpoch: " + str(epoch) + "\tLambda: " + str(lamb))
        Theta1, Theta2 = sGD(1024, 50, 3, X_train, y_labels, lamb, 0.01, epoch)
        p_train, h_x_train = predict(Theta1, Theta2, X_train)
        p_test, h_x_test = predict(Theta1, Theta2, X_test)
        train_accuracy = np.mean(clear == y_labels_train.flatten()) * 100
        test_accuracy = np.mean(p_test == y_labels_test.flatten()) * 100
        end_time = time.time()
        print("Training accuracy: " + str(round(train_accuracy, 2)) + "%")
        print("Testing accuracy: " + str(round(test_accuracy, 2)) + "%")
        print("Training cost: " + str(round(nnCost(Theta1, Theta2, X_train, y_labels_train, 3, lamb), 2)))
        print("Testing cost: " + str(round(nnCost(Theta1, Theta2, X_test, y_labels_test, 3, lamb), 2)))
        print("Execution time: " + str(round((end_time-start_time)/60, 2)) + " minutes.")