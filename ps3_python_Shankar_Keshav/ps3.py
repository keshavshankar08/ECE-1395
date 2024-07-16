import sys
sys.path.append(".")
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from sigmoid import *
from costFunction import *
from gradFunction import *
from normalEqn import *

# ----- Problem 1 -----
print("----- Problem 1 -----")
# load data1
with open('input/hw3_data1.txt', 'r') as f:
    reader = csv.reader(f)
    data1 = list(reader)
data1 = np.array(data1, dtype=float)

# 1a
X = np.concatenate((np.ones((len(data1),1)), np.take(data1, indices=[0,1], axis=1)), axis=1)
Y = np.take(data1, indices=[2], axis=1)
X_rows, X_cols = np.shape(X)
Y_rows, Y_cols = np.shape(Y)
print("X has " + str(X_rows) + " data points and " + str(X_cols) + " features")
print("Y has " + str(Y_rows) + " data points and " + str(Y_cols) + " labels")

# 1b
plt.title("Exam Scores")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.scatter(X[:, 1][Y.flatten() == 1], X[:, 2][Y.flatten() == 1], c='black', marker='+')
plt.scatter(X[:, 1][Y.flatten() == 0], X[:, 2][Y.flatten() == 0], c='green', marker='o')
plt.savefig("output/ps3-1-b")
plt.close()

# 1c
indices = np.arange(len(X))
np.random.shuffle(indices)
X_shuffled = X[indices]
Y_shuffled = Y[indices]
train_split = round(len(X) * 0.9)
test_split = len(X) - train_split
X_train = np.array(X_shuffled[:train_split, :])
X_test = np.array(X_shuffled[-test_split:])
Y_train = np.array(Y_shuffled[:train_split, :])
Y_test = np.array(Y_shuffled[-test_split:])

# 1d
z = np.arange(-15, 15.1, 0.1).reshape(-1, 1)
g_z = sigmoid(z)
plt.title("Sigmoid")
plt.xlabel("z")
plt.ylabel("g(z)")
plt.scatter(z, g_z)
plt.savefig("output/ps3-1-c")
plt.close()

# 1e
print("")
toy_data = np.array([[1,0,0],[1,3,1],[3,1,0],[3,4,1]])
X_toy = np.concatenate((np.ones((len(toy_data),1)), np.take(toy_data, indices=[0,1], axis=1)), axis=1) # appends bias and takes 0 and 1 from data
Y_toy = np.take(toy_data, indices=[2], axis=1) # takes 2 from data
theta_transpose = np.array([[2],[0],[0]])
cost_1e = costFunction(theta_transpose, X_toy, np.squeeze(Y_toy)) # need to make y 1d for cost function
gradient_1e = gradFunction(theta_transpose, X_toy, Y_toy)
print("Cost is: " + str(cost_1e))
print("Gradient of cost is: ")
print(gradient_1e)

# 1f
print("")
theta_initial = np.zeros((len(X_train[0]), 1))
theta_optimal = fmin_bfgs(f=costFunction, x0=theta_initial, fprime=gradFunction, args=(X_train, np.squeeze(Y_train))) # need to make y 1d for cost function
cost_1f = costFunction(theta_optimal, X_train, np.squeeze(Y_train))
print("Optimal theta is: ")
print(theta_optimal)
print("Cost with optimal theta is: " + str(cost_1f))

# 1g
X1_pred = np.linspace(min(X_train[:, 1]), max(X_train[:, 1]), 90) # uses min and max of train data to creates range
X2_pred = -(theta_optimal[0] + theta_optimal[1]*X1_pred) / theta_optimal[2]
plt.title("Exam Scores")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.scatter(X_train[:, 1][Y_train.flatten() == 1], X_train[:, 2][Y_train.flatten() == 1], c='black', marker='+') # these check the values of y, and seperate them based on 1 or 0 equality
plt.scatter(X_train[:, 1][Y_train.flatten() == 0], X_train[:, 2][Y_train.flatten() == 0], c='green', marker='o')
plt.plot(X1_pred, X2_pred)
plt.savefig("output/ps3-1-f")
plt.close()

# 1h
print("")
num_correct = 0
for i in range(len(X_test)):
    if(theta_optimal[0] + theta_optimal[1]*X_test[i,1] + theta_optimal[2]*X_test[i,2] > 0):
        num_correct += 1
Accuracy = num_correct/len(X_test) # accuracy = # correct points / total points
print("The accuracy of the prediction is: " + str(round(Accuracy, 2)) + "%")

# 1i
x_test = np.array([1, 60, 65])
y_test = sigmoid(theta_optimal[0] + theta_optimal[1]*x_test[1] + theta_optimal[2]*x_test[2])
print("The admission probability is: " + str(round(y_test, 2)) + "%")
print("The admission decision is: admitted") if(y_test > 0.5) else print("The admission decision is: denied") 

# ----- Problem 2 -----
print("\n----- Problem 2 -----")
# 2a
# load data2
with open('input/hw3_data2.csv', 'r') as f:
    reader = csv.reader(f)
    data2 = list(reader)
data2 = np.array(data2, dtype=float)
X_2_col0 = np.ones((len(data2), 1))
X_2_col1 = np.take(data2, indices=[0], axis=1)
X_2_col2 = X_2_col1**2
X_2 = np.concatenate((X_2_col0, X_2_col1), axis=1)
X_2 = np.concatenate((X_2, X_2_col2), axis=1)
Y_2 = np.take(data2, indices=[1], axis=1)
theta_2a = normalEqn(X_2, Y_2)
print("Learned model parameters: ")
print(theta_2a)

# 2b
X_result = np.linspace(min(X_2_col1.flatten()), max(X_2_col1.flatten()), 100)
Y_result = theta_2a[0] + (theta_2a[1]*X_result) + (theta_2a[2]*(X_result**2))
plt.title("Population vs. Profit")
plt.xlabel("population in thousands, n")
plt.ylabel("profit")
plt.scatter(X_2_col1, Y_2, marker='o', color='green')
plt.plot(X_result, Y_result)
plt.savefig("output/ps3-2-b")
plt.close()