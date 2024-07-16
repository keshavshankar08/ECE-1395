import sys
sys.path.append(".")
import numpy as np
import csv
import matplotlib.pyplot as plt
from costFunction import *
from gradientDescent import *
from normalEqn import *

# ----- 1 - Cost Function -----
print("\n----- Problem 1 -----")
x = np.array([[1,1],[2,2],[3,3],[4,4]])
x = np.concatenate((np.ones((len(x),1)), x), axis=1)
y = np.array([[8],[6],[4],[2]])
theta_1 = [[0],[1],[0.5]]
theta_2 = [[10],[-1],[-1]]
theta_3 = [[3.5],[0],[0]]
print("Cost using theta 1: " + str(computeCost(x, y, theta_1)))
print("Cost using theta 2: " + str(computeCost(x, y, theta_2)))


# ----- 2 - Gradient Descent -----
print("\n----- Problem 2 -----")
theta_gd_2, cost_gd_2 = gradientDescent(x, y, 0.001, 15)
print("Estimate of theta: ")
print(theta_gd_2)
print("Cost after 15 iterations: " + str(round(cost_gd_2[14], 2)))


# ----- 3 - Normal Equation -----
print("\n----- Problem 3 -----")
theta_normal_3 = normalEqn(x, y)
print("Estimate of theta: ")
print(theta_normal_3)


# ----- 4 - Linear Regression With One Variable -----
print("\n----- Problem 4 -----")
# 4a
with open('input/hw2_data1.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
automobile_data = np.array(data, dtype=float)
X = np.take(automobile_data, indices=[0], axis=1)
Y = np.take(automobile_data, indices=[1], axis=1)

# 4b
plt.title("Automobile Data")
plt.xlabel("Horse Power (in 100's hp)")
plt.ylabel("Price (in $1000's)")
plt.scatter(X, Y)
plt.savefig("output/ps2-4-b")
plt.close()

# 4c
X = np.concatenate((np.ones((len(X),1)), X), axis=1)
print("X has " + str(len(X)) + " points and " + str(len(X[0])) + " features per point.")
print("Y has " + str(len(Y)) + " points and " + str(len(Y[0])) + " labels per point.")

# 4d
indices = np.arange(len(X))
np.random.shuffle(indices)
X_shuffled = X[indices]
Y_shuffled = Y[indices]
train_split = round(len(X) * 0.9)
test_split = len(X) - train_split
X_train = X_shuffled[:train_split, :].copy()
X_test = X_shuffled[-test_split:].copy()
Y_train = Y_shuffled[:train_split, :].copy()
Y_test = Y_shuffled[-test_split:].copy()

# 4e
theta_gd_4, cost_gd_4 = gradientDescent2(X_train, Y_train, 0.3, 500)
plt.title("Cost from gradient descent")
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.scatter(list(range(1, 501)), cost_gd_4)
plt.savefig("output/ps2-4-e")
plt.close()
print("Estimate of theta:")
print(theta_gd_4)

# 4f
best_fit_line_y = np.zeros((len(X),1))
for k in range(len(X)):
    best_fit_line_y[k][0] = theta_gd_4[0] + theta_gd_4[1]*X[k][1]
plt.title("Learned model")
plt.xlabel("Horse Power (in 100's hp)")
plt.ylabel("Price (in $1000's)")
plt.scatter(np.take(X, indices=[1], axis=1), Y)
plt.plot(np.take(X, indices=[1], axis=1), best_fit_line_y)
plt.savefig("output/ps2-4-f")
plt.close()

# 4g
diff_4g = 0
for point in range(len(X)):
    diff_4g += (Y[point][0] - best_fit_line_y[point][0])**2
mean_squared_error_4g = 1/(2 * len(X)) * diff_4g
print("The error in prediction using gradient descent is: " + str(round(mean_squared_error_4g, 2)) + " %")

# 4h
theta_normal_4 = normalEqn(X_test, Y_test)
cost_normal_4 = computeCost(X_test, Y_test, theta_normal_4)
best_fit_line_yg = np.zeros((len(X),1))
for k in range(len(X)):
    best_fit_line_yg[k][0] = theta_normal_4[0] + theta_normal_4[1]*X[k][1]
diff_4h = 0
for point in range(len(X)):
    diff_4h += (Y[point][0] - best_fit_line_yg[point][0])**2
mean_squared_error_4h = 1/(2 * len(X)) * diff_4h
print("The error in prediction using normal is: " + str(round(mean_squared_error_4h, 2)) + " %")

# 4i
theta_gd_4i_1, cost_4i_1 = gradientDescent2(X_train, Y_train, 0.001, 300)
plt.title("Cost from gradient descent (alpha = 0.001)")
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.scatter(list(range(1, 301)), cost_4i_1)
plt.savefig("output/ps2-4-i-1")
plt.close()
theta_gd_4i_2, cost_4i_2 = gradientDescent2(X_train, Y_train, 0.003, 300)
plt.title("Cost from gradient descent (alpha = 0.003)")
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.scatter(list(range(1, 301)), cost_4i_2)
plt.savefig("output/ps2-4-i-2")
plt.close()
theta_gd_4i_3, cost_4i_3 = gradientDescent2(X_train, Y_train, 0.03, 300)
plt.title("Cost from gradient descent (alpha = 0.03)")
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.scatter(list(range(1, 301)), cost_4i_3)
plt.savefig("output/ps2-4-i-3")
plt.close()
theta_gd_4i_4, cost_4i_4 = gradientDescent2(X_train, Y_train, 3, 300)
plt.title("Cost from gradient descent (alpha = 3)")
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.scatter(list(range(1, 301)), cost_4i_4)
plt.savefig("output/ps2-4-i-4")
plt.close()

# ----- 5 - Linear Regression With Multiple Variables -----
print("\n----- Problem 5 -----")
# 5a
with open('input/hw2_data3.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
automobile_emission_data = np.array(data, dtype=float)
engine_size_mean = round(np.mean(automobile_emission_data[:, 0]), 2)
car_weight_mean = round(np.mean(automobile_emission_data[:, 1]), 2)
co2_emision_mean = round(np.mean(automobile_emission_data[:, 2]), 2)
engine_size_std = round(np.std(automobile_emission_data[:, 0]), 2)
car_weight_std = round(np.std(automobile_emission_data[:, 1]), 2)
co2_emission_std = round(np.std(automobile_emission_data[:, 2]), 2)
automobile_emission_data[:, 0] = (automobile_emission_data[:, 0] - engine_size_mean) / engine_size_std
automobile_emission_data[:, 1] = (automobile_emission_data[:, 1] - car_weight_mean) / car_weight_std
automobile_emission_data[:, 2] = (automobile_emission_data[:, 2] - co2_emision_mean) / co2_emission_std
automobile_emission_data = np.concatenate((np.ones((len(automobile_emission_data),1)), automobile_emission_data), axis=1)
print("Engine Size: Mean = " + str(engine_size_mean) + " , Std = " + str(engine_size_std))
print("Car Weight: Mean = " + str(car_weight_mean) + " , Std = " + str(car_weight_std))
print("CO2 Emission: Mean = " + str(co2_emision_mean) + " , Std = " + str(co2_emission_std))
print("Size of X: " + str(len(automobile_emission_data[:, :3][0])))
print("Size of y: " + str(len(automobile_emission_data[:, 3:][0])))

# 5b
theta_gd_5, cost_gd_5 = gradientDescent(automobile_emission_data[:, :3], automobile_emission_data[:, 3:], 0.01, 750)
plt.title("Cost from gradient descent (alpha = 0.01)")
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.scatter(list(range(1, 751)), cost_gd_5)
plt.savefig("output/ps2-5-b")
plt.close()
print("Estimate of theta:")
print(theta_gd_5)

# 5c
engine_size_test = (2300.0 - engine_size_mean) / engine_size_std
car_weight_test = (1300.0 - car_weight_mean)/ car_weight_std
best_fit_line_gd_5 = round(theta_gd_5[0][0] + theta_gd_5[1][0]*engine_size_test + theta_gd_5[2][0]*car_weight_test, 2)
print("Prediction for co2 emission: " + str((best_fit_line_gd_5 * co2_emission_std) + co2_emision_mean))