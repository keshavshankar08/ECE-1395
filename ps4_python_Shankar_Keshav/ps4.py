import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Reg_normalEqn import *
from costFunction import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from logReg_multi import *
import pandas as pd

# ----- Problem 1 -----
print("----- Problem 1 -----")
# load data1
data1 = loadmat("input/hw4_data1.mat")
X_data1 = np.array(data1["X_data"])
y_data1 = np.array(data1["y"])

# 1b
X_data1 = np.concatenate((np.ones((len(X_data1), 1)), X_data1), axis=1)
print("X has " + str(len(X_data1[0])) + " features and " + str(len(X_data1)) + " data points.")

# 1c
lambdas = [0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017]
thetas = np.zeros((20,8,501,1))
training_errors = np.zeros((20,8))
testing_errors = np.zeros((20,8))
for i in range(20):
    indices = np.arange(len(X_data1))
    np.random.shuffle(indices)
    X_shuffled = X_data1[indices]
    y_shuffled = y_data1[indices]
    train_split = round(len(X_data1) * 0.85)
    test_split = len(X_data1) - train_split
    X_train = np.array(X_shuffled[:train_split, :])
    X_test = np.array(X_shuffled[-test_split:])
    y_train = np.array(y_shuffled[:train_split, :])
    y_test = np.array(y_shuffled[-test_split:])
    for j in range (8):
        thetas[i, j, :, :] = Reg_normalEqn(X_train, y_train, lambdas[j])
        training_errors[i, j] = computeCost(X_train, y_train, thetas[i, j, :, :])
        testing_errors[i, j] = computeCost(X_test, y_test, thetas[i, j, :, :])
training_errors_avg = np.mean(training_errors, axis=0)
testing_errors_avg = np.mean(testing_errors, axis=0)
plt.title("Average Error vs. Lambda")
plt.xlabel("Lambda")
plt.ylabel("Average Error")
plt.plot(lambdas, testing_errors_avg, c='blue', marker='o', label='testing error')
plt.plot(lambdas, training_errors_avg, c='red', marker='*', label='training error')
plt.savefig("output/ps4-1-a")
plt.close()


# ----- Problem 2 -----
print("----- Problem 2 -----")

# load data2
data2 = loadmat("input/hw4_data2.mat")
X1_data2 = np.array(data2["X1"])
X2_data2 = np.array(data2["X2"])
X3_data2 = np.array(data2["X3"])
X4_data2 = np.array(data2["X4"])
X5_data2 = np.array(data2["X5"])
y1_data2 = np.array(data2["y1"])
y2_data2 = np.array(data2["y2"])
y3_data2 = np.array(data2["y3"])
y4_data2 = np.array(data2["y4"])
y5_data2 = np.array(data2["y5"])

# 2a
X_train1 = [X1_data2, X2_data2, X3_data2, X4_data2]
X_train2 = [X1_data2, X2_data2, X3_data2, X5_data2]
X_train3 = [X1_data2, X2_data2, X4_data2, X5_data2]
X_train4 = [X1_data2, X3_data2, X4_data2, X5_data2]
X_train5 = [X2_data2, X3_data2, X4_data2, X5_data2]

y_train1 = [y1_data2, y2_data2, y3_data2, y4_data2]
y_train2 = [y1_data2, y2_data2, y3_data2, y5_data2]
y_train3 = [y1_data2, y2_data2, y4_data2, y5_data2]
y_train4 = [y1_data2, y3_data2, y4_data2, y5_data2]
y_train5 = [y2_data2, y3_data2, y4_data2, y5_data2]

X_test1 = X5_data2
X_test2 = X4_data2
X_test3 = X3_data2
X_test4 = X2_data2
X_test5 = X1_data2

y_test1 = y5_data2
y_test2 = y4_data2
y_test3 = y3_data2
y_test4 = y2_data2
y_test5 = y1_data2

accuracies = np.zeros((5,8))
k_arr_pos = 0

# compute for k=1 through k=15 by odds
for k in range(1, 16, 2):
    model1 = KNeighborsClassifier(k)
    model1.fit(np.concatenate(X_train1, axis=0), np.concatenate(y_train1, axis=0).ravel())
    predictions = model1.predict(X_test1)
    accuracy = accuracy_score(y_test1, predictions)
    accuracies[0, k_arr_pos] = accuracy

    model2 = KNeighborsClassifier(k)
    model2.fit(np.concatenate(X_train2, axis=0), np.concatenate(y_train2, axis=0).ravel())
    predictions = model2.predict(X_test2)
    accuracy = accuracy_score(y_test2, predictions)
    accuracies[1, k_arr_pos] = accuracy

    model3= KNeighborsClassifier(k)
    model3.fit(np.concatenate(X_train3, axis=0), np.concatenate(y_train3, axis=0).ravel())
    predictions = model3.predict(X_test3)
    accuracy = accuracy_score(y_test3, predictions)
    accuracies[2, k_arr_pos] = accuracy

    model4 = KNeighborsClassifier(k)
    model4.fit(np.concatenate(X_train4, axis=0), np.concatenate(y_train4, axis=0).ravel())
    predictions = model4.predict(X_test4)
    accuracy = accuracy_score(y_test4, predictions)
    accuracies[3, k_arr_pos] = accuracy

    model5 = KNeighborsClassifier(k)
    model5.fit(np.concatenate(X_train5, axis=0), np.concatenate(y_train5, axis=0).ravel())
    predictions = model5.predict(X_test5)
    accuracy = accuracy_score(y_test5, predictions)
    accuracies[4, k_arr_pos] = accuracy

    k_arr_pos += 1
accuracies_average = np.mean(accuracies, axis=0)
plt.title("Average Accuracy vs K")
plt.xlabel("K")
plt.ylabel("Average Accuracy")
k_arr = [1,3,5,7,9,11,13,15]
plt.plot(k_arr, accuracies_average, c='blue', marker='o')
plt.savefig("output/ps4-2-a")
plt.close()

# ----- Problem 3 -----
print("----- Problem 3 -----")
# load data3
data3 = loadmat("input/hw4_data3.mat")
X_test_data3 = np.array(data3["X_test"])
X_train_data3 = np.array(data3["X_train"])
y_test_data3 = np.array(data3["y_test"])
y_train_data3 = np.array(data3["y_train"])

# 3b
y_train_prediction = logReg_multi(X_train_data3, y_train_data3, X_train_data3)
y_test_prediction = logReg_multi(X_train_data3, y_train_data3, X_test_data3)
train_accuracy = accuracy_score(y_train_data3, y_train_prediction)
test_accuracy = accuracy_score(y_test_data3, y_test_prediction)
accuracy_table = pd.DataFrame({
    'Dataset': ['Training', 'Testing'],
    'Accuracy': [train_accuracy, test_accuracy]
})
print(accuracy_table)