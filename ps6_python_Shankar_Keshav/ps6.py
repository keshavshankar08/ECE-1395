import numpy as np
import csv
import pandas as pd
import math


# ----- Problem 0 -----
print("----- Problem 0 -----")

# 0a
with open('input/iris_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    data0 = list(reader)
data0 = np.array(data0, dtype=float)
X0 = data0[:, :4]
y0 = data0[:, 4:]

# 0b
indices = np.arange(len(X0))
np.random.shuffle(indices)
X0_shuffled = X0[indices]
Y0_shuffled = y0[indices]
train_split = 125
test_split = 25
X0_train = X0_shuffled[:train_split, :]
X0_test = X0_shuffled[-test_split:, :]
Y0_train = Y0_shuffled[:train_split, :]
Y0_test = Y0_shuffled[-test_split:, :]

# 0c
X0_train_1 = X0_train[Y0_train[:, 0] == 1]
X0_train_2 = X0_train[Y0_train[:, 0] == 2]
X0_train_3 = X0_train[Y0_train[:, 0] == 3]
print("Size of X_train_1: " + str(X0_train_1.shape[0]))
print("Size of X_train_2: " + str(X0_train_2.shape[0]))
print("Size of X_train_3: " + str(X0_train_3.shape[0]))


# ----- Problem 1 -----
print("\n----- Problem 1 -----")

# 1a
mean_class1 = np.mean(X0_train_1, 0)
mean_class2 = np.mean(X0_train_2, 0)
mean_class3 = np.mean(X0_train_3, 0)
stdv_class1 = np.std(X0_train_1, 0)
stdv_class2 = np.std(X0_train_2, 0)
stdv_class3 = np.std(X0_train_3, 0)
df_class1 = pd.DataFrame({'Mean': mean_class1, 'Stdv': stdv_class1})
df_class2 = pd.DataFrame({'Mean': mean_class2, 'Stdv': stdv_class2})
df_class3 = pd.DataFrame({'Mean': mean_class3, 'Stdv': stdv_class3})
df_class1.index = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
df_class2.index = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
df_class3.index = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
print("Class 1: \n" + str(df_class1) + "\n")
print("Class 2: \n" + str(df_class2) + "\n")
print("Class 3: \n" + str(df_class3) + "\n")

# 1b
correct_predictions = 0
for i in range(len(X0_test)):
    # b1
    p_w1 = np.zeros((1,4))
    p_w2 = np.zeros((1,4))
    p_w3 = np.zeros((1,4))
    for j in range(len(X0_test[0])):
        p_w1[0][j] = (1 / (math.sqrt(2*math.pi)*stdv_class1[j])) * math.exp(-((X0_test[i][j] - mean_class1[j])**2)/(2*(stdv_class1[j]**2)))
        p_w2[0][j] = (1 / (math.sqrt(2*math.pi)*stdv_class2[j])) * math.exp(-((X0_test[i][j] - mean_class2[j])**2)/(2*(stdv_class2[j]**2)))
        p_w3[0][j] = (1 / (math.sqrt(2*math.pi)*stdv_class3[j])) * math.exp(-((X0_test[i][j] - mean_class3[j])**2)/(2*(stdv_class3[j]**2)))

    # b2
    ln_p_w1 = 0
    ln_p_w2 = 0
    ln_p_w3 = 0
    for k in range(len(X0_test[0])):
        ln_p_w1 += math.log(p_w1[0][k])
        ln_p_w2 += math.log(p_w2[0][k])
        ln_p_w3 += math.log(p_w3[0][k])

    # b3
    ln_p_x = np.zeros((1,3))
    ln_p_x[0][0] = ln_p_w1 + math.log(1.0/3.0)
    ln_p_x[0][1] = ln_p_w2 + math.log(1.0/3.0)
    ln_p_x[0][2] = ln_p_w3 + math.log(1.0/3.0)

    # b4
    posterior_probability = np.argmax(ln_p_x) + 1
    if(posterior_probability == Y0_test[i]):
        correct_predictions += 1

accuracy = correct_predictions/len(X0_test) * 100
print("Accuracy: " + str(accuracy) + "%")


# ----- Problem 2 -----
print("\n----- Problem 2 -----")

# 2a
covariance_1 = np.cov(np.transpose(X0_train_1))
covariance_2 = np.cov(np.transpose(X0_train_2))
covariance_3 = np.cov(np.transpose(X0_train_3))
print("Covariance matrix 1 size: " + str(covariance_1.shape))
print(covariance_1)
print("\nCovariance matrix 2 size: " + str(covariance_2.shape))
print(covariance_2)
print("\nCovariance matrix 3 size: " + str(covariance_3.shape))
print(covariance_3)

# 2b
print("\nClass 1 mean vector: " + str(mean_class1.shape))
print(mean_class1)
print("\nClass 2 mean vector: " + str(mean_class2.shape))
print(mean_class2)
print("\nClass 3 mean vector: " + str(mean_class3.shape))
print(mean_class3)

# 2c
correct_predictions = 0
for i in range(len(X0_test)):
    # c1
    g = np.zeros((1,3))
    for j in range(len(X0_test[0])):
        g[0][0] = (-((X0_test[i][j] - mean_class1[j])**2)/(2*(stdv_class1[j]**2))) - math.log(stdv_class1[j]) - (.5*math.log(2*math.pi))
        g[0][1] = (-((X0_test[i][j] - mean_class2[j])**2)/(2*(stdv_class2[j]**2))) - math.log(stdv_class2[j]) - (.5*math.log(2*math.pi))
        g[0][2] = (-((X0_test[i][j] - mean_class2[j])**2)/(2*(stdv_class3[j]**2))) - math.log(stdv_class3[j]) - (.5*math.log(2*math.pi))

    # c2
    posterior_probability = np.argmax(g) + 1
    if(posterior_probability == Y0_test[i]):
        correct_predictions += 1

accuracy = correct_predictions/len(X0_test) * 100
print("\nAccuracy: " + str(accuracy) + "%")