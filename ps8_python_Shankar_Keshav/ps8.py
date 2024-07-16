import numpy as np
from scipy.io import *
import matplotlib.pyplot as plt
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ----- Problem 1 -----
# 1a
mnist_data = loadmat("input/HW8_data1.mat")
X = mnist_data["X"]
y = mnist_data["y"]
random_indices = np.random.choice(X.shape[0], 25, replace=False)
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    image = X[random_indices[i]].reshape(20, 20).T
    ax.imshow(image, cmap='gray')
    ax.set_title(str(y[random_indices[i]]))
    ax.axis('off')
plt.savefig("output/ps8-1-a.png")
plt.close()

# 1b
indices = np.arange(len(X))
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]
train_split = 4500
test_split = 500
X_train = X_shuffled[:train_split, :]
X_test = X_shuffled[-test_split:, :]
y_train = y_shuffled[:train_split, :]
y_test = y_shuffled[-test_split:, :]

# 1c
X_train_subset = []
y_train_subset = []
for i in range(5):
    subset_indicies = np.random.choice(X_train.shape[0], 1000, replace=True)
    X_train_subset.append(X_train[subset_indicies])
    y_train_subset.append(y_train[subset_indicies])

for i in range(len(X_train_subset)):
    savemat(f"input/subset_{i+1}.mat", {"X": X_train_subset[i], "y": y_train_subset[i]})

# 1d
print("\nOne-vs-All SVM:")
classifier_SVM_1vsAll = SVC(kernel='rbf')
classifier_SVM_1vsAll.fit(X_train_subset[0], y_train_subset[0].ravel())

error_train_SVM_1vsAll = []
prediction_x1_train_SVM_1vsAll = classifier_SVM_1vsAll.predict(X_train_subset[0])
error_x1_train_SVM_1vsAll = round(1 - accuracy_score(y_train_subset[0], prediction_x1_train_SVM_1vsAll), 2)
error_train_SVM_1vsAll.append(error_x1_train_SVM_1vsAll)
for subset_X, subset_y in zip(X_train_subset[1:], y_train_subset[1:]):
    prediction_train = classifier_SVM_1vsAll.predict(subset_X)
    error_train_other = 1 - accuracy_score(subset_y, prediction_train)
    error_train_SVM_1vsAll.append(round(error_train_other, 2))

predictions_test_SVM_1vsAll = classifier_SVM_1vsAll.predict(X_test)
error_test_SVM_1vsAll = round(1 - accuracy_score(y_test, predictions_test_SVM_1vsAll), 2)

print("Classification error on training set: ", error_train_SVM_1vsAll)
print("Classification error on testing set: ", error_test_SVM_1vsAll)

# 1e
print("\nKNN:")
classifier_KNN = KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train_subset[1], y_train_subset[1].ravel())

error_train_KNN = []
prediction_x2_train_KNN = classifier_KNN.predict(X_train_subset[1])
error_x2_train_KNN = round(1 - accuracy_score(y_train_subset[1], prediction_x2_train_KNN), 2)
error_train_KNN.append(error_x2_train_KNN)
for subset_X, subset_y in zip(X_train_subset[0:1] + X_train_subset[2:], y_train_subset[0:1] + y_train_subset[2:]):
    prediction_train = classifier_KNN.predict(subset_X)
    error_train_other = 1 - accuracy_score(subset_y, prediction_train)
    error_train_KNN.append(round(error_train_other, 2))

predictions_test_KNN = classifier_KNN.predict(X_test)
error_test_KNN = round(1 - accuracy_score(y_test, predictions_test_KNN), 2)

print("Classification error on training set: ", error_train_KNN)
print("Classification error on testing set: ", error_test_KNN)

# 1f
print("\nLogistic Regression:")
classifier_logisticRegression = LogisticRegression(max_iter=1000)
classifier_logisticRegression.fit(X_train_subset[2], y_train_subset[2].ravel())

error_train_logisticRegression = []
prediction_x3_train_logisticRegression = classifier_logisticRegression.predict(X_train_subset[2])
error_x3_train_logisticRegression = round(1 - accuracy_score(y_train_subset[2], prediction_x3_train_logisticRegression), 2)
error_train_logisticRegression.append(error_x3_train_logisticRegression)
for subset_X, subset_y in zip(X_train_subset[0:2] + X_train_subset[3:], y_train_subset[0:2] + y_train_subset[3:]):
    prediction_train = classifier_logisticRegression.predict(subset_X)
    error_train_other = 1 - accuracy_score(subset_y, prediction_train)
    error_train_logisticRegression.append(round(error_train_other, 2))

predictions_test_logisticRegression = classifier_logisticRegression.predict(X_test)
error_test_logisticRegression = round(1 - accuracy_score(y_test, predictions_test_logisticRegression), 2)

print("Classification error on training set: ", error_train_logisticRegression)
print("Classification error on testing set: ", error_test_logisticRegression)

# 1g
print("\nDecision Tree:")
classifier_decisionTree = DecisionTreeClassifier()
classifier_decisionTree.fit(X_train_subset[3], y_train_subset[3].ravel())

error_train_decisionTree = []
prediction_x4_train_decisionTree = classifier_decisionTree.predict(X_train_subset[3])
error_x4_train_decisionTree = round(1 - accuracy_score(y_train_subset[3], prediction_x4_train_decisionTree), 2)
error_train_decisionTree.append(error_x4_train_decisionTree)
for subset_X, subset_y in zip(X_train_subset[0:3] + X_train_subset[4:], y_train_subset[0:3] + y_train_subset[4:]):
    prediction_train = classifier_decisionTree.predict(subset_X)
    error_train_other = 1 - accuracy_score(subset_y, prediction_train)
    error_train_decisionTree.append(round(error_train_other, 2))

predictions_test_decisionTree = classifier_decisionTree.predict(X_test)
error_test_decisionTree = round(1 - accuracy_score(y_test, predictions_test_decisionTree), 2)

print("Classification error on training set: ", error_train_decisionTree)
print("Classification error on testing set: ", error_test_decisionTree)

# 1h
print("\nRandom Forest:")
classifier_randomForest = RandomForestClassifier(n_estimators=85)
classifier_randomForest.fit(X_train_subset[4], y_train_subset[4].ravel())

error_train_randomForest = []
prediction_x5_train_randomForest = classifier_randomForest.predict(X_train_subset[3])
error_x5_train_randomForest = round(1 - accuracy_score(y_train_subset[4], prediction_x5_train_randomForest), 2)
error_train_randomForest.append(error_x5_train_randomForest)
for subset_X, subset_y in zip(X_train_subset[0:4], y_train_subset[0:4]):
    prediction_train = classifier_randomForest.predict(subset_X)
    error_train_other = 1 - accuracy_score(subset_y, prediction_train)
    error_train_randomForest.append(round(error_train_other, 2))

predictions_test_randomForest = classifier_randomForest.predict(X_test)
error_test_randomForest = round(1 - accuracy_score(y_test, predictions_test_randomForest), 2)

print("Classification error on training set: ", error_train_randomForest)
print("Classification error on testing set: ", error_test_randomForest)

# 1i
print("\nCombined:")
predictions_test_combined = []
for i in range(len(X_test)):
    votes = [predictions_test_SVM_1vsAll[i], predictions_test_KNN[i], predictions_test_logisticRegression[i], predictions_test_decisionTree[i], predictions_test_randomForest[i]]
    majority_vote = max(set(votes), key=votes.count)
    predictions_test_combined.append(majority_vote)
error_test_combined = round(1 - accuracy_score(y_test, predictions_test_combined), 2)

print("Classification error on testing: ", error_test_combined)