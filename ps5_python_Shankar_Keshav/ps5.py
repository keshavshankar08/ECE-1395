import numpy as np
from scipy.io import loadmat
from weightedKNN import *
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import os
import shutil
import random
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import time
from sklearn.svm import SVC
import pandas as pd

# ----- Problem 1 -----
print("----- Problem 1 -----")
# load data3
data3 = loadmat("input/hw4_data3.mat")
X_test_data3 = np.array(data3["X_test"])
X_train_data3 = np.array(data3["X_train"])
y_test_data3 = np.array(data3["y_test"])
y_train_data3 = np.array(data3["y_train"])

# sigmas to test with
sigma_values = [0.01, 0.07, 0.15, 1.5, 3, 4.5]

# test for every sigma
test_accuracy = []
for sigma in sigma_values:
    y_predict = weightedKNN(X_train_data3, y_train_data3, X_test_data3, sigma)
    test_accuracy.append(accuracy_score(y_test_data3, y_predict))

# print table of scores
print("Sigma    Accuracy")
for sigma, accuracy in zip(sigma_values, test_accuracy):
    print(f"{sigma:<8} {accuracy}")


# ----- Problem 2 -----
print("----- Problem 2 -----")
# ----- 2.0 -----
# clear test and train directories
shutil.rmtree("input/train", ignore_errors=True)
shutil.rmtree("input/test", ignore_errors=True)

# make test and train directories
os.makedirs("input/train", exist_ok=True)
os.makedirs("input/test", exist_ok=True)

# store test and train data to directories
for person_number in range(1, 41):
    person_folder = os.path.join("input/all", f"s{person_number}")
    images = [f"{i}.pgm" for i in range(1,11)]
    train_images = random.sample(images, 8)

    for img in train_images:
        shutil.copy(os.path.join(person_folder, img), os.path.join("input/train", f"{person_number}_{img}"))
    
    for img in images:
        if img not in train_images:
            shutil.copy(os.path.join(person_folder, img), os.path.join("input/test", f"{person_number}_{img}"))

# subplot of 3 random images in train
subplot_images = random.sample(os.listdir("input/train"), 3)
fig, axes = plt.subplots(1, 3, figsize=(12,4))
for i, image_name in enumerate(subplot_images):
    img = mpimg.imread(os.path.join("input/train", image_name))
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(image_name)
plt.savefig("output/ps5-2-0.png")
plt.close()

# ----- 2.1 -----
# a
T = np.zeros((10304,320))
train_data = [f for f in os.listdir("input/train")]
for i, image_name in enumerate(train_data): # flatten every image and add to T
    image = mpimg.imread(os.path.join("input/train", image_name))
    T[:, i] = image.flatten()
plt.imshow(T, cmap='gray', aspect='auto')
plt.savefig('output/ps5-1-a.png', bbox_inches='tight')
plt.close()

# b
m = np.mean(T, axis=1)
m_reshaped = m.reshape((112, 92))
plt.imshow(m_reshaped, cmap='gray')
plt.savefig('output/ps5-2-1-b.png', bbox_inches='tight')
plt.close()

# c
m_orig_dim = np.mean(T, axis=1, keepdims=True)
A = T - m_orig_dim
C = A @ np.transpose(A)
plt.imshow(C, cmap='gray')
plt.savefig('output/ps5-2-1-c.png', bbox_inches='tight')
plt.close()

# d
AT_A = np.transpose(A) @ A
eigen_values, eigen_vectors = np.linalg.eig(AT_A) # used eig here instead of eigh since no need for sorted
eigen_values_sorted = np.sort(eigen_values)[::-1] # sort it
variance = np.sum(eigen_values_sorted)
variance_percent = np.cumsum(eigen_values_sorted)/variance
k = np.argmax(variance_percent >= 0.95) + 1
print("K for 95%: " + str(k))
plt.plot(variance_percent)
plt.xlabel('k')
plt.ylabel('v(k)')
plt.title('k vs v(k)')
plt.savefig('output/ps5-2-1-d.png')
plt.close()

# e
eigen_values, eigen_vectors = np.linalg.eigh(C) # used eigh here so garauntee sort
eigen_values_sorted = eigen_values[np.argsort(eigen_values)[::-1]]
eigen_vectors_sorted = eigen_vectors[:, np.argsort(eigen_values)[::-1]]
U = eigen_vectors_sorted[:, :k]

# plot 9 faces using subplots
fig, axes = plt.subplots(3, 3, figsize=(8,8))
for i in range(9):
    eigen_face = U[:, i].reshape(112,92)
    axes[i//3, i%3].imshow(eigen_face, cmap='gray')
plt.savefig('output/ps5-2-1-e.png', bbox_inches='tight')
print("U has " + str(len(U)) + " rows and " + str(len(U[0])) + " columns.")

# ----- 2.2 -----
# a
W_training = []
y_train_labels = []
for image_name in train_data: # generate each w and corresponding y for train
    image = mpimg.imread(os.path.join("input/train", image_name))
    I = image.flatten()
    w = np.transpose(U) @ (I - m)
    W_training.append(w)
    y_train_labels.append(int((image_name.split("."))[0].split("_")[0]))
W_training = np.array(W_training)
y_train_labels = np.array(y_train_labels)

# b
W_testing = []
y_test_labels = []
test_data = [f for f in os.listdir("input/test")]
for image_name in test_data: # generate each w and corresponding y for test
    image = mpimg.imread(os.path.join("input/test", image_name))
    I = image.flatten()
    w = np.transpose(U) @ (I - m)
    W_testing.append(w)
    y_test_labels.append(int((image_name.split("."))[0].split("_")[0]))
W_testing = np.array(W_testing)
y_test_labels = np.array(y_test_labels)
print("W_training has " + str(len(W_training)) + " rows and " + str(len(W_training[0])) + " columns.")
print("W_testing has " + str(len(W_testing)) + " rows and " + str(len(W_testing[0])) + " columns.")

# ----- 2.3 -----
# a
k_values = [1, 3, 5, 7, 9, 11]
for k in k_values: # run knn for each k value
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(W_training, y_train_labels)
    predictions = knn.predict(W_testing)
    accuracy = accuracy_score(y_test_labels, predictions)
    print("K: " + str(k) + ", Accuracy: " + str(accuracy))

# b
training_times = []
testing_times = []
testing_accuracies = []

# linear kernel one vs one
start_time = time.time() # measuring train time
svm_classifier = SVC(kernel='linear', degree=3, decision_function_shape='ovo', C=1.0)
svm_classifier.fit(W_training, y_train_labels)
end_time = time.time()
elapsed_time = end_time - start_time # this is the train time
training_times.append(elapsed_time)
start_time = time.time() # measuring test time
y_pred = svm_classifier.predict(W_testing)
end_time = time.time()
elapsed_time = end_time - start_time # this is the test time
testing_times.append(elapsed_time)
testing_accuracies.append(accuracy_score(y_test_labels, y_pred))

# linear kernel one vs all
start_time = time.time()
svm_classifier = SVC(kernel='linear', degree=3, decision_function_shape='ovr', C=1.0)
svm_classifier.fit(W_training, y_train_labels)
end_time = time.time()
elapsed_time = end_time - start_time
training_times.append(elapsed_time)
start_time = time.time()
y_pred = svm_classifier.predict(W_testing)
end_time = time.time()
elapsed_time = end_time - start_time
testing_times.append(elapsed_time)
testing_accuracies.append(accuracy_score(y_test_labels, y_pred))

# 3rd order polynomial one vs one
start_time = time.time()
svm_classifier = SVC(kernel='poly', degree=3, decision_function_shape='ovo', C=1.0)
svm_classifier.fit(W_training, y_train_labels)
end_time = time.time()
elapsed_time = end_time - start_time
training_times.append(elapsed_time)
start_time = time.time()
y_pred = svm_classifier.predict(W_testing)
end_time = time.time()
elapsed_time = end_time - start_time
testing_times.append(elapsed_time)
testing_accuracies.append(accuracy_score(y_test_labels, y_pred))

# 3rd order polynomial one vs all
start_time = time.time()
svm_classifier = SVC(kernel='poly', degree=3, decision_function_shape='ovr', C=1.0)
svm_classifier.fit(W_training, y_train_labels)
end_time = time.time()
elapsed_time = end_time - start_time
training_times.append(elapsed_time)
start_time = time.time()
y_pred = svm_classifier.predict(W_testing)
end_time = time.time()
elapsed_time = end_time - start_time
testing_times.append(elapsed_time)
testing_accuracies.append(accuracy_score(y_test_labels, y_pred))

# RBF kernel one vs one
start_time = time.time()
svm_classifier = SVC(kernel='rbf', degree=3, decision_function_shape='ovo', C=1.0)
svm_classifier.fit(W_training, y_train_labels)
end_time = time.time()
elapsed_time = end_time - start_time
training_times.append(elapsed_time)
start_time = time.time()
y_pred = svm_classifier.predict(W_testing)
end_time = time.time()
elapsed_time = end_time - start_time
testing_times.append(elapsed_time)
testing_accuracies.append(accuracy_score(y_test_labels, y_pred))

# RBF kernel one vs all
start_time = time.time()
svm_classifier = SVC(kernel='rbf', degree=3, decision_function_shape='ovr', C=1.0)
svm_classifier.fit(W_training, y_train_labels)
end_time = time.time()
elapsed_time = end_time - start_time
training_times.append(elapsed_time)
start_time = time.time()
y_pred = svm_classifier.predict(W_testing)
end_time = time.time()
elapsed_time = end_time - start_time
testing_times.append(elapsed_time)
testing_accuracies.append(accuracy_score(y_test_labels, y_pred))

data_training_times = pd.DataFrame({
    'Kernel': ['Linear', 'Linear', 'Poly', 'Poly', 'RBF', 'RBF'],
    'Paradigm': ['OVR', 'OVO', 'OVR', 'OVO', 'OVR', 'OVO'],
    'Training Time (s)': training_times
})
print(data_training_times)

data_testing_times = pd.DataFrame({
    'Kernel': ['Linear', 'Linear', 'Poly', 'Poly', 'RBF', 'RBF'],
    'Paradigm': ['OVR', 'OVO', 'OVR', 'OVO', 'OVR', 'OVO'],
    'Testing Time (s)': testing_times
})
print(data_testing_times)

data_testing_accuracies = pd.DataFrame({
    'Kernel': ['Linear', 'Linear', 'Poly', 'Poly', 'RBF', 'RBF'],
    'Paradigm': ['OVR', 'OVO', 'OVR', 'OVO', 'OVR', 'OVO'],
    'Accuracy': testing_accuracies
})
print(data_testing_accuracies)


# ----- Problem 3 -----
print("----- Problem 3 -----")
