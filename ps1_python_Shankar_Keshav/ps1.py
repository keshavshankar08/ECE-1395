import numpy as np
import matplotlib.pyplot as plt
import time

# ----- 3 - Basic Operations -----
print("\n----- Problem 3 -----")
# 3a
mean = 1.5
std = 0.6
x = mean + std + np.random.randn(1000000, 1)

# 3b
z = np.random.uniform(-1, 3, (1000000,1))

# 3c
plt.hist(x, bins=1000)
plt.title("vector x gaussian distribution")
plt.savefig("output/ps1-3-c-1.png")
plt.close()

plt.hist(z, bins=1000)
plt.title("vector z uniform distribution")
plt.savefig("output/ps1-3-c-2.png")
plt.close()

# 3d
vector_size = x.shape[0]
start_time = time.time()
for i in range(vector_size):
    x[i] += 1
end_time = time.time()
elapsed_time = round(end_time - start_time, 2)
print("Time to run 3d: " + str(elapsed_time) + " seconds")

# 3e
start_time = time.time()
x += 1
end_time = time.time()
elapsed_time = round(end_time - start_time, 5)
print("Time to run 3e: " + str(elapsed_time) + " seconds")

# 3f
y = z[(z > 0) & (z < 1.5)]
print("Elements retrieved in 3f: " + str(y.shape[0]))
'''

'''
# ----- 4 - Linear Algebra -----
print("\n----- Problem 4 -----")
# 4a
A = np.array([[2,1,3],[2,6,8],[6,8,18]])
min_of_each_column = np.min(A, axis=0)
max_of_each_row = np.max(A, axis=1)
max = np.max(A)
sum_of_each_column = np.sum(A, axis=0)
sum = np.sum(A)
B = np.square(A)

# 4b
solution_vector = np.array([[1],[3],[5]])
result_vector = np.round(np.dot(np.linalg.inv(A), solution_vector), decimals=2)

print("The resultant solution for 4b is: \n" + str(result_vector))

# 4c
x1 = np.array([-0.5, 0 , 1.5])
x2 = np.array([-1, -1, 0])
norm_x1 = np.linalg.norm(x1)
norm_x2 = np.linalg.norm(x2)
print("Norm of x1 for 4c is: " + str(round(norm_x1, 2)))
print("Norm of x2 for 4c is: " + str(round(norm_x2, 2)))



# ----- 5 - Splitting Data -----
print("\n----- Problem 5 -----")
# 5a
X = (np.arange(10) + 1).reshape((10, 1)) * np.ones((1, 3), dtype=int)
y = (np.arange(10) + 1).reshape((10, 1)) * np.ones((1, 1), dtype=int)
print("X for 5a is: \n" + str(X))

# 5b, 5c
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

x_shuffled = X[indices]
y_shuffled = y[indices]

X_train = x_shuffled[:8, :].copy()
X_test = x_shuffled[-2:].copy()

Y_train = y_shuffled[:8, :].copy()
Y_test = y_shuffled[-2:].copy()

print("X train: \n" + str(X_train))
print("Y train: \n" + str(Y_train))
print("X test: \n" + str(X_test))
print("Y test: \n" + str(Y_test))
