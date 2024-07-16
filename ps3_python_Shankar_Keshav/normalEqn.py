import numpy as np

# Computes closed-form solution using normal equation
def normalEqn(X_train, y_train):
    # compute (X^T * X)^-1 * (X^T * y)
    X_train_transposed = np.transpose(X_train)
    term1 = np.linalg.pinv(np.dot(X_train_transposed, X_train))
    term2 = np.dot(X_train_transposed, y_train)
    theta = np.dot(term1, term2)
    return theta