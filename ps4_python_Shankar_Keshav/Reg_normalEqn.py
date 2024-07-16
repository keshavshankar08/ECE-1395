import numpy as np

# Calculates closed form solution for regularized linear regression
def Reg_normalEqn(X_train, y_train, _lambda):
    m = len(X_train)
    regularized = np.eye(len(X_train[0]))
    regularized[0,0] = 0
    regularized *= _lambda
    theta = np.linalg.pinv((np.transpose(X_train) @ X_train) + regularized) @ np.transpose(X_train) @ y_train

    return theta