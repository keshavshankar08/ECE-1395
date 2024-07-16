import numpy as np
from scipy.spatial.distance import cdist

# Returns prediction on test set using weighted KNN
def weightedKNN(X_train, y_train, X_test, sigma):
    # var to store predictions
    y_predict = np.zeros((len(X_test), 1))
    
    distances = cdist(X_test, X_train, metric='euclidean')
    weights = np.exp(-(distances**2) / (sigma**2))
    
    # loop through points and find distances, calculate weights, then predict based on weights
    for i in range(len(X_test)):
        weighted_vote = np.zeros(np.max(y_train) + 1)
        for j in range(len(X_train)):
            weighted_vote[y_train[j]] += weights[i,j]
        y_predict[i, 0] = np.argmax(weighted_vote)

    return y_predict