import numpy as np

def kmneans_single(X, k, iters):    
    # find min and max to get range of features
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    # use min and max to randomly assign k centroids
    means = np.random.uniform(mins, maxs, size=(k, X.shape[1]))
    
    # loop through for defined iterations
    for i in range(iters):
        # find distance from each point to k centroids
        distances = np.linalg.norm(X[:, np.newaxis] - means, axis=2)

        # find the minimum distance from each point to k centroids and assign label using index
        # ex: k = 3 -> distances[0] = distance to centroid 1, ...
        ids = np.argmin(distances, axis=1)
        
        # loop through centroids and update them by finding average of points in each cluster
        for k in range(k):
            if np.sum(ids == k) > 0:
                means[k] = np.mean(X[ids == k], axis=0)
    
    # compute sum of squared distances
    ssd = np.sum(np.min(distances, axis=1)**2)
    
    return ids, means, ssd