import numpy as np
from kmeans_single import *

def kmneans_multiple(X, k, iters, R):
    # initialize holders for best values
    ssd_ideal = np.inf
    ids_ideal = np.zeros((X.shape[0],1))
    means_ideal = np.zeros((k,X.shape[1]))
    
    # loop for num restarts
    for i in range(R):
        # call kmenas clustering
        ids, means, ssd = kmneans_single(X, k, iters)
        
        # check if ssd less than ideal, if so, update
        if ssd < ssd_ideal:
            ssd_ideal = ssd
            ids_ideal = ids
            means_ideal = means
    
    return ids_ideal, means_ideal, ssd_ideal