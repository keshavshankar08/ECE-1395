import numpy as np
from sklearn.linear_model import LogisticRegression

# performs one vs all approach using logistic regression
def logReg_multi(X_train, y_train, X_test):
    # define vars
    num_classes = 3 # C
    classifiers = [] # models for each class
    predictions = np.zeros((len(X_test),num_classes)) # probabilities

    # for "C" classifiers
    for i in range(1, num_classes + 1):
        y_binary = (y_train ==i).astype(int)
        mdl_c = LogisticRegression(random_state=0).fit(X_train, y_binary.ravel())
        classifiers.append(mdl_c)
        proba_c = mdl_c.predict_proba(X_test)[:, 1]
        predictions[:, i - 1] = proba_c

    # get highest prob calss for each sample
    y_predict = np.argmax(predictions, axis=1) + 1

    return y_predict