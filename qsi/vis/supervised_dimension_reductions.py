import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from .plotComponents2D import *

def pls(X, y, k = 2):

    pls = PLSRegression(n_components=k, scale = False)
    X_pls = pls.fit(X, y).transform(X)

    plotComponents2D(X_pls, y) # , tags = range(len(y)), ax = ax
    print('PLS score = ', np.round(pls.score(X, y),3)) # Returns the coefficient of determination R^2 of the prediction.
    return X_pls