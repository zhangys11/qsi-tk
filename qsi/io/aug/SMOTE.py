import random
import pandas as pd
import numpy as np


def createOneRandomsample(X, l=[], d=0.5):
    '''
    Create a random sample from two reference data points.

    Parameters
    ----------
    X : dataset.
    l : two ref data point indices. if None, will select two random indices.
    d : the max distance from the center of the two ref points. default is 0.5, which means the range of [x1, x2].
    '''

    m = len(X)
    
    if m < 2:
        return None

    if len(l) == 0:
        l = np.random.choice(m, 2, replace=False)

    # print(l[0], l[1])
    x1 = X[l[0], :]
    x2 = X[l[1], :]

    if np.allclose(x1, x2):
        return None

    k = random.random() * 2 * d - d  # random.random(), [0,1) -> [-1, 1)

    s = (x1 + x2)/2 + np.multiply(k, (x2 - x1))  # element-wise multiply

    return s


def expand_dataset(X, y, names, savepath, d = 0.5, NX = 3):
    '''
    Parameters
    ----------
    d : the max distance from the center of the two ref points. 
        default is 0.5, which means the range of [x1, x2].
        defines the variance / fluctuation of generated points.
    NX : how many times to expand the dataset.
    '''

    # use SMOTE to upsample
    labels = set(y)

    for label in labels:
        # the iteration size doesn't change once set
        for _ in range(len(y[y == label]) * NX):
            # y_grp = y[y == label]  # the dataset is updated in each iteration.
            X_grp = X[y == label]

            X_blend = X_grp
            # X_blend = np.vstack((X_grp, X_grp, X)) # blend 1/4 other class samples

            # print(l,k)
            s = createOneRandomsample(X_blend, d=d)

            if s is None:
                continue

            X = np.vstack((X, s))
            y = np.append(y, label)

    dfX = pd.DataFrame(X)
    dfX.columns = names

    dfY = pd.DataFrame(y)
    dfY.columns = ['label']

    df = pd.concat([dfY, dfX], axis=1)
    df = df.sort_values(by=['label'], ascending=True)
    df.to_csv(savepath, index=False)  # don't create the index column

    return X, y
