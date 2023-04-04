import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def expand_dataset(X, y, NX = 3):
    '''
    Parameters
    ----------
    NX : how many times to expand the dataset.
    '''

    # use SMOTE to upsample
    labels = set(y)

    for label in labels:
        X_grp = X[y == label]
        params = {'kernel':['gaussian', 'tophat'], 'bandwidth': np.logspace(-1, 5, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(X_grp)
        print("best KDE kernel: {0}, bandwidth: {1}".format(grid.best_estimator_.kernel, grid.best_estimator_.bandwidth))
        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_
        new_data = kde.sample(len(y[y == label]) * NX)            

        X = np.vstack((X, new_data))
        y = np.append(y, [label] * len(y[y == label]) * NX)

    return X, y