import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def expand_dataset(X,y,nobs,bandwidth=100):
    '''
    Gaussian-KDE data aug

    '''
    final_data=pd.DataFrame()
    for label in set(y):
        data_sam=[]
        for i in range(X.shape[1]):
            x = np.linspace(-5000, 5000, 20000)
            data=X.iloc[:,i]

            model = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            model.fit(np.array(data).reshape(-1, 1))

            log_dens = model.score_samples(np.array(x).reshape(-1, 1))
            density_estimation = [dens for dens in np.exp(log_dens)]
            p = density_estimation
            p = p / np.sum(p)
            y_sam=[]
            for m in range(nobs):
                random_var = np.random.choice(x, size=1, p=p)
                y_sam.append(float(random_var))
            data_sam.append(y_sam)
        KDE_sam=pd.DataFrame(data_sam).T
        KDE_sam.columns=X.columns
        KDE_sam[y.name]=[label]*len(KDE_sam)
        final_data=pd.concat([final_data,KDE_sam],axis=0)

    final_data.reset_index(drop=True, inplace=True)
    return final_data.iloc[:,:-1],final_data.iloc[:,-1]


def expand_dataset_cv(X, y, NX = 3):
    '''
    KDE with GridSearchCV

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