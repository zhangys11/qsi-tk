from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

def expand_dataset(X, y, nobs, n_gaussians=20):
    '''
    Generate equal number of samples for each class.

    nobs : samples to generate per class.
    '''
    final_data=pd.DataFrame()
    for label in set(y):
        data_sam=[]
        for i in range(X.shape[1]):
            gmm = GaussianMixture(n_components=n_gaussians)  # 设定混合成分数为2
            gmm.fit(np.array(X.iloc[:,i]).reshape(-1, 1))  # 对数据进行拟合
            y_sam=[]
            for m in range(nobs):
                y_data=0
                for i,j,k in zip(gmm.means_,gmm.covariances_,gmm.weights_):
                    x = np.random.normal(loc=i[0], scale=j[0][0], size=1)
                    x = x*k
                    y_data = y_data+x
                y_sam.append(float(y_data))
            data_sam.append(y_sam)
        GMM_sam=pd.DataFrame(data_sam).T
        GMM_sam.columns=X.columns
        GMM_sam[y.name]=[label]*len(GMM_sam)
        final_data=pd.concat([final_data,GMM_sam],axis=0)

    final_data.reset_index(drop=True, inplace=True)
    return final_data.iloc[:,:-1],final_data.iloc[:,-1]