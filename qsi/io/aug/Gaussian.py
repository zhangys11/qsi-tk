import numpy as np
import pandas as pd
import scipy.stats

def expand_dataset(X,y,nobs):
    final_data=pd.DataFrame()
    for label in set(y):
        data_sam=[]
        for i in range(X.shape[1]):
            x = np.linspace(-5000, 5000, 20000)
            data=X.iloc[:,i]
            mean,std=scipy.stats.norm.fit(data)
            y_sam=[]
            for m in range(nobs):
                random_var=np.random.normal(mean, std, 1)
                y_sam.append(float(random_var))
            data_sam.append(y_sam)
        Gaussian_sam=pd.DataFrame(data_sam).T
        Gaussian_sam.columns=X.columns
        Gaussian_sam[y.name]=[label]*len(Gaussian_sam)
        final_data=pd.concat([final_data,Gaussian_sam],axis=0)
    final_data.reset_index(drop=True, inplace=True)

    return final_data.iloc[:,:-1],final_data.iloc[:,-1]