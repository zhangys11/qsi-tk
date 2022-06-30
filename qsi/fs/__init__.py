from sklearn.feature_selection import chi2, f_classif
from ..vis import plotComponents2D, plot_feature_importance, unsupervised_dimension_reductions
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV, ElasticNetCV
from scipy.fftpack import fft, dct

def chisq_stats_fs(X, y, N = 30, display = True):
    '''
    chi-squared stats
    This score can be used to select the n_features features with the highest values for the test chi-squared statistic from X, which must contain only non-negative features such as booleans or frequencies (e.g., term counts in document classification), relative to the classes.
    Recall that the chi-square test measures dependence between stochastic variables, so using this function “weeds out” the features that are the most likely to be independent of class and therefore irrelevant for classification.
    '''

    c2,pval = chi2(X, y)
    return __fs__(X, c2, N, display)

def anova_stats_fs(X, y, N = 30, display = True):
    '''
    An Analysis of Variance Test or an ANOVA is a generalization of the t-tests to more than 2 groups. Our null hypothesis states that there are equal means in the populations from which the groups of data were sampled. More succinctly:
        μ1=μ2=...=μn
    for n groups of data. Our alternative hypothesis would be that any one of the equivalences in the above equation fail to be met.
    f_classif and chi-squared stats are both univariate feature selection methods
    '''

    F,pval = f_classif(X, y)
    return __fs__(X, F, N, display)

def __fs__(X, fi, N = 30, display = True):
    '''
    fi : feature importance
    N : how many features to be kept
    '''

    if display:
        plot_feature_importance(fi, 'feature-wise chi2 values', xtick_angle=0)

    idx = (np.argsort(fi)[-N:])[::-1]
    # idx = np.where(pval < 0.1)[0] # np.where(chi2 > 4.5)
    print('Important feature Number: ', len(idx))

    X_chi2 = X[:,idx]

    return X_chi2, idx
    # X_chi2_dr = unsupervised_dimension_reductions(X_chi2, y)

def lasso_fs(X, y, N = 30, display = True, verbose = True):

    lasso = LassoCV(cv = 5)
    lasso.fit(X,y)
    N = np.count_nonzero(lasso.coef_)

    if verbose:
        print('LASSO alpha = ', lasso.alpha_)
        print('Non-zero feature coefficients:',N)
   
    return __fs__(X, np.abs(lasso.coef_), N, display)
    
def elastic_net_fs (X, y, N = 30, display = True, verbose = True):
    '''
    Elastic Net vs LASSO
    --------------------
    Advantages:
        Elastic net is able to select groups of variables when they are highly correlated.
        It doesn't have the problem of selecting more than m predictors when n≫m. Whereas lasso saturates when n≫m
        When there are highly correlated predictors lasso tends to just pick one predictor out of the group.
        When m≫n and the predictors are correlated, the prediction performance of lasso is smaller than that of ridge.
    Disadvantages:
        One disadvantage is the computational cost. You need to cross-validate the relative weight of L1 vs. L2 penalty, α, and that increases the computational cost by the number of values in the α grid.
        Another disadvantage (but at the same time an advantage) is the flexibility of the estimator. With greater flexibility comes increased probability of overfitting.
    '''

    elastic_net = ElasticNetCV(cv = 5)
    elastic_net.fit(X,y)
    N = np.count_nonzero(elastic_net.coef_)

    if verbose:
        print('alpha = ', elastic_net.alpha_, ', L1 ratio = ', elastic_net.l1_ratio_ )
        print('Non-zero feature coefficients:',N)
   
    return __fs__(X, np.abs(elastic_net.coef_), N, display)
    

def nch_time_series_fs(X, fft_percentage = 0.05, dct_percentage = 0.1, conv_mask = [1,-2,1], display = True, y = None, labels = None):
    '''
    Multi-channel time series data feature selection. 
    Suitable for e-nose and e-tongue signals.

    与质谱、光谱不同，电子舌、电子鼻数据反应了各个传感器的时间响应特性。
    我们需要设计特征集合，以反映这种随时间变换的动态特点（时间响应特性）。

    提供的基础特征：

        AUC(Area Under CurveA)，积分/面积
        Max peak height (响应的最高峰值)
        一阶导数的AUC、max、min
        二阶导数的AUC、max、min
        变换域中的低频特征，如FFT、DCT。考虑前5%的低频组分。
        一维卷积核（sliding window, 1d conv kernel, e.g., Laplace mask)

    X : input data. Should have shape (m,ch,n)
    fft_percentage : default 0.05, means to keep the top 5% FFT components
    dct_percentage : default 0.1, means to keep the top 10% DCT components.
    conv_mask : convolution mask. default is 1D-laplacian mask [1,-2,1]

    y, labels : only used in the visualization part. If you don't need visualizaiton, just pass None or ignore.

    '''

    LV = [] # concated long vector

    FS1 = []
    FS2 = []
    FS3 = []
    FS4 = []

    for x in X:

        fs1 = []
        fs2 = []
        fs3 = []
        fs4 = []
        LV.append(x.flatten().tolist())

        for xx in x: 
            
            ch = xx # one sample's one channel   
            
            ###### Feature Set 1 #######
            
            fs1.append(ch.sum())
            fs1.append(ch.max())
            der = np.diff(ch)
            fs1.append(der.sum())
            fs1.append(der.max())
            fs1.append(der.min())
            der2 = np.diff(der)
            fs1.append(der2.sum())
            fs1.append(der2.max())
            fs1.append(der2.min())
            
            # der3 = np.diff(der2) # adding 3-ord derivative doesn't improve classifiablity
            # fs.append(der3.sum())
            # fs.append(der3.max())
            # fs.append(der3.min())
            
            
            ###### Feature Set 2 #######
            L = len(ch)

            fft_arr = fft(ch).real [:round(L * fft_percentage)] # tne first 5% （this is a hyper-parameter） low-freq components
            # plt.plot(fft_arr)
            # plt.plot(dct_arr)
            # plt.show()
            fs2 = fs2 + fft_arr.tolist()        
            
            ###### Feature Set 3 #######
            
            dct_arr = dct(ch) [:round(L * dct_percentage)]
            fs3 = fs3 + dct_arr.tolist()
            
            ###### Feature Set 4 #######
            
            conved = np.convolve(ch, conv_mask, 'valid')
            # plt.plot(laplace) # not sparse at all
            # plt.show()
            fs4 = fs4 + conved.tolist()
            
        FS1.append(fs1)
        FS2.append(fs2)
        FS3.append(fs3)
        FS4.append(fs4)
        
    LV = np.array(LV)

    FS_names = ['Concatenated Long Vector', 'Basic Descriptive Features', 
            'FFT top-n Low-Frequency Components', 
            'DCT top-n Low-Frequency Components',
           '1D Convolution Kernel']

    # return FS_names, [LV, FS1, FS2, FS3, FS4]

    if display:

        for name, FS in zip(FS_names, [LV, FS1, FS2, FS3, FS4]):
        
            ################ Feature Scaling ###############
            
            scaler = StandardScaler()
            scaler.fit(FS)
            FS = scaler.transform(FS)
            
            if y is None:

                ################ PCA ####################
                
                pca = PCA(n_components=2, scale = False)
                F_2d = pls.fit_transform(FS)

                plotComponents2D(F_2d, legends = labels)            
                plt.title(name + ' - PCA')

            else:

                ################ PLS ####################
                
                pls = PLSRegression(n_components=2, scale = False)
                F_2d = pls.fit(FS, y).transform(FS)

                # plt.figure(figsize = (20,15))
                plotComponents2D(F_2d, y, legends = labels)            
                title = name + ' - PLS'

                # Returns the coefficient of determination R^2 of the prediction.
                title = title + '\nR2 = ' + str( np.round(pls.score(FS, y),3) )

                plt.title(title)

            plt.show()

        if y is not None:
            print('About the PLS R2 core: \nThe score is the coefficient of determination of the prediction, defined as 1 - u/v, where u is the residual sum of squares ((y_true - y_pred)** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a score of 0.0.')
        


    
