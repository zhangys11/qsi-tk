import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV, ElasticNetCV, MultiTaskLassoCV, MultiTaskElasticNetCV
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from scipy.fftpack import fft, dct
from sklearn.linear_model import LogisticRegressionCV

from ..vis import plotComponents2D, plot_feature_importance, unsupervised_dimension_reductions
from .alasso import *
from .glasso import *
from .aenet import *
from .fsse import *

from IPython.display import HTML, display

def __fs__(X, fi, X_names = None, N = 30, display = True):
    '''
    Feature selection based on some feature importance metric.

    Parameters
    ----------
    fi : feature importance
    N : how many features to be kept

    Returns
    -------
    X_s : top-N selected features
    idx : top-N selected feature indices
    fi : top-N feature importances (e.g., coef abs values)

    Notes
    -----
    Some FS algorithms will output the R2 metric.
    You should remember R2 is not a good measure to assess goodness of fit 
    for a classification task. 
    R2 is suitable for predicting continuous variable (regression). 
    When dependent variable is continuous R2 usually takes values between 0 and 1 
    (in linear regression for example it is impossible to have R2 beyond these boundaries),
    and it is interpreted as share of variance of dependent variable that 
    model is able to correctly reproduce. 
    When R2 equals 1 it means that the model is able to fully recreate 
    dependent variable, when it equals 0, it means that the model completely 
    failed at this task.
    When the dependent variable is categorical it makes no sense, 
    because R2 uses distances between predicted and actual values, 
    while distances between let say '1' meaning class 'A', '2' meaning class 'B' 
    and '3' meaning class 'C' make no sense.
    '''

    if display:
        plot_feature_importance(fi, X_names, 'feature-wise coefs/values', xtick_angle=0)

    idx = (np.argsort(fi)[-N:])[::-1]
    # idx = np.where(pval < 0.1)[0] # np.where(chi2 > 4.5)

    X_s = X[:,idx]
    print('Important feature Number: ', len(idx))
    print('Important features indices: ' , idx) 
    if X_names is not None:
        print('Important features names: ', X_names[idx])
    print('Top-'+str(len(idx))+' feature Importance: ', fi[idx])

    return X_s, idx, fi[idx]
    # X_s_dr = unsupervised_dimension_reductions(X_s, y)

def pearson_r_fs(X, y, X_names=None, N = 30, display = True):
    '''
    The pearson r doesn't have a strong feature selection effect (not sparse).
    We seldom use this method for fs. This is just for theoretical analysis.
    '''
    CM = np.corrcoef(np.hstack((X,np.array(y).reshape(-1,1))), rowvar=False) 
    
    if display:
        plt.figure(figsize=(20,20))
        plt.matshow(CM)
        plt.title('Correlaton Coef Matrix between all the X and y.\ny is at the last row/col.')
        plt.axis('off')
        plt.show()

    rs = np.abs(CM[-1,:-1]) # this the corrcoef abs between y and all xs
    return __fs__(X, rs, X_names, N, display)

def mi_fs(X, y, X_names=None, N = 30, display = True):
    '''
    info-gain / Mutual Information

    Information gain has been used in decision tree. For a specific feature, Information gain (IG) measures how much “information” a feature gives us about the class.
    𝐼𝐺(𝑌|𝑋)=𝐻(𝑌)−𝐻(𝑌|𝑋)
    IG/MI returns zero for independent variables and higher values the more dependence there is between the variables (can be used to rank features by their independence).
    In information theory, IG answers "if we transmit Y, how many bits can be saved if both sender and receiver know X?" Or "how much information of Y is implied in X?"
    Attribute/feature X with a high IG is a good split on Y.
    Pearson r only captures linear correlations, while information gain also captures non-linear correlations.
    '''
    # use mutual_info_regression if y is continous.
    mi = mutual_info_classif(X, y, discrete_features=False)
    return __fs__(X, mi, X_names, N, display)

def chisq_stats_fs(X, y, X_names=None, N = 30, display = True):
    '''
    chi-squared stats
    This score can be used to select the n_features features with the highest values for the test chi-squared statistic from X, which must contain only non-negative features such as booleans or frequencies (e.g., term counts in document classification), relative to the classes.
    Recall that the chi-square test measures dependence between stochastic variables, so using this function “weeds out” the features that are the most likely to be independent of class and therefore irrelevant for classification.
    '''

    c2,pval = chi2(X, y)
    return __fs__(X, c2, X_names, N, display)

def anova_stats_fs(X, y, X_names=None, N = 30, display = True):
    '''
    An Analysis of Variance Test or an ANOVA is a generalization of the t-tests to more than 2 groups. Our null hypothesis states that there are equal means in the populations from which the groups of data were sampled. More succinctly:
        μ1=μ2=...=μn
    for n groups of data. Our alternative hypothesis would be that any one of the equivalences in the above equation fail to be met.
    f_classif and chi-squared stats are both univariate feature selection methods
    '''

    F,pval = f_classif(X, y)
    return __fs__(X, F, X_names, N, display)

def lasso_fs(X, y, X_names=None,  N = 30, display = True, verbose = True):

    lasso = LassoCV(cv = 5)
    lasso.fit(X,y)
    N = min( np.count_nonzero(lasso.coef_), N)

    if verbose:
        print('R2 = ', round(lasso.score(X,y), 3) )
        print("LASSO alpha = %.3g" %lasso.alpha_)
        print('Non-zero feature coefficients:', np.count_nonzero(lasso.coef_))
   
    return __fs__(X, np.abs(lasso.coef_), X_names, N, display)
    
def elastic_net_fs (X, y, X_names=None, N = 30, display = True, verbose = True):
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
    NZ = np.count_nonzero(elastic_net.coef_)

    if verbose:
        print('R2 = ', round(elastic_net.score(X,y),3) )
        print('alpha =  %.3g' % elastic_net.alpha_ , ', L1 ratio = ', elastic_net.l1_ratio_ )
        print('Non-zero feature coefficients:',NZ)
    
    if N is None or N <=0 or N > NZ:
        N = NZ
   
    return __fs__(X, np.abs(elastic_net.coef_), X_names, N, display)
    
def alasso_fs(X, y, X_names=None, N = 30, LAMBDA = 0.1, flavor = 2, \
    display = True, verbose = True):
    '''
    Adaptive lasso

    Parameters
    ----------
    LAMBDA : controls regularization / sparsity. The effect may vary for different flavors.
    flavor : we provide 3 implementations. 
        1 - alasso_v1
        2 - alasso_v2
        3 - alasso_v3
    '''
    if flavor == 1:
        coef_ = alasso_v1(X, y, W = None, LAMBDA = LAMBDA)
        if len(coef_) != X.shape[1]:
            print('Error: returned coef dim differs from X. \nTry flavor 3.')
            coef_, R2 = alasso_v3(X, y, LAMBDA=LAMBDA)
    elif flavor == 2:
        _, coef_, R2, _, _ = alasso_v2(X, y, LAMBDAS = [LAMBDA], display=display)
    else: # flavor 3
        coef_, R2 = alasso_v3(X, y, LAMBDA=LAMBDA)

    eps = 0 # 1e-9 # epsilon - non-zero coef cut threshold
    NZ = np.count_nonzero(coef_ > eps)
    
    if verbose:
        print('R2 = ', round(R2,3) ) # R2 - the coefficient of determination 
        # print(coef_)
        print('Non-zero feature coefficients (eps = %2g'%eps + '):', NZ)
    
    if N is None or N <=0 or N > NZ:
        N = NZ
   
    return __fs__(X, np.abs(coef_), X_names, N, display)

def glasso_fs(X, y, X_names=None, N = 30, WIDTH = 8, ALPHA = 0.5, LAMBDA = 0.1, display = True, verbose = True):
    '''
    Group lasso

    Parameters
    ----------
    WIDTH : window size
    ALPHA : adjust 
    LAMBDA : controls regularization / sparsity. The effect may vary for different flavors.
    '''

    THETAS = group_lasso(X, y, WIDTH=WIDTH, LAMBDA=LAMBDA, ALPHA=ALPHA)   
    
    NZ = np.count_nonzero(THETAS)
    if verbose:
        print('Non-zero feature coefficients:',NZ)
    
    if N is None or N <=0 or N > NZ:
        N = NZ

    return __fs__(X, np.abs(THETAS), X_names, N, display)

def glasso_cv_fs(X, y, X_names=None, N=30, N2=None, \
    WIDTHS=[2,8,32], LAMBDAS=[0.01,0.1,1], ALPHAS=[0,0.5,1], \
    display = True):
    '''
    Parameters
    ----------
    N1 : tier-1 (glasso fs) features to be kept. 
    N2 : tier-2 (select common features from multiple runs) features to be kept.
    '''
    HPARAMS, FSIS, THETAS, SCORES = group_lasso_cv(X, y, MAXF=N, \
        WIDTHS=WIDTHS, LAMBDAS=LAMBDAS, ALPHAS=ALPHAS, cv_size = 0.2)
    if N2 is None:
        N2 = N
    COMMON_FSI = select_features_from_group_lasso_cv(HPARAMS, FSIS, THETAS, SCORES, MAXF = N2, THRESH = 1.0)
    if COMMON_FSI is None or len(COMMON_FSI) <= 0: # no common important features are selected
        return None, None, None 
    return X[:,COMMON_FSI], COMMON_FSI, None

def aenet_cv_fs(X, y, X_names=None, N = 30, display = True, verbose = True):
    if (X.shape[1] > 5000):
        print('Be patient. Your data is high-dimensional. It will take long time.')
    aen = AdaptiveElasticNetCV().fit(X, y)
    if verbose:
        print('R2 =', round(aen.score(X, y),3) )
        # print(aen.__dict__)

    NZ = np.count_nonzero(aen.coef_)

    if verbose:
        print('Non-zero feature coefficients:', NZ)
    
    if N is None or N <=0 or N > NZ:
        N = NZ

    return __fs__(X, np.abs(aen.coef_), X_names, N, display)

def multitask_lasso_fs(X, y, X_names=None, N = 30, display = True, verbose = True):
    '''
    Parameters
    ----------
    y : should have shape [m, 2/3/4...]. Each col for one class labels. e.g., clf.fit([[0, 1], [1, 2], [2, 4]], [[0, 0], [1, 1], [2, 3]])
    '''
    if len(y.shape) != 2:
        print('Error: y must have shape (m,K), each col for one class.')
        print('Because y is single task, use lasso fs.')
        return lasso_fs(X, y, X_names, N, display, verbose)

    clf = MultiTaskLassoCV()
    clf.fit(X, y)
    
    NZ = np.count_nonzero(clf.coef_)
    if verbose:
        print('Non-zero feature coefficients:',NZ)
    
    if N is None or N <=0 or N > NZ:
        N = NZ

    return __fs__(X, np.abs(clf.coef_), X_names, N, display)

def multitask_elastic_net_fs(X, y, X_names=None, N = 30, display = True, verbose = True):
    '''
    Parameters
    ----------
    y : should have shape [m, 2/3/4...]. Each col for one class labels. e.g., clf.fit([[0, 1], [1, 2], [2, 4]], [[0, 0], [1, 1], [2, 3]])
    '''
    if len(y.shape) != 2:
        print('Error: y must have shape (m,K), each col for one class.')
        print('Because y is single task, use elastic net fs.')
        return elastic_net_fs(X, y, X_names, N, display, verbose)

    clf = MultiTaskElasticNetCV()
    clf.fit(X, y)
    
    NZ = np.count_nonzero(clf.coef_)
    if verbose:
        print('Non-zero feature coefficients:',NZ)
    
    if N is None or N <=0 or N > NZ:
        N = NZ

    return __fs__(X, np.abs(clf.coef_), X_names, N, display)

def fsse_fs(X, y, X_names=None, N = 30, base_learner=ensemble.create_elmcv_instance, \
    WIDTHS = [1, 2, 10, 30], ALPHAS = [0.5,0.75,1.0], display = True, verbose = True):
    
    idx = fsse_cv(X,y,X_names,N, base_learner=base_learner, \
    WIDTHS = WIDTHS, ALPHAS = ALPHAS, display=display,verbose=verbose)

    if verbose and len(WIDTHS) > 1:

        print('')
        print('-------- Combine the FS results from multiple runs. --------')

        print('Most important common feature indices from fsse_cv( WIDTHS = ', WIDTHS , '): ', idx)
        if X_names is not None and X_names != []:
            print('Most important common feature names from fsse_cv( WIDTHS = ', WIDTHS , '): ', np.array(X_names)[idx])

    return X[:,idx], idx, None

FS_DICT={
    "pearsion-r": pearson_r_fs,
    "info-gain / mutual information": mi_fs,
    "chi-squared statistic": chisq_stats_fs,
    "anova statistic": anova_stats_fs,
    "lasso": lasso_fs,
    "elastic net": elastic_net_fs,
    "adaptive lasso": alasso_fs,
    "group lasso": glasso_cv_fs,
    # "group lasso cv": glasso_cv_fs,
    "adaptive elastic net": aenet_cv_fs,
    "multi-task lasso": multitask_lasso_fs,
    "multi-task elastic net": multitask_elastic_net_fs,
}

FS_DESC_DICT={
    "pearson-r": '''
    Two reasons why to prefer Pearson correlation when the relationship is close to linear.

    Pearson r correlation is more efficient and faster.
    The range of the correlation coefficient is [-1, 1], which reveal positive / negative correlations.
    ''',    

    "info-gain / mutual information": '''
Information gain has been used in decision tree. For a specific feature, Information gain (IG) measures how much “information” a feature gives us about the class.

𝐼𝐺(𝑌|𝑋)=𝐻(𝑌)−𝐻(𝑌|𝑋)

IG/MI returns zero for independent variables and higher values the more dependence there is between the variables (can be used to rank features by their independence).
In information theory, IG answers "if we transmit Y, how many bits can be saved if both sender and receiver know X?" Or "how much information of Y is implied in X?"

Attribute/feature X with a high IG is a good split on Y.

Pearson r only captures linear correlations, while information gain also captures non-linear correlations.
    ''',

    "chi-squared statistic": '''
    This score can be used to select the n_features features with the highest values for the test chi-squared statistic from X, which must contain only non-negative features such as booleans or frequencies (e.g., term counts in document classification), relative to the classes.

Recall that the chi-square test measures dependence between stochastic variables, so using this function “weeds out” the features that are the most likely to be independent of class and therefore irrelevant for classification.
    
    ''',

    "anova statistic": "ANOVA F-value",

    "elastic net": '''
    LASSO系列基于L1-norm, 特征选择的稀疏性较强烈。ElasticNet(L1+L2) 更为"温和"。
    Elastic net is "a doubly regularized technique which encourages grouping effect i.e. either selection or omission of the correlated variable together and is particularly useful when the number of covariates (p) is much larger than the number of observations (n). "
    '''
    ,
    "adaptive lasso": '''
    Reference: https://ricardocarvalho.ca/post/lasso/

An oracle estimator selects the truly significant variables with probability tending to one. Asymptotically （渐进、逼近）, both subsets coincide.

Idea: add some weights w that corrects the bias in lasso, to convert your regression estimator into an oracle, something that knows the truth about your dataset.
    
    Implementation (flavor 2): Get the initial W (coef) via ridge regression, then solve the weighted lasso. 
    
    MSE 随 𝜆 增加而变大, N 变小（特征选择效应、稀疏性加强）。You may try "qsi.fs.alasso.alasso_v2(X_scaled, y, LAMBDAS = np.logspace(-10, 0, 11))" to decide the best lambda.
    ''',

    "group lasso": '''
    The group lasso regulariser is a well known method to achieve structured sparsity in machine learning and statistics.

An extension of the group lasso regulariser is the sparse group lasso regulariser [2], which imposes both group-wise sparsity and coefficient-wise sparsity. This is done by combining the group lasso penalty with the traditional lasso penalty.

Reference: /py/machine learning/source/19. Kernel/Group Lasso.ipynb

This method implements the sparse group lasso, which is a linear combination between lasso and group lasso, so it provides solutions that are both between and within group sparse. 
    
    ''',
    "adaptive elastic net": '''
    " The adaptive elastic-net combines the strengths of the quadratic regularization and the adaptively weighted lasso shrinkage. Under weak regularity conditions, we establish the oracle property of the adaptive elastic-net. We show by simulations that the adaptive elastic-net deals with the collinearity problem better than the other oracle-like methods, thus enjoying much improved finite sample performance."
    ''',
    "multi-task lasso": '''
    The MultiTaskLasso is a linear model that estimates sparse coefficients for multiple regression problems jointly: y is a 2D array, of shape (n_samples, n_tasks). The constraint is that the selected features are the same for all the regression problems, also called tasks.

    Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.

    The optimization objective for Lasso is:''' + r"$ \
    (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21 \
    Where: \
    ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2} $",

    "multi-task elastic net": 'Multi-task elastic net model trained with L1/L2 mixed-norm as regularizer.'
}

def RUN_ALL_FS(X, y, X_names, labels=None, N = 30, output = None, multitask = False):
    '''
    Iterate all FS methods and apply to the target dataset.

    Parameters
    ----------
    output : specify which FS result to output. Can be an FS name, 'all' or 'none'/None.
    multitask : whether run multi-task FS methods. Default is False, i.e., only run single-task FS methods.

    Return
    ------
    FS_OUTPUT : a dict of specified FS results.
    '''

    X_names = np.array(X_names)
    FS_OUTPUT = {}

    for key in FS_DICT:

        if multitask and 'multi-task' in key:
            pass
        elif not multitask and 'multi-task' not in key:
            pass
        else:
            continue # otherwise, skip this alg

        if key == 'pearsion-r':
            display(HTML('<h2>Univariate feature selection</h2><br/><p>Univariate feature selection examines each feature individually to determine the strength of the relationship of the feature with the response variable.</p><br/>'))
        elif key == 'lasso':
            display(HTML('<h4>Generally speaking, univariate methods dont generate sparsity and have weak FS effect. </h4><hr/>'))
            display(HTML('<h2>Model based ranking</h2><br/><p>Use a machine learning method to build a discriminative model for the response variable using each individual feature, and measure the performance of each model.</p><br/>'))

        display(HTML('<h3>' + key + '</h3><br/>'))
        if key in FS_DESC_DICT:
            display(HTML('<p>' + FS_DESC_DICT[key] + '</p><br/>'))

        f = FS_DICT[key]
        X_s, idx, fi = f(X, y, X_names, N = N, display = True)

        ax = plotComponents2D(X_s[:,:2], y)
        ax.set_title('Scatter Plot of Top-2 Selected Features')
        plt.show()

        X_s_dr = unsupervised_dimension_reductions(X_s, y, legends = labels)

        clf = LogisticRegressionCV().fit(X_s, y)
        print('Classification accurary with the selected features (LogisticRegressionCV) = ', \
            round(clf.score(X_s, y), 3))

        if output == key:
            FS_OUTPUT[key] = X_s
        elif output == 'all':
            FS_OUTPUT[key] = X_s
        else:
            pass

        display(HTML('<hr/>'))

    return FS_OUTPUT

########### e-nose / e-tongue functions ###############

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
        


    
