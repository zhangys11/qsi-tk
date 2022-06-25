from sklearn.feature_selection import chi2, f_classif
from ..vis import plot_feature_importance, unsupervised_dimension_reductions
import numpy as np
from sklearn.linear_model import LassoCV, ElasticNetCV

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
    
        

    
