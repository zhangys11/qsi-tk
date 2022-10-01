import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import cvxpy as cp

def raman_prior():
    '''
    Return Raman prior knowledge, i.e., what wavenumber ranges correspond to what functional groups (chemical bonds).
    '''

    d = {}

    d['alkane_1'] = list(range(1295, 1305 + 1))
    d['alkane_2'] = list(range(800, 900 + 1)) + list(range(1040, 1100 + 1))
    d['branched_alkane_1'] = list(range(900, 950 + 1))
    d['branched_alkane_2'] = list(range(1040, 1060 + 1))
    d['branched_alkane_3'] = list(range(1140, 1170 + 1))
    d['branched_alkane_4'] = list(range(1165, 1175 + 1))
    d['haloalkane_1'] =  list(range(605, 615 + 1))
    d['haloalkane_2'] =  list(range(630, 635 + 1))
    d['haloalkane_3'] =  list(range(655, 675 + 1))
    d['haloalkane_4'] =  list(range(740, 760 + 1))
    d['alkene'] = list(range(1638, 1650 + 1))
    d['alkyne'] = list(range(2230, 2237 + 1))
    d['toluence'] = list(range(990, 1010 + 1))
    d['alcohol'] = list(range(800, 900 + 1))
    d['aldehyde'] = list(range(1725, 1740 + 1))
    d['ketone'] = list(range(1712, 1720 + 1))
    d['ether'] = list(range(820, 890 + 1))
    d['carboxylic_acid'] = list(range(820, 890 + 1))
    d['ester'] = list(range(634, 644 + 1))
    d['amine_1'] = list(range(740, 833 + 1))
    d['amine_2'] = list(range(1000, 1250 + 1))
    d['amide'] = list(range(700, 750 + 1))
    d['nitrile'] = list(range(2230, 2250 + 1))

    return d

def plot_raman_prior():

    d = raman_prior()

    plt.figure(figsize = (14,7))

    for idx, key in enumerate(d):
        # print(d[key])
        plt.scatter(d[key], [-idx] * len(d[key]), lw = 5, label = key)
        
    plt.legend(loc = "upper right")
    plt.yticks([])
    plt.xticks(range(500, 3001, 500))
    plt.show()

def binning_op(xaxis, region, filter = 'rbf', SD = 1):
    '''
    xaxis : the entire x axis range, e.g., [0, 3000]
    region : e.g., [100,200]
    filter : can be 'rbf', 'sinc', 'logistic', 'uniform'. Uniform is just averaging filter.
    SD : for rbf kernel, the region will lie inside +/-SD
    
    Return : op array. Has the length of xaxis.
    '''
    if filter == 'uniform':
        op = np.ones(len(xaxis)) / len(xaxis)
    # todo: others
    return op


def adaptive_binning(X, regions, filter = 'rbf'):
    '''
    Convert one data to binned features.
    Break down the axis as sections. Each seection is an integral of the signal intensities in the region.
    Integration can be done by radius basis function / sinc kernel, etc.

    filter : Apply a filter operator to a continuous region. Can be 'rbf', 'sinc', 'logistic', 'uniform'. Uniform is just averaging filter.
    '''

    Fss = []
    for x in X:

        Fs = [] # the discrete features for one data sample
        for region in regions:
            op = binning_op([0, len(x)], region, filter)
            F = (op*x).sum()
            Fs.append(F)

        Fss.append(Fs)

    return np.array(Fss)


def group_lasso(X_scaled, y, WIDTH, offset = 0, LAMBDA = 1, ALPHA = 0.5):
    """
    Group Lasso Feature Selection

    Parameters
    ----------
    X_scaled : X, should be rescaled;
    y : target var;
    WIDTH : sliding window's width; 
    LAMBDA : regularization coefficient; 
    ALPHA : ratio of L1 vs Group;
    """

    assert(offset < WIDTH)

    # Problem data.
    m,n = X_scaled.shape
    X_scaled_e =  np.hstack((np.ones( (len(X_scaled),1 ) ) , X_scaled )) 

    # Construct the problem.
    theta = cp.Variable(n+1)

    group_loss = cp.norm(theta[1:][:offset]) # cp.norm(np.zeros(WIDTH))
    for i in range(offset, n, WIDTH):
        # +1 for skipping bias
        group_loss = group_loss + cp.norm(theta[1:][i:i+WIDTH]) # the features are already scaled. No need for group-sepecific weights

    group_loss = group_loss + cp.norm(theta[1:][i+WIDTH:])

    objective = cp.Minimize(cp.sum_squares(X_scaled_e @ theta - y) / 2 
                            + ALPHA * LAMBDA * cp.norm(theta[1:], 1) 
                            + (1-ALPHA)*LAMBDA * group_loss
                           )
    constraints = []
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    
    THETA = theta.value[1:] # skip the bias/intercept  
    # plot_feature_importance(np.abs(THETA), 'All feature coefficiences')
    
    return THETA #, biggest_gl_fs, X_gl_fs

def group_lasso_cv(X_scaled, y, MAXF, WIDTHS, LAMBDAS, ALPHAS, cv_size = 0.2, verbose = False):
    '''
    Optimize hyper-parameters by grid search.

    Parameters
    ----------
    MAXF : max features to be selected. We compare each iteration's ACC with the same number of features. 
    WIDTHS : a list of window width / group size. 
    LAMBDAS : a list of lambdas (regularization).
    ALPHAS : a list of alphas.
    cv_size : cross validation set size. Default 20%.
    '''

    SCORES = []
    HPARAMS = [] # hyper-parameter values
    
    FSIS=[]
    THETAS = []

    pbar = tqdm(total=len(WIDTHS)*len(ALPHAS)*len(LAMBDAS)) # np.sum(WIDTHS)
    
    for w in WIDTHS:
        for offset in [int(w/2)]: # range(w)
            for alpha in ALPHAS:
                for lam in LAMBDAS:
                    
                    train_X,test_X, train_y, test_y = train_test_split(X_scaled, y,
                                                   test_size = cv_size)
                    
                    hparam = 'Window Size: ' + str(w) + ', offset = ' + str(offset) + ', alpha = ' + str(alpha) + ', lambda = ' + str(lam) 
                    HPARAMS.append(hparam)

                    if verbose:
                        print('=== ' + hparam + ' ===')
                    
                    THETA = group_lasso(train_X, train_y, 
                                    w, offset,
                                    LAMBDA = lam, ALPHA = alpha)
                    
                    biggest_gl_fs = (np.argsort(np.abs(THETA))[-MAXF:])[::-1]
                    # biggest_gl_fs = X_scaled[:,MAXF]

                    FSIS.append(list(biggest_gl_fs))
                    
                    if verbose:
                        print('Selected Feature Indices: ', biggest_gl_fs)

                    THETAS.append(THETA)
                                       
                    # No selected features
                    if (len(biggest_gl_fs) <= 0):
                        SCORES.append(0)
                    else:
                        reg = LinearRegression().fit(test_X[:,biggest_gl_fs], test_y)
                        score = reg.score(test_X[:,biggest_gl_fs], test_y)
                        SCORES.append(score)

                    pbar.update(1)
                    
    pbar.close()

    assert (len(set([len(HPARAMS), len(FSIS), len(THETAS), len(SCORES)])) == 1)
    return HPARAMS, FSIS, THETAS, SCORES

def select_features_from_group_lasso_cv(HPARAMS, FSIS, THETAS, SCORES, MAXF = 50, THRESH = 1.0):
    '''
    This is a further processing that selects MAXF most common important features.

    Parameters
    ----------
    HPARAMS, FSIS, THETAS, SCORES : returned by group_lasso_cv() 
    THRESH : coef_ abs minimum threshold 
    '''

    CAT_FS = []
    IDX = []
    FS_HPARAMS = []

    plt.figure(figsize = (16, math.ceil(MAXF/2)))

    idxx = 0
    for idx, score in enumerate(SCORES):
        # only keep whose score >= THRESH
        if (score >= THRESH):
            IDX.append(idx)
            CAT_FS += FSIS[idx]
            FS_HPARAMS.append(HPARAMS[idx])
            plt.plot(THETAS[idx] + idxx*0.1, label = str(HPARAMS[idx]))
            idxx += 1

    print('top-' + str(MAXF) + ' features and their frequencies: ', Counter(CAT_FS).most_common(MAXF))

    plt.yticks([])
    if (idxx <= 10):
        plt.legend()
    plt.show()

    COMMON_FSI = []
    for f in Counter(CAT_FS).most_common(MAXF):
        COMMON_FSI.append(f[0])
               
    return np.array(COMMON_FSI)