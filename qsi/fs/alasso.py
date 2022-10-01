import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, Lasso, LassoCV
import matplotlib.pyplot as plt
# from unsupervised_dimension_reductions import *

def alasso_v1(X_scaled, y, W = None, LAMBDA = 1):
    """
    Adaptive Lasso Feature Selection: standard LASSO + ALASSO

    Parameters
    ----------
    X_scaled : X, should be rescaled;
    y : target var;
    W : weights, which can be returned by a normal LASSO;
    width : sliding window's width; 
    LAMBDA : regularization coefficient;
    """

    if W == None:
        ### do normal LASSO to get W ### 

        lasso = LassoCV(cv = 5, max_iter = 5000)
        lasso.fit(X_scaled,y)

        N = np.count_nonzero(lasso.coef_)
        biggest_lasso_fs = np.argsort(np.abs(lasso.coef_))[::-1][:N] # take last N item indices and reverse (ord desc)
        X_lasso_fs = X_scaled[:,biggest_lasso_fs[0:N]]

        W = 1/np.abs(lasso.coef_)[biggest_lasso_fs]

        X_scaled = X_lasso_fs # use the X after FS

    # Problem data.
    m,n = X_scaled.shape
    X_scaled_e =  np.hstack((np.ones( (len(X_scaled),1 ) ) , X_scaled )) 

    # Construct the problem.
    theta = cp.Variable(n + 1)
    objective = cp.Minimize(cp.sum_squares(X_scaled_e @ theta - y) / 2 + LAMBDA * W.T @ cp.abs(theta[1:]))
    constraints = []
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    
    THETA = theta.value[1:] # skip the bias/intercept  
    # plot_feature_importance(np.abs(THETA), 'All feature coefficiences')

    return THETA #, biggest_al_fs, X_al_fs

def alasso_v2(X_scaled, y, LAMBDAS = np.logspace(-10, 0, 11), gamma = 1, display = True, verbose = False):
    '''
    Adaptive Lasso Feature Selection: standard Ridge regression + ALASSO

    Parameters
    ----------
    gamma : 0.5, 1, or 2

    Return
    ------ 
    lambd_values, theta_values, train_errors, test_errors : tried lambda values and corresponding theta, train and test errors.
    '''

    def loss_fn(X, Y, theta):
        return cp.norm2(X @ theta - Y)**2

    def regularizer(w, theta):
        # print(w.shape, beta.shape)
        # print(cp.multiply(w, beta))
        return cp.norm1( cp.multiply(w, theta) ) # do an element-wise multiplication. w is guaranteed non-negative

    def objective_fn(X, Y, theta, w, lambd):
        return loss_fn(X, Y, theta) + lambd * regularizer(w, theta)

    def mse(X, Y, theta):
        return (1.0 / X.shape[0]) * loss_fn(X, Y, theta).value

    ridge = RidgeCV(cv = 5) # use ridge to get initial coef
    ridge.fit(X_scaled, y)
    print('alpha = ', ridge.alpha_)
    print('coef = ', ridge.coef_)

    w = np.array(ridge.coef_)
    w = 1 / ( np.absolute(w)** gamma )
    # print('w = ', w)    
    
    m,n = X_scaled.shape
    theta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)
    # w = cp.Constant(w)
    problem = cp.Problem(cp.Minimize(objective_fn(X_scaled, y.flatten(), theta, w, lambd)))
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.33)

    lambd_values = LAMBDAS # np.logspace(-10, 0, 11)
    train_errors = []
    test_errors = []
    theta_values = []
    for v in tqdm(lambd_values):
        lambd.value = v
        try:
            problem.solve(verbose = verbose) # in case of SolverError: Solver 'ECOS' failed.
        except:
            problem.solve(solver='SCS')
    
        train_errors.append(mse(X_train, Y_train, theta))
        test_errors.append(mse(X_test, Y_test, theta))
        theta_values.append(theta.value)

    if display and len(lambd_values) > 1:
        
        # plot training curve
        plt.plot(lambd_values, train_errors, label="Train error")
        plt.plot(lambd_values, test_errors, label="Test error")
        plt.xscale("log")
        plt.legend(loc="upper left")
        plt.xlabel(r"$\lambda$", fontsize=16)
        plt.title("Mean Squared Error (MSE)")
        plt.show()

        # plot non-zero coef curve
        eps = 1e-9 # epsilon - cut threshold
        nz = []

        for coef in theta_values:
            N = np.count_nonzero(coef > eps)
            nz.append(N)

        plt.plot(lambd_values[1:], nz[1:])
        plt.xscale("log")
        plt.xlabel(r"$\lambda$", fontsize=16)
        plt.title("Non-zero Coefficients")
        plt.show()

    return lambd_values, theta_values, train_errors, test_errors

def alasso_v3(X_train, y_train, X_names = None, n_lasso_iterations = 10, LAMBDA = 0.1, tol = 0.001):
    '''
    Yet another alasso implementation.
    '''
    # set checks
    higher  = float('inf')
    lower   = 0
    
    # set lists
    coefficients_list = []
    iterations_list   = []

    g = lambda w: np.sqrt(np.abs(w))
    gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)

    n_samples, n_features = X_train.shape
    p_obj = lambda w: 1. / (2 * n_samples) * np.sum((y_train - np.dot(X_train, w)) ** 2) \
                      + LAMBDA * np.sum(g(w))

    weights = np.ones(n_features)

    X_w = X_train / weights[np.newaxis, :]
    X_w  = np.nan_to_num(X_w)
    X_w  = np.round(X_w,decimals = 3)

    y_train    = np.nan_to_num(y_train)

    adaptive_lasso = Lasso(alpha = LAMBDA, fit_intercept=False)
    adaptive_lasso.fit(X_w, y_train)

    for k in range(n_lasso_iterations):

        X_w = X_train / weights[np.newaxis, :]
        adaptive_lasso = Lasso(alpha = LAMBDA, tol = 0.01, max_iter = 10000) # fit_intercept=False
        adaptive_lasso.fit(X_w, y_train)
        coef_ = adaptive_lasso.coef_ / weights
        weights = gprime(coef_)
        
        print ('Iteration #',k+1,':   ',p_obj(coef_))  # should go down
        
        iterations_list.append(k)
        coefficients_list.append(p_obj(coef_))
        
    print (np.mean((adaptive_lasso.coef_ != 0.0) == (coef_ != 0.0)))   
    
    if (X_names is None):
        X_names = list ( range(X_train.shape[1]) )
    coef = pd.Series(adaptive_lasso.coef_, index = X_names) # index = X_train.columns
    print('=============================================================================')
    print("Adaptive LASSO picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables.")
    print('=============================================================================')

    # plt.rcParams["figure.figsize"] = (18,8)

    # subplot of the predicted vs. actual

    plt.plot(iterations_list,coefficients_list,color = 'orange')
    plt.scatter(iterations_list,coefficients_list,color = 'green')
    plt.title('Iterations vs. p_obj(coef_)')
    plt.show()

    # plot of the coefficients'

    imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
    imp_coef.plot(kind = "barh", color = 'green',  fontsize=10)
    plt.title("Top and Botton 10 Coefficients Selected by the Adaptive LASSO Model", fontsize = 10)
    plt.show()

    return adaptive_lasso.coef_ # adaptive_lasso