import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ._group_lasso import LogisticGroupLasso

def window_op(x_names, region, window = 'rbf', sd = 1, display = False):
    '''
    Parameters
    ----------
    x_names : the entire x axis ticks, e.g., [50.04522, 51.97368, ... 2310.208] cm-1 for Raman
    region : e.g., [100,200]
    window : can be 'rbf', 'sinc', 'logistic', 'uniform'. Uniform is just averaging filter.
    sd : controls the std of rbf kernel. The standard region will lie inside +/-3d of the gaussian distribution.
    
    Return
    ------ 
    op array. Has the length of x_names.

    Note
    ----
    The resolution is 0.1 cm-1. When x_names has large/small intervals, spikes may miss or spike becomes a rectangle.
    '''
    
    if region[0] > np.max(x_names) or region[1] < np.min(x_names): # out of range
        return np.zeros(len(x_names))
    
    op = [0]*round(10*(np.max(x_names))+1)
    if window == 'spike' or window == 'vanilla':
        op[round(10*region[0])] = 1      
    elif window == 'uniform' or window == 'rectangle' or window == 'average':
        if region[-1]-region[0] != 0:
            op[round(10*region[0]):round(10*region[-1])] = [10 / (round(10*region[-1])-round(10*region[0]))] * (round(10*region[-1])-round(10*region[0]))
        elif region[-1]-region[0] == 0:
            op[round(10*region[0])] = 1
    elif window == 'triangle':
        if region[-1]-region[0] != 0:
            d = 10*region[-1]-10*region[0]
            h = 20/d
            k = h/(d/2)
            op_triangle = []
            for x in range(round(10*region[0]),round(10*region[-1])+1):
                if x >= round(10*region[0]) and x <  d/2+round(10*region[0]):
                    op_value1 = k*x+(h-(d/2+10*region[0])*k)
                    op_triangle.append(op_value1)
                elif x >= d/2+round(10*region[0]) and x <=round(10*region[-1]):
                    op_value2 = -k*x+(h+(d/2+10*region[0])*k)
                    op_triangle.append(op_value2)
            op[round(10*region[0]):round(10*region[-1])+1] = op_triangle
        elif region[-1]-region[0] == 0:
            op[round(10*region[0])] = 1
    elif window == 'gaussian' or window == 'rbf':
        if region[-1]-region[0] != 0:
            start_region,end_region=round(10*region[0]),round(10*region[-1])
            sd *= (end_region-start_region)/6 # use 6-simga region
            x = np.linspace(0, round(10*(x_names[-1]) + 2), round(10*(x_names[-1]) + 2))
            op = norm.pdf(x, loc=(start_region+end_region)/2, scale=sd)
            op = op / op.sum()
        elif region[-1]-region[0] == 0:
            op[round(10*region[0])] = 1
    
    #############  map range ##############
    mop = []
    
    for idx in x_names:
        mop.append(op[round(10*idx)])
    # normalization: make sure integral is always 1
    mop = np.array(mop)
    if mop.sum() > 0:
        mop = mop/mop.sum()
    
    if display:
        plt.title('Window Operator ' + window + ' on ' + str(region))
        plt.plot(x_names,mop)
        plt.show()

    return mop

def window_fs(X, x_names, regions, window = 'rbf', sd = 1, display = False):
    '''
    Convert one data to binned features.
    Break down the axis as sections. Each seection is an integral of the signal intensities in the region.
    Integration can be done by radius basis function / sinc kernel, etc.

    window : Apply a window operator to a continuous region. Can be 'rbf / gaussian', 'uniform', 'spike', 'triangle'. Uniform is just averaging filter.
    '''

    fss = []
    filtered_regions = [] # filtered regions
    filtered_region_centers = [] # filtered region centers
    
    # filter regions x_names
    
    for region in regions:
        if np.min(x_names) <= region[0] and np.max(x_names) >= region[-1]:
            filtered_regions.append(region)
            filtered_region_centers.append((region[0]+region[1])/2)
            
    for i, x in enumerate(X):
        # the discrete features for one data sample
        Fs = []
        for region in regions:
            if np.min(x_names) <= region[0] and np.max(x_names) >= region[-1]:
                op = window_op(x_names, region, window, sd, display = False)
                F = np.dot(op, x)
                Fs.append(F)

        if display:
            plt.title('Feature Selection on Sample ' + str(i))
            plt.xlabel('Region Centers')
            plt.ylabel('Feature')
            plt.scatter(filtered_region_centers, Fs, s=50, facecolors='0.8', edgecolors='0.2', alpha = .5)
            plt.show()

        fss.append(Fs)

    return np.array(fss), filtered_regions, filtered_region_centers

def raman_window_fs(X , x_names, raman_peak_list, window = 'rbf', sd = 1, display = False):
    '''
    Extract features from Raman spectra with specified window operator.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data.
    x_names : array-like, shape (n_features,)
        The feature names, i.e., raman shifts / wavenumbers (in cm-1), of the input data.
    raman_peak_list : list of RamanPeak objects
    window : Apply a window operator to a continuous region. Can be 'rbf / gaussian', 'uniform', 'spike', 'triangle'. Uniform is just averaging filter.
    sd : standard deviation of the 'rbf / gaussian' window.
    display : whether display the per-sample feature selection result.

    Returns
    -------
    Fss : array-like, shape (n_samples, n_extracted_features)
        The extracted features.
    group_info : list of extracted feature information
        group_info[i] = [chemical, vibration, peak_start, peak_end, group_id]
    group_ids : list of group ids, e.g., [-1,-1,1,1,1,2,2,-1,-1]
        Required by the group_lasso function.
    filtered_regions : list of regions that are within the range of x_names (in cm-1)
    filtered_region_centers : list of region centers that are within the range of x_names (in cm-1)
    '''

    raman_peak_key_list = [[x.chemical +x.vibration]+[(x.peak_start,x.peak_end)] for x in raman_peak_list]
    
    Fss = []
    filtered_keys = []
    filtered_regions = [] # filtered regions
    filtered_region_centers = [] # filtered region centers
        
    # filter regions x_names 
    for sublist in raman_peak_key_list:
        if np.min(x_names) <= sublist[-1][0] and np.max(x_names) >= sublist[-1][-1]:
            filtered_keys.append(sublist) # 不仅添加该物质的region，同时将其对应的键值添加进去
            filtered_regions.append(sublist[-1])
            filtered_region_centers.append((sublist[-1][0]+sublist[-1][1])/2)
    
    # 分组----------------------------------------------------
    # 先对filtered_keys不同的物质键值进行排列
    d = {}
    group_num = 0
    for sublist in filtered_keys:
        if sublist[0] not in d:
            d[sublist[0]] = group_num
            group_num += 1
    # filtered_keys中物质键值相同的给予相同的组号
    result = []
    for sublist in filtered_keys:
        group_id = d.get(sublist[0], -1)
        if group_id == -1:
            result.append(sublist + [-1])
        else:
            result.append(sublist + [group_id])        
    # 将result中只出现了一次的组号修改为-1
    count_dict = {}
    for sublist in result:
        count_dict[sublist[2]] = count_dict.get(sublist[2], 0) + 1
    group_info = [[sublist[0], sublist[1], -1] if count_dict[sublist[2]] == 1 else sublist for sublist in result] # 分组表[物质和键，对应的region，对应的组号]             
    # 分组完成------------------------------------------------------------------------------
    

    for i, x in enumerate(X):
        # the discrete features for one data sample
        Fs = []
        for sublist in raman_peak_key_list:
            if np.min(x_names) <= sublist[-1][0] and np.max(x_names) >= sublist[-1][-1]:
                op = window_op(x_names, sublist[-1], window, sd, display = False)
                F = np.dot(op, x)
                Fs.append(F)
        if display:
            plt.figure(figsize=(10, 3))
            plt.title('Feature Selection on Sample ' + str(i))
            plt.xlabel('Region Centers')
            plt.ylabel('Feature')
            plt.scatter(filtered_region_centers, Fs, s=50, facecolors='0.8', edgecolors='0.2', alpha = .5)
            plt.show()

        Fss.append(Fs)
    group_ids = [x[-1] for x in group_info] 
    return np.array(Fss), group_info, group_ids, filtered_regions, filtered_region_centers

def group_lasso(X, y, groups = None, group_reg = 100, l1_reg = 100, split = 0.3, verbose = False):
    '''
    Group Lasso Feature Selection. 
    The most important param is groups. It can be generated from raman_window_fs(), i.e., the 3rd returned result.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    groups : array-like, shape (n_features,). This is the most important parameter for group lasso. It specifies which group each column corresponds to. For columns that should not be regularised, the corresponding group index should either be None or negative. For example, the list [1, 1, 1, 2, 2, -1] specifies that the first three columns of the data matrix belong to the first group, the next two columns belong to the second group and the last column should not be regularised.
        Use raman.get_groups() to get the groups.
    split : test set split ratio. Default is 0.3.
    group_reg : group regularization strength
    l1_reg : l1 regularization strength

    Returns
    -------
    coef : array-like, shape (n_features,). Coefficients returned from the LogisticGroupLasso model, indicating the importance of each feature.
    mask : array-like, shape (n_features,). The mask of selected features. 1 for selected, 0 for not selected. It can be very sparse.
    acc : accuracy of the model on the test set.
    '''

    if groups is None or len(groups) == 0: # degraded to routine lasso
        groups = - np.ones(X.shape[1]) # -1 for ungrouped features
    
    LogisticGroupLasso.LOG_LOSSES = True

    gl = LogisticGroupLasso(
        groups = groups, # Iterable that specifies which group each column corresponds to. For columns that should not be regularised, the corresponding group index should either be None or negative. For example, the list [1, 1, 1, 2, 2, -1] specifies that the first three columns of the data matrix belong to the first group, the next two columns belong to the second group and the last column should not be regularised.
        group_reg = group_reg, # If ``group_reg`` is an iterable (pre-initilized weights), then its length should be equal to the number of groups.
        l1_reg = l1_reg, # default 0.05
        scale_reg="inverse_group_size", # for dummy vars, should be None. In statistics and econometrics, particularly in regression analysis, a dummy variable is one that takes only the value 0 or 1 to indicate the absence or presence of some categorical effect that may be expected to shift the outcome.
        # subsampling_scheme=1,
        # supress_warning=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=y)

    gl.fit(X_train, y_train)

    # Extract info from estimator
    pred_c = gl.predict(X_test)
    # w_hat = gl.coef_
    # Compute performance metrics
    acc = (pred_c == y_test).mean()

    if (verbose):
        
        print(f"Group Lasso Parameters (group_reg, l1_reg, split): {group_reg, l1_reg, split}")
        print(f"Number of selected features: {gl.sparsity_mask_.sum()}")
        print(f"Test accuracy: {acc}")
        print(group_reg,l1_reg)
    
    return gl.coef_, gl.sparsity_mask_, acc


"""

# Sliding N-Gram LASSO, i.e., Group lasso with SLIDING WINDOW of EQUAL WIDTH
# These functions are obselete. We can use io.pre.x_binning + lasso to achieve the same effect.

def group_lasso(X_scaled, y, WIDTH, offset = 0, LAMBDA = 1, ALPHA = 0.5):
    '''
    Group Lasso Feature Selection

    Parameters
    ----------
    X_scaled : X, should be rescaled;
    y : target var;
    WIDTH : sliding window's width; 
    LAMBDA : regularization coefficient; 
    ALPHA : ratio of L1 vs Group;
    '''

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
                                                   test_size = cv_size, stratify=y)
                    
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
                    
                    if verbose:
                        print('R2 = ', SCORES[-1])

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

    print('top-' + str(MAXF) + ' common features and their frequencies: ', Counter(CAT_FS).most_common(MAXF))

    plt.yticks([])
    if (idxx <= 10):
        plt.legend()
    plt.show()

    COMMON_FSI = []
    for f in Counter(CAT_FS).most_common(MAXF):
        COMMON_FSI.append(f[0])
               
    return np.array(COMMON_FSI)

"""