from sklearn.decomposition import PCA
from ..cla import ensemble
from ..vis import *
from collections import Counter


def fsse_cv(X_scaled,y, X_names = None, N = 30, base_learner=ensemble.create_elmcv_instance, \
    WIDTHS = [1, 2, 10], ALPHAS = [0.5,0.75,1.0], display = True, verbose = True):

    '''
    Feature subspace based ensemble (FSSE)

    Parameters
    ----------
    base_learner : e.g., ensemble.create_elmcv_instance or create_rvflcv_instance
    WIDTHS : group size / window width. each base learner will process one group.  
    ALPHAS : meta learner's L1 ratio    
    '''

    CAT_FS = []

    for SPLIT in WIDTHS:
        
        if verbose:
            print('====== Group Size = ', SPLIT, '======')

        fsse = ensemble.FSSE(base_learner,
                    feature_split = SPLIT, 
                    meta_l1_ratios = ALPHAS) # use l1_ratio > 0.5 to get sparse results
        fsse.fit(X_scaled, y)
        acc, r2 = fsse.evaluate(X_scaled, y)

        _, fs_importance = fsse.get_important_features() 
        # # there may be some redundant paddings for the last group
        # valid_idx = np.where(biggest_fsse_fs < X_scaled.shape[1])
        # biggest_fsse_fs = biggest_fsse_fs [ valid_idx ]
        fs_importance = fs_importance[ :X_scaled.shape[1] ]

        if verbose:
            print('acc = ', round(acc,3))
            print('R2 = ', round(r2))
            print('meta_learner.l1_ratio_ = ', fsse.meta_learner.l1_ratio_) 

        biggest_fsse_fs = (np.argsort(np.abs(fs_importance))[-N:])[::-1]

        CAT_FS += list(biggest_fsse_fs)
        xfsse = X_scaled[:,biggest_fsse_fs] # 前N个系数 non-zero
        
        if verbose:
            print('Most important feature indices (WIDTH=' + str(SPLIT) + '): ', biggest_fsse_fs)
            if X_names is not None and X_names != []:
                print('Most important feature names (WIDTH=' + str(SPLIT) + '): ', np.array(X_names) [biggest_fsse_fs])

        if display:

            plot_feature_importance(np.abs(fs_importance), X_names, 'FSSE FS Result, GROUP SIZE = ' + str(SPLIT))
        
            plt.figure()
            xfsse_pca = PCA(n_components = 2).fit_transform(xfsse)
            plot_components_2d(xfsse_pca, y, set(y), ax = None)
            plt.legend()
            plt.show()

    COMMON_FSI = []
    commons = Counter(CAT_FS).most_common(N)
    for f in commons:
        COMMON_FSI.append(f[0])

    if verbose:
        print('top-' + str(N) + ' common features and their frequencies: ', commons)

    return np.array(COMMON_FSI)