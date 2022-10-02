from sklearn.decomposition import PCA
from qsi.cla import ensemble
from qsi.vis import *
from collections import Counter

def fsse_cv(X_scaled,y, X_names = None, N = 30, base_learner=ensemble.create_elmcv_instance, \
    WIDTHS = [1, 2, 5, 10, 100], ALPHAS = [0.5,0.75,1.0], display = True, verbose = True):

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
        acc = fsse.evaluate(X_scaled, y) # accuracy

        biggest_fsse_fs, fs_importance = fsse.get_important_features()

        if verbose:
            print('acc = ', acc)
            print('meta_learner.l1_ratio_ = ', fsse.meta_learner.l1_ratio_)  
            print('Non-zero feature coefficients:', len(biggest_fsse_fs))

        CAT_FS += biggest_fsse_fs
        xfsse = X_scaled[:,biggest_fsse_fs] # 前N个系数 non-zero
        
        if display:

            plot_feature_importance(np.abs(fs_importance), 'FSSE FS Result, GROUP SIZE = ' + str(SPLIT))
        
            plt.figure()
            xfsse_pca = PCA(n_components = 2).fit_transform(xfsse)
            plotComponents2D(xfsse_pca, y, set(y), ax = None)
            plt.legend()
            plt.show()

    COMMON_FSI = []
    for f in Counter(CAT_FS).most_common(N):
        COMMON_FSI.append(f[0])

    if verbose:
        print('top-' + str(N) + ' common features and their frequencies: ', Counter(CAT_FS).most_common(N))

    return np.array(COMMON_FSI)