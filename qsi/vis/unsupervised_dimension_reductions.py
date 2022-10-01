import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE
from .plotComponents2D import plotComponents2D
from .plotComponents1D import plotComponents1D


def unsupervised_dimension_reductions(X, y, legends = None):

    # %matplotlib notebook
    
    if X is None or X.shape[1] < 1:
        print('ERROR: X HAS NO FEATURE/COLUMN!')
        return
        
    labels = list(set(y))

    fig = plt.figure(figsize=(18, 10))
    # plt.title("", fontsize=14)

    ax = fig.add_subplot(331) # , projection='polar' , projection='3d'
    
    if X.shape[1] == 1:
        plotComponents1D(X, y, labels, legends=legends, ax = ax)
        ax.set_title('X has only 1 FEATURE/COLUMN. Plot X directly.')
        return X
        
    # Standard PCA: equals to the linear-kernel PCA
    pca = PCA(n_components=2) # keep the first 2 components
    X_pca = pca.fit_transform(X)
    plotComponents2D(X_pca, y, labels, legends=legends, ax = ax)
    ax.set_title ('PCA(var% ' + str(np.round(pca.explained_variance_ratio_[0:5],3)) + ')')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    ax = fig.add_subplot(332)  
    X_kpca = None
    try:
        kpca = KernelPCA(n_components=2, kernel='rbf', eigen_solver='arpack') # keep the first 2 components. default gamma = 1/n_features
        X_kpca = kpca.fit_transform(X)    
        plotComponents2D(X_kpca, y, labels, legends=legends, ax=ax)     
    except Exception as e:
        # if kernel PCA throws exception, print the error message
        '''
        If eigenvalue computation does not converge, an error occurred, 
        or b matrix is not definite positive. 
        Note that if input matrices are not symmetric or Hermitian, 
        no error will be reported but results will be wrong.
        ---------------
        Use 'arpack' solves the exception
        '''
        ax.text(0.01,0.5,'rbf-kernel PCA exception: ' + str( getattr(e, 'message', repr(e) ) ) )    
    ax.set_title ('Kernel PCA (rbf)')    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    ax = fig.add_subplot(333)  
    X_kpca = None
    try:
        kpca = KernelPCA(n_components=2, kernel='poly')
        X_kpca = kpca.fit_transform(X)    
        plotComponents2D(X_kpca, y, labels, legends=legends, ax=ax)     
    except Exception as e:
        # if kernel PCA throws exception, print the error message
        ax.text(0.01,0.5,'poly-kernel PCA exception: ' + str( getattr(e, 'message', repr(e) ) ) )    
    ax.set_title ('Kernel PCA (poly)')    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    ax = fig.add_subplot(334)  
    X_kpca = None
    try:
        kpca = KernelPCA(n_components=2, kernel='cosine')
        X_kpca = kpca.fit_transform(X)    
        plotComponents2D(X_kpca, y, labels, legends=legends, ax=ax)     
    except Exception as e:
        # if kernel PCA throws exception, print the error message
        ax.text(0.01,0.5,'cosine-kernel PCA exception: ' + str( getattr(e, 'message', repr(e) ) ) )    
    ax.set_title ('Kernel PCA (cosine)')    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    ax = fig.add_subplot(335)
    X_kpca = None
    try:
        kpca = KernelPCA(n_components=2, kernel='sigmoid', eigen_solver='arpack')
        X_kpca = kpca.fit_transform(X)    
        plotComponents2D(X_kpca, y, labels, legends=legends, ax=ax)     
    except Exception as e:
        # if kernel PCA throws exception, print the error message
        ax.text(0.01,0.5,'sigmoid-kernel PCA exception: ' + str( getattr(e, 'message', repr(e) ) ) )    
    ax.set_title ('Kernel PCA (sigmoid)')    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    ax = fig.add_subplot(336)
    msg_duplicate2 = ''
    if X.shape[1] == 2: # requires > n_components
        # X = np.hstack((X,X[:,-2:]))
        X_tsvd = X
        msg_duplicate2 = '\nTruncatedSVD requires X have more columns than n_components. \nDrawing the original X.'
    else:
        tsvd = TruncatedSVD(n_components=2)
        X_tsvd = tsvd.fit_transform(X)
        plotComponents2D(X_tsvd, y, labels, legends=legends, ax=ax)
    ax.set_title ('Truncated SVD' + msg_duplicate2 ) # PCA on uncentered data
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    ax = fig.add_subplot(337)
    mds = MDS(n_components=2)
    X_mds = mds.fit_transform(X)
    plotComponents2D(X_mds, y, labels, legends=legends, ax = ax)
    ax.set_title ('MDS')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    ax = fig.add_subplot(338)    
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    plotComponents2D(X_tsne, y, labels, legends=legends, ax=ax)
    ax.set_title ('t-SNE')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()     

    # return X_pca, X_kpca x 4, X_tsvd, X_mds, X_tsne