import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
# from ..dr import lda # slow
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .plot_components import plot_components_1d, plot_components_2d

def supervised_dimension_reductions(X, y, legends = None):


    if X is None or X.shape[1] < 1:
        print('ERROR: X HAS NO FEATURE/COLUMN!')
        return
        
    labels = list(set(y))

    fig = plt.figure(figsize=(18, 4))
    # plt.title("", fontsize=14)

    ax = fig.add_subplot(131)
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit(X, y).transform(X)
    if X_lda.shape[1] == 1:
        plot_components_1d(X_lda, y, labels, legends=legends, ax = ax)
    else:
        plot_components_2d(X_lda, y, labels, legends=legends, ax = ax)
    ax.set_title('LDA')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    ax = fig.add_subplot(132)
    pls = PLSRegression(n_components=2)
    X_pls = pls.fit(X, y).transform(X)
    plot_components_2d(X_pls, y, labels, legends=legends, ax = ax)
    ax.set_title('PLS(R2 = ' + str(np.round(pls.score(X, y),3)) + ')')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
       
    plt.show()