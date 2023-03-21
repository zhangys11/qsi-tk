import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from ..dr import lda
from .plot_components import plot_components_1d, plot_components_2d

def supervised_dimension_reductions(X, y, legends = None):


    if X is None or X.shape[1] < 1:
        print('ERROR: X HAS NO FEATURE/COLUMN!')
        return
        
    labels = list(set(y))

    fig = plt.figure(figsize=(6, 10))
    # plt.title("", fontsize=14)

    ax = fig.add_subplot(131)
    X_lda = lda(X, y)
    if X_lda.shape[1] == 1:
        plot_components_1d(X, y, labels, legends=legends, ax = ax)
        ax.set_title('X has only 1 FEATURE/COLUMN. Plot X directly.')   
    plot_components_2d(X_lda, y, labels, legends=legends, ax = ax)
    ax.set_title('LDA')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    print('LDA is like PCA, but focuses on maximizing the seperatibility between categories. \
LDA for two categories tries to maximize distance between group means, meanwhile minimize intra-group variances. \n\
{ (\mu_1 - \mu_2)^2 } \over { s_1^2 + s_2^2 }')
    print('The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix.\n \
        The fitted model can be used to reduce the dimensionality of the input by projecting it to the most discriminative directions.')
    print('Risk of using LDA: Possible Warning - Variables are collinear. \n\
LDA, like regression techniques involves computing a matrix inversion, which is inaccurate if the determinant is close to 0 (i.e. two or more variables are almost a linear combination of each other). \n\
More importantly, it makes the estimated coefficients impossible to interpret. If an increase in X1 , say, is associated with an decrease in X2 and they both increase variable Y, every change in X1 will be compensated by a change in X2 and you will underestimate the effect of X1 on Y. In LDA, you would underestimate the effect of X1 on the classification. If all you care for is the classification per se, and that after training your model on half of the data and testing it on the other half you get 85-95% accuracy I\'d say it is fine. ')
    

    ax = fig.add_subplot(132)
    pls = PLSRegression(n_components=2)
    X_pls = pls.fit(X, y).transform(X)
    plot_components_2d(X_pls, y, labels, legends=legends, ax = ax)
    ax.set_title('PLS')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    print('PLS STEPS: \n\
X and Y are decomposed into latent structures in an iterative way. \n\
The latent structure corresponding to the most variation of Y is explained by a best latent strcture of X. \n\n \
ADVANTAGES: Deal with multi-colinearity; Interpretation by data structure')
    
    plt.show()