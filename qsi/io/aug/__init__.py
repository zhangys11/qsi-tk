# data augmentation
from . import SMOTE, ctGAN
from sklearn.decomposition import PCA
from ...vis import plotComponents2D
import matplotlib.pyplot as plt

def upsample(target_path, X, y, X_names, method = 'SMOTE', folds = 3, d = 0.5, 
epochs = 10, batch_size = 100, cuda = True, display = False, verbose = True):
    '''
    Upsample a dataset by SMOTE, GAN (todo), Gaussian (todo), or ctGAN.

    Parameters
    ----------
    X_names : the labels for each X feature
    folds : expand to N folds
    d : sampling distance in SMOTE
    epochs, batch_size, cuda : ctgan params
    '''

    if method == 'SMOTE':
        Xn, yn = SMOTE.expand_dataset(X, y, X_names, target_path, d, folds)
    elif method == 'ctGAN':
        Xn, yn = ctGAN.expand_dataset(X, y, X_names, target_path, folds, epochs, batch_size, cuda, verbose)

    if display:

        pca = PCA(n_components=2) # keep the first 2 components
        X_pca = pca.fit_transform(Xn)
        plotComponents2D(X_pca, yn)
        plt.title('PCA of extended dataset')
        plt.show()

    return Xn, yn