'''
Dataset augmentation methods
'''
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ...vis import plot_components_2d
from . import SMOTE, KDE, VAE

def upsample(target_path, X, y, X_names, method = 'SMOTE', folds = 3, d = 0.5, 
epochs = 10, batch_size = 100, cuda = True, display = False, verbose = True):
    '''
    Upsample a dataset by SMOTE, KDE, VAE, GAN (todo), Gaussian (todo), or ctGAN.

    Parameters
    ----------
    target_path : where to save the generated dataset.
    X_names : the labels for each X feature
    folds : expand to N folds
    d : sampling distance in SMOTE
    epochs, batch_size, cuda : ctgan-specific params
    '''

    if folds == 0:
        return X, y

    if method == 'SMOTE':
        Xn, yn = SMOTE.expand_dataset(X, y, d, folds)
    elif method == 'KDE':
        Xn, yn = KDE.expand_dataset(X, y, folds)
    elif method == 'VAE':
        Xn, yn = VAE.expand_dataset(X, y, folds)
    elif method == 'ctGAN' or method == 'CTGAN':
        from . import ctGAN
        Xn, yn = ctGAN.expand_dataset(X, y, folds, epochs, batch_size, cuda, verbose)
    else:
        print('Unsupported method: ' + method + '. Use SMOTE / ctGAN / KDE.' )
        return X, y

    dfX = pd.DataFrame(Xn)
    dfX.columns = X_names

    dfY = pd.DataFrame(yn)
    dfY.columns = ['label']

    df = pd.concat([dfY, dfX], axis=1)
    df = df.sort_values(by=['label'], ascending=True)
    df.to_csv(target_path, index=False)  # don't create the index column

    if display:

        pca = PCA(n_components=2) # keep the first 2 components
        X_pca = pca.fit_transform(Xn)
        plot_components_2d(X_pca, yn)
        plt.title('PCA of the augmented dataset')
        plt.show()

    return Xn, yn