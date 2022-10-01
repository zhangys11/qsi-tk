import numpy as np
import matplotlib.pyplot as plt
from .unsupervised_dimension_reductions import *

# plot the importance for all features
def plot_feature_importance(feature_importances, feature_names, title, \
    xtick_angle=None, xlabel = '', ylabel = '', row_size = None, figsize=(12,2)):
    
    feature_importances = np.array(feature_importances)

    if row_size is None: # use dynamic row size
        row_size = round( len(feature_importances) / 5000 + 0.5) * 100

    # matrix chart
    ROW_SIZE = row_size # math.ceil(feature_importances.size / 100) * 10     
    rows = int((feature_importances.size - 1)/ROW_SIZE) + 1
    fig = plt.figure(figsize=(12,12*rows/ROW_SIZE+0.5))
    if title:
        plt.title(title + "\n" + 'importance marked by color depth')
    plt.axis('off')
    
    for i in range(rows):
        ax = fig.add_subplot(rows, 1, i + 1)
        ax.axis('off')
        arr = feature_importances[i*ROW_SIZE: min(i*ROW_SIZE+ROW_SIZE - 1, feature_importances.size - 1)]
        s = ax.matshow(arr.reshape(1,-1), cmap=plt.cm.Blues)    
    
    # bar chart
    plt.figure(figsize=figsize)
    if title:
        plt.title(title + "\n" + 'importance marked by bar height')
    if feature_names is None:
        feature_names = range(feature_importances.size)
    plt.bar(feature_names, feature_importances, alpha=.8) # , width=2
    if xtick_angle is None:
        plt.xticks([])
    else:
        plt.xticks(rotation = 90)
    plt.yticks([])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

'''
# plot the importance of the selected features
def plot_important_features(feature_names, feature_importances, xlabel = '', ylabel = '', title = '', figsize=(20,3)):

    assert len(feature_names) == len(feature_importances)
    plt.figure(figsize=figsize)
    plt.bar(feature_names.astype(str), feature_importances)

    plt.xticks(rotation = 90)
    plt.title(title)
    plt.show()
'''


def visualize_important_features(X, y, coef, X_names = None, title = None, row_size = 100, eps = 1e-10):
    '''
    epsilon - cut threshold
    '''
    N = np.count_nonzero(np.abs(coef) > eps)
    biggest_fs = np.argsort(np.abs(coef))[::-1][:N] # take last N item indices and reverse (ord desc)
    X_fs = X[:,biggest_fs] # 前N个系数 non-zero

    print('Important feature Number:',N)
    print('Important feature Indice:',biggest_fs)
    if (X_names and len(X_names) == X.shape[1] ):
        print('Important features:', np.array(X_names)[biggest_fs])
    print('Important feature coefficents:', coef[biggest_fs])
    
    plot_feature_importance(np.abs(coef), X_names, title, row_size = row_size)
    unsupervised_dimension_reductions(X_fs, y)


def get_important_features(X, coef, eps = 1e-10):

    N = np.count_nonzero(np.abs(coef) > eps)
    biggest_fs = np.argsort(np.abs(coef))[::-1][:N]
    X_fs = X[:,biggest_fs]

    return X_fs