'''
# We can directly import needed functions from cla

try:
    from cla.metrics import grid_search_svm_hyperparams, plot_svm_boundary, \
        plot_lr_boundary
except Exception as e:
    print(e)
    print('Please try: pip install cla==1.0.2 or above')

'''

'''
import cla
assert( '__version__' in cla.__dict__ and cla.__version__ >= '1.1.7')
'''

import sys
import os
import math
import re
import json
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import IPython.display

from sklearn.metrics import *  # we use global() to access the imported functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier # ExtraTreeClassifier only works in ensembles
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
# from scipy.integrate import quad
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import OneHotEncoder
from statsmodels.multivariate import manova
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from pyNNRW.elm import ELMClassifier
from pyNNRW.rvfl import RVFLClassifier
from ..io.pre import stratified_kennardstone_split
from ..vis.plt2base64 import plt2html
from ..vis.plot_components import plot_components_2d
from ..vis.feature_importance import plot_feature_importance
from ..vis.unsupervised_dimension_reductions import unsupervised_dimension_reductions
from ..vis.confusion_matrix import plot_confusion_matrix

# The following functions are copied from cla
def run_multiclass_clfs_presplit(X_train, y_train, X_test = None, y_test = None, clfs = 'all', cv_seed = 0, show = True):
    '''
    Use grid search to train and evaluate various multi-class classifiers.

    原生支持多分类的模型：

    naive_bayes.BernoulliNB (only for binary features)
    tree.DecisionTreeClassifier
    ensemble.ExtraTreesClassifier (only in ensembles)
    naive_bayes.GaussianNB
    neighbors.KNeighborsClassifier
    discriminant_analysis.LinearDiscriminantAnalysis
    svm.LinearSVC (multi_class=”crammer_singer”)
    linear_model.LogisticRegression(CV) (multi_class=”multinomial”)
    neural_network.MLPClassifier
    ensemble.RandomForestClassifier
    '''

    matplotlib.rcParams.update({'font.size': 17})

    n_classes = len(np.unique(y_train))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state = 2, stratify=y)
    dic_train_accs = {}
    dic_test_accs = {}

    html_str = ''
    kfold = KFold(n_splits=min(3,len(y_train)), shuffle=True, random_state=cv_seed)
    for base_learner, param_grid in zip( [GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(), 
                LinearSVC(multi_class="crammer_singer"), LogisticRegressionCV (multi_class="multinomial", max_iter=1000), 
                MLPClassifier(), 
                KNeighborsClassifier(), # NearestCentroid(),
                LinearDiscriminantAnalysis(), 
                # ELMClassifier(), RVFLClassifier()
                ], 
                [{}, {'max_depth': [1,2,3,4,5,10,20]}, {'n_estimators': list(range(2, max(10, n_classes)))},
                 {'C': [0.01, 0.1, 1, 10]}, {},
                 {'hidden_layer_sizes': [(x,) for x in range(1, 102, 10)], 'alpha': [0.0001, 0.01, 1] },
                  {'n_neighbors': list(range(1, max(5, n_classes)))}, 
                  {},
                  #{'n_hidden_nodes': [1, 2, 5, 10, 20, 50, 75, 100, 150, 200]}, {'n_hidden_nodes': [1, 2, 5, 10, 20, 40, 70, 100]}
                ]):
        
        if clfs != 'all' and str(base_learner) not in clfs:
            continue

        gs = GridSearchCV(base_learner, param_grid, cv= kfold, n_jobs=1, verbose=0)
        gs.fit(X_train, y_train)

        clf = gs.best_estimator_
        html_str += '<h4>' + str(clf) + '</h4>'
        if show:
            IPython.display.display(IPython.display.HTML('<h4>' + str(clf) + '</h4>'))

        if show:
            IPython.display.display(IPython.display.HTML('<h5>Training set</h5>'))
            
        y_pred = clf.predict(X_train) 
        dic_train_accs[str(clf)] = [clf.score(X_train, y_train)]
        report = '<p>top-1 acc on the training set = ' + str(round( clf.score(X_train, y_train), 3 )) + '</p>'        
        
        if len(np.unique(y_train)) >= 8:
            y_score = np.eye(n_classes)[y_pred] # one-hot encoding, make sure y is 0-indexed.
            if hasattr(clf, 'predict_proba') and callable(clf.predict_proba):
                y_score = clf.predict_proba(X_train)

            y_score = np.nan_to_num(y_score)
            dic_train_accs[str(clf)].append(top_k_accuracy_score(y_train, y_score, k=3))
            dic_train_accs[str(clf)].append(top_k_accuracy_score(y_train, y_score, k=5))
            report += '<p>top-3 acc = ' + str(round( top_k_accuracy_score(y_train, y_score, k=3),3)) + '</p>'
            report += '<p>top-5 acc = ' + str(round( top_k_accuracy_score(y_train, y_score, k=5),3)) + '</p>'

        if show:
            IPython.display.display(IPython.display.HTML(report)) 
            html_str += report

            n_classes = len(np.unique(y_train))
            _, ax = plt.subplots(1, 2, figsize=(8 + n_classes, 4 + round(n_classes/2))) # gridspec_kw={'width_ratios': [6, 3 + n_classes, 3 + n_classes]})

            plot_confusion_matrix(y_train, y_pred, normalize=False, ax=ax[0], cax = None) # cax = ax[2]
    
            # ax[2].set_title('confusion matrix \n(normalized)')
            plot_confusion_matrix(y_train, y_pred, normalize=True, ax=ax[1], cax = None) # cax = ax[4]
            
            # plt.tight_layout()
            html_str += plt2html(plt)

            plt.show() # plt.close()

            html_str = html_str + '<br/>'

            IPython.display.display(IPython.display.HTML('<br/>'))
            
        if X_test is None or y_test is None:
            print('No test set specified.')
            pass
        else:        
            if show:
                IPython.display.display(IPython.display.HTML('<h5>Test set</h5>'))

            y_pred = clf.predict(X_test) 
            # report = '<br/><pre>' + str(classification_report(y_test, y_pred)) + '</pre>'
            dic_test_accs[str(clf)] = [clf.score(X_test, y_test)]
            report = '<p>top-1 acc on the test set = ' + str(round( clf.score(X_test, y_test), 3 )) + '</p>'        

            if len(np.unique(y_train)) >= 8:
                y_score = np.eye(n_classes)[y_pred] # one-hot encoding, make sure y is 0-indexed.
                if hasattr(clf, 'predict_proba') and callable(clf.predict_proba):
                    y_score = clf.predict_proba(X_test)

                y_score = np.nan_to_num(y_score)
                dic_test_accs[str(clf)].append(top_k_accuracy_score(y_test, y_score, k=3))
                dic_test_accs[str(clf)].append(top_k_accuracy_score(y_test, y_score, k=5))
                report += '<p>top-3 acc = ' + str(round( top_k_accuracy_score(y_test, y_score, k=3),3)) + '</p>'
                report += '<p>top-5 acc = ' + str(round( top_k_accuracy_score(y_test, y_score, k=5),3)) + '</p>'

            if show:
                IPython.display.display(IPython.display.HTML(report)) 
                html_str += report

                n_classes = len(np.unique(y_train))
                _, ax = plt.subplots(1, 2, figsize=(8 + n_classes, 4 + round(n_classes/2))) # gridspec_kw={'width_ratios': [6, 3 + n_classes, 3 + n_classes]})

                # ax[0].set_title('classification report\n')
                #ax[0].text(0.1, 0, classification_report(y_test, y_pred), 
                #           fontsize = 18, horizontalalignment='right', verticalalignment='top')
                #ax[0].axis('off')

                # ax[1].set_title('confusion matrix\n')
                plot_confusion_matrix(y_test, y_pred, normalize=False, ax=ax[0], cax = None) # cax = ax[2]

                # ax[2].set_title('confusion matrix \n(normalized)')
                plot_confusion_matrix(y_test, y_pred, normalize=True, ax=ax[1], cax = None) # cax = ax[4]

                # plt.tight_layout()
                html_str += plt2html(plt)

                plt.show() # plt.close()

                html_str = html_str + '<br/>'

                IPython.display.display(IPython.display.HTML('<br/>'))

    matplotlib.rcParams.update({'font.size': 12})    
    return dic_train_accs, dic_test_accs, html_str


def run_multiclass_clfs(X, y, clfs = 'all', split = .3, split_type = 'ks', cv_seed = 0, show = True):
    '''
    This version will first split the dataset into a training set and test set.

    Parameters
    ----------
    split_type: 
        'ks' - use balanced kennard stone split. 
        any integer value - use sklearn train_test_split(). The value is used as its random seed.
        In general, ks works better than the vanilla random split.
    
    '''
    if split_type == 'ks':
        X_train, X_test, y_train, y_test = stratified_kennardstone_split(X, y, test_size=split)
    else:
        if type(split_type) is not int:
            print("Warning: split_type is not an integer. Use 0 as the default value.")
            split_type = 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state = split_type, stratify=y)
    return run_multiclass_clfs_presplit(X_train, y_train, X_test, y_test, clfs = clfs, cv_seed = cv_seed, show = show)


def visualize_multiclass_result(dic_train_accs, dic_test_accs):

    html_str = '<table><tr><th>classifier with optimal hparams</th><th>train accuracy*</th><th>test accuracy*</th></tr>'
    for k, v in dic_train_accs.items():
        # , dic_test_accs
        html_str += '<tr><td>' + k + '</td><td>' + str(np.round(v,3))+ '</td><td>' + str(np.round(dic_test_accs[k],3)) + '</td></tr>'
    html_str +='</table>'
    html_str += '<i>*if a list, it means top-1/3/5 accuracies.</i>'
    
    IPython.display.display(IPython.display.HTML(html_str))
    return html_str