import os.path
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display
import pickle
import joblib

from qsi.io.pre import filter_dataset
from . import io
from .vis import *
from .dr import dataset_dct_row_wise
from .dr import mf, lda
from .fs import *
from .cla import metrics
from .cla import grid_search_svm_hyperparams, plot_svm_boundary, plot_lr_boundary

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA, SparsePCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC # SVC(C=1.0, ...) 与 nuSVC(nu = 0.5, ...) 的区别：前者使用C～[0，inf），后者使用nu～(0,1]，惩罚性的解释不同，优化都可以使用GridSearch方法
from scipy.signal import savgol_filter

def analyze(id, x_range = None, y_subset=None, shift = 100, pre=None, pre_param=None):
    '''
    A general and standard data analysis flow. Provides an overview for the target dataset.
    
    Parameters
    ----------
    id : dataset id or a full path
    x_range, y_subset : select specific rows and cols for analysis, if you don't wish to use the entire dataset.
    shift : interval between classes in the average wave plot
    pre, pre_param : additional preprocessing (with params) applied, e.g. 'max_bin, 100'
        pre = 'max(target_dim = pre_param)' : binning and take the maximum in each interval
        pre = 'sum(target_dim = pre_param)' or 'rect(window_size = pre_param)' : 
            binning and take the sum/mean in each interval, i.e., rectangle window
        pre = 'tri(target_dim = pre_param)' : triangle window kernel.

        pre = 'rbf(sigma = pre_param)' : applied a radial basis function kernel. Not implemented yet.
        pre = 'z(sigma = pre_param)' : applied z (normal distribution PDF) kernel. Not implemented yet.
        pre = 'trapezoidal(pre_param)' : trapezoid window kernel. Not implementated yet.
        
        pre = 'diff' : first-order op. We use io.pre.diff_dataset()
        default is '' or None. 
    '''
    display(HTML('<h2>数据加载 Load Dataset</h2>'))

    labels = None
    if (id in io.DATASET_MAP.keys()):
        X, y, X_names, _, labels = io.load_dataset(id, x_range=x_range, y_subset=y_subset, shift = shift)
    elif os.path.exists(id):
        X, y, X_names, _ = io.open_dataset(id, x_range=x_range, y_subset=y_subset, shift = shift)
    else:
        display(HTML('<h3>数据加载失败，请传入正确的id或文件路径。<br/>Load data failed. Please specific a correct dataset ID or file path.</h3>'))

    display(HTML('<hr/><h2>预处理 Preprocessing </h2>'))
    
    if pre:
        display(HTML('<h3>' + pre + '</h3>'))
        if pre in ['max', 'sum', 'rect', 'tri', 'mean']:
            X, X_names = io.pre.x_binning(X,X_names,target_dim=pre_param,flavor=pre)
        elif pre == 'diff':
            X = io.pre.diff_dataset(X)
            X_names = X_names[1:] # pop the 1st x label, as diff reduces dim by 1

        display(HTML('预处理后的维度：X.shape = ' + str(X.shape) +'<hr/>'))

    analyze_dataset(X, y, X_names)

def analyze_dataset(X, y, X_names, cla_feature_num = 10):
    '''
    Parameters
    ----------
    X : preprocessed data matrix
    cla_feature_num : how many features to use for classification
    '''

    if X_names is None or X_names == []:
        X_names = list(range(X.shape[1]))
    
    display(HTML('<h3>Savitzky-Golay Filter 滤波</h3>'))
    display(HTML('<h4>此处仅做额外的预处理效果预览，并不叠加到后续分析中，需要用户自行判断使用。</br>This preprocessing effect is just for preview. Users should choose the algorithms on demand. </h4>'))
    
    # Savgol filtering
    np.set_printoptions(precision=2)
    X_sg = savgol_filter(X, window_length = 9, polyorder = 3, axis = 1)  # axis = 0 for vertically; axis = 1 for horizontally
    plt.figure(figsize = (40,10))
    plt.scatter(X_names, np.mean(X_sg,axis=0).tolist()) 
    plt.title(u'Averaged Spectrum After Savitzky-Golay Filtering', fontsize=30)
    plt.show()

    display(HTML('<h3>高通滤波|消除基线漂移 Highpass Filter | Baseline Drift Removal</h3>'))
    display(HTML('<h4>此处仅做额外的预处理效果预览，并不叠加到后续分析中，需要用户自行判断使用。</br>This preprocessing effect is just for preview. Users should choose the algorithms on demand. </h4>'))

    # Butterworth filter
    X_f = filter_dataset(X, nlc = 0.002, nhc = None)  # axis = 0 for vertically; axis = 1 for horizontally
    plt.figure(figsize = (40,10))
    plt.scatter(X_names, np.mean(X_f,axis=0).tolist()) 
    plt.title(u'Averaged Spectrum After Butterworth Filter\nIf the filtered result is not good, you will need to finetune the Butterworth highpass filter cutoff freq.', fontsize=30)
    plt.show()

    display(HTML('<hr/><h3>特征缩放 Feature Scaling</h3>'))
    # normalization

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # scaling to [0,1]

    mm_scaler = MinMaxScaler()
    X_mm_scaled = mm_scaler.fit_transform(X)
    print('X_mm_scaled is rescaled to [0,1]. We use X_mm_scaled in the MFDR and feature selectioin section.')

    display(HTML('<hr/><h2>降维 Dimensionality Reduction</h2>'))
    # Dimension Reduction & Visualization

    if X.shape[1] > 6000:
         display(HTML('<h3>数据维度过高，降维算法运行时间过长，请等待。<br/>The dataset is high-dimensional (>6000). Some DR algorithms may take very long time. Please wait.</h3>'))
    else:
        # SparsePCA(n_components=None, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method=’lars’, n_jobs=1, U_init=None, V_init=None, verbose=False, random_state=None)    # [Warn] SparsePCA takes very long time to run.
        X_spca = SparsePCA(n_components=2).fit_transform(X) # keep the first 2 components. default L1 = 1; default L2 = 0.01 pca.fit(X) X_spca = pca.transform(X) print(X_spca.shape)
        ax = plotComponents2D(X_spca, y)
        ax.set_title('Sparse PCA')
        plt.show()    
        display(HTML('<hr/>'))

        for kernel in ['linear', 'rbf','sigmoid', 'cosine','poly']:
            try:
                X_kpca = KernelPCA(n_components=2, kernel=kernel).fit_transform(X) # keep the first 2 components. default gamma = 1/n_features
                ax = plotComponents2D(X_kpca, y)
                ax.set_title('Kernel PCA (' + kernel + ')')
                plt.show()
                display(HTML('<hr/>'))
            except:
                print('Exception in kernel-PCA: ' + kernel)


        X_tsvd = TruncatedSVD(n_components=2).fit_transform(X)
        ax = plotComponents2D(X_tsvd, y)
        ax.set_title('Truncated SVD')    
        plt.show()
        print('PCA is (truncated) SVD on centered data (by per-feature mean substraction). If the data is already centered, those two classes will do the same. In practice TruncatedSVD is useful on large sparse datasets which cannot be centered easily. ')
        display(HTML('<hr/>'))   

        X_tsne = TSNE(n_components=2).fit_transform(X)
        ax = plotComponents2D(X_tsne, y)
        ax.set_title('t-SNE')
        plt.show()
        print('t-SNE (t-distributed Stochastic Neighbor Embedding) is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high. This will suppress some noise and speed up the computation of pairwise distances between samples.')
        display(HTML('<hr/>'))

        X_mds = MDS(n_components=2).fit_transform(X_scaled)
        ax = plotComponents2D(X_mds, y)
        ax.set_title('MDS')
        plt.show()
        print('MDS (Multidimensional scaling) is a simplification of kernel PCA, and can be extensible with alternate kernels. PCA selects influential dimensions by eigenanalysis of the N data points themselves, while MDS (Multidimensional Scaling) selects influential dimensions by eigenanalysis of the N2 data points of a pairwise distance matrix. This has the effect of highlighting the deviations from uniformity in the distribution. Reference manifold.ipynb')
        display(HTML('<hr/>'))

        Z = dataset_dct_row_wise(X, K = 2, verbose = False)
        ax = plotComponents2D(Z, y)
        ax.set_title('DCT')
        plt.show()
        display(HTML('<hr/>'))
        
        for alg in mf.get_algorithms():
            W,_,_,_ = mf.mf(X_mm_scaled, k = 2, alg = alg, display = False) # some MFDR algs (e.g., NMF) require non-negative X
            ax = plotComponents2D(W, y)
            ax.set_title(alg)
            plt.show()
            display(HTML('<hr/>'))
        
        X_lda = lda(X, y)
        ax = plotComponents2D(X_lda, y)
        ax.set_title('LDA')
        plt.show()
        print('LDA is like PCA, but focuses on maximizing the seperatibility between categories. \
    LDA for two categories tries to maximize distance between group means, meanwhile minimize intra-group variances. \n\
    { (\mu_1 - \mu_2)^2 } \over { s_1^2 + s_2^2 }')
        print('Risk of using LDA: Possible Warning - Variables are collinear. \n\
    LDA, like regression techniques involves computing a matrix inversion, which is inaccurate if the determinant is close to 0 (i.e. two or more variables are almost a linear combination of each other). \n\
    More importantly, it makes the estimated coefficients impossible to interpret. If an increase in X1 , say, is associated with an decrease in X2 and they both increase variable Y, every change in X1 will be compensated by a change in X2 and you will underestimate the effect of X1 on Y. In LDA, you would underestimate the effect of X1 on the classification. If all you care for is the classification per se, and that after training your model on half of the data and testing it on the other half you get 85-95% accuracy I\'d say it is fine. ')
        display(HTML('<hr/>'))

        pls = PLSRegression(n_components=2, scale = True)
        X_pls = pls.fit(X, y).transform(X)
        ax = plotComponents2D(X_pls, y)
        ax.set_title('PLS')
        plt.show()
        print('PLS STEPS: \n\
    X and Y are decomposed into latent structures in an iterative way. \n\
    The latent structure corresponding to the most variation of Y is explained by a best latent strcture of X. \n\n \
    ADVANTAGES: Deal with multi-colinearity; Interpretation by data structure')
        display(HTML('<hr/>'))


    display(HTML('<hr/><h2>特征选择 Feature Selection</h2>'))
    X_names = np.array(X_names)

    X_ch2,idx = chisq_stats_fs(X_mm_scaled, y, X_names)
    ax = plotComponents2D(X_ch2[:,:2], y)
    ax.set_title('FS via Chi Square Stats')
    plt.show()
    print('Important Features:', X_names[idx])
    display(HTML('<hr/>'))

    X_anova,idx = anova_stats_fs(X_mm_scaled, y, X_names)
    ax = plotComponents2D(X_anova[:,:2], y)
    ax.set_title('FS via ANOVA F Stats')
    plt.show()
    print('Important Features:', X_names[idx])
    display(HTML('<hr/>'))

    X_lasso,idx = lasso_fs(X_mm_scaled, y, X_names)
    ax = plotComponents2D(X_lasso[:,:2], y)
    ax.set_title('FS via LASSO')
    plt.show()
    print('Important Features:', X_names[idx])
    display(HTML('<hr/>'))

    X_enet,idx = elastic_net_fs(X_mm_scaled, y, X_names)
    ax = plotComponents2D(X_enet[:,:2], y)
    ax.set_title('FS via Elastic Net')
    plt.show()
    print('Important Features:', X_names[idx])
    display(HTML('<hr/>'))

    if len(set(y)) == 2:
        if cla_feature_num is None:
            cla_feature_num = X_enet.shape[1] # use all selected features

        display(HTML('<hr/><h2>可分性度量 Classifiablity Analysis(top-'+ str(cla_feature_num) +' selected features)</h2>'))
        display(HTML('若运行报错，请尝试安装R运行环境，并执行 install.packages("ECoL") '))
        display(HTML(metrics.get_html(X_enet[:,:cla_feature_num],y)))
    
    display(HTML('<hr/><h2>分类 Classification</h2><h3>超参数优化及模型选择 Hyper-parameter Optimization （SVM）</h3>'))
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-1, 1e-2],'C': [0.01, 0.1, 1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000,10000,100000]}]

    print('Use elastic net selected features after PCA as input: ')
    X_enet_pca = PCA(n_components=2).fit_transform(X_enet)
    best_params, clf, _ = grid_search_svm_hyperparams(X_enet_pca, y, 0.2, tuned_parameters)
    plot_svm_boundary(X_enet_pca, y, clf)


    display(HTML('<hr/><h3>线性分类（逻辑回归模型）Linear Classifier (Logistic Regression)</h3>'))
    
    lr = LogisticRegression(penalty='l2', 
                        tol=0.0001, 
                        C=1.0,                         
                        class_weight=None, 
                        solver='liblinear', # good for small dataset, but requires ovr for multi-class senarios
                        max_iter=100, 
                        multi_class='ovr', # one-vs-rest: a binary problem is fit for each label
                        verbose=0,                        
                        l1_ratio=None).fit(X_enet_pca, y)
    
    plot_lr_boundary(X_enet_pca, y, lr)

def build_simple_pipeline(X, y, save_path = None):
    '''
    Build a simple pipeline: Standard Scaler + LASSO + PCA + Grid Search - SVM.

    Remarks
    -------
    sklearn.pipeline.Pipeline(steps, memory=None):  
    Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods. The final estimator only needs to implement fit. The transformers in the pipeline can be cached using memory argument.
    '''

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-1, 1e-2],'C': [0.01, 0.1, 1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000,10000,100000]}]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', SelectFromModel(LassoCV(cv=5), threshold=1e-4)),
        ('pca', PCA(n_components=2)), 
        ('grid_search', GridSearchCV(SVC(), tuned_parameters, cv=5)) ]) 

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipe.fit(X_train, y_train) 
    print('Test accuracy: %.3f' % pipe.score(X_test, y_test))

    # We also build a feature-extraction pipeline, 
    fe_pipe = Pipeline([
        ('scaler', pipe.named_steps['scaler']),
        ('lasso', pipe.named_steps['lasso']),
        ('pca', pipe.named_steps['pca'])
    ])
    Xfe = fe_pipe.transform(X)

    # save the pipeline models to specified path
    if save_path is not None and save_path != '':

        d = {}
        d['pipeline'] = pipe
        d['pipeline_fe'] = fe_pipe
        d['X'] = X
        d['y'] = y
        d['Xfe'] = Xfe
        joblib.dump(dict, save_path)

        ##### To load back ##########
        # dict_r = joblib.load('your.pkl')
        # dict_r['pipeline'].predict(X_test)

    return pipe