import os.path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import IPython.display

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.signal import savgol_filter

from cla import metrics
from . import io
from .vis import supervised_dimension_reductions, unsupervised_dimension_reductions
from .fs import RUN_ALL_FS

def analyze(id, x_range = None, y_subset=None, shift = None, pres = None, fs_output = '', fs_feature_num = 30, cla_feature_num = 3):
    '''
    A standard data analysis flow. 
    1. Load dataset 2. Preprocess 3. General dataset analysis

    Don't use this funciton if you want to do some extra preprocessing or feature selection.
    Instead, use the individual functions, i.e., preprocess_dataset and analyze_dataset.
    
    Parameters
    ----------
    id : dataset id or a full path
    x_range, y_subset : select specific rows and cols for analysis, if you don't wish to use the entire dataset.
    shift : interval between classes in the average wave plot
    '''

    IPython.display.display(IPython.display.HTML('<h2>数据加载 Load Dataset</h2>'))

    if (id in io.DATASET_MAP.keys()):
        X, y, X_names, _, _ = io.load_dataset(id, x_range=x_range, y_subset=y_subset, shift = shift)
    elif os.path.exists(id):
        X, y, X_names, _ = io.open_dataset(id, x_range=x_range, y_subset=y_subset)
    else:
        IPython.display.display(IPython.display.HTML('<h3>数据加载失败，请传入正确的id或文件路径。<br/>Load data failed. Please specific a correct dataset ID or file path.</h3>'))
        return

    X, X_names = preprocess_dataset(X, X_names, pres)
    analyze_dataset(X, y, X_names, fs_output, fs_feature_num, cla_feature_num)

def preprocess_dataset(X, X_names, pres = None):
    '''
    Load and preprocess a dataset.

    Parameters
    ----------
    
    pres : a list of tuple (pre_type, pre_param) : additional preprocessing (with params) applied, e.g. ['max_bin, 100', 'threshold', 10]
        
        pre = 'max(target_dim = pre_param)' : binning and take the maximum in each interval
        pre = 'sum(target_dim = pre_param)' or 'rect(window_size = pre_param)' : 
            binning and take the sum/mean in each interval, i.e., rectangle window
        pre = 'tri(target_dim = pre_param)' : triangle window kernel.

        pre = 'rbf(sigma = pre_param)' : applied a radial basis function kernel. Not implemented yet.
        pre = 'z(sigma = pre_param)' : applied z (normal distribution PDF) kernel. Not implemented yet.
        pre = 'trapezoidal(pre_param)' : trapezoid window kernel. Not implementated yet.
        
        pre = 'diff' : first-order op. We use io.pre.diff_dataset()
        pre = 'threshold' : call io.pre.x_thresholding()
        pre = 'peak_normalize' / 'rowvec_normalize' : call io.pre.x_normalize()
        pre = 'baseline_removal' : call io.pre.x_baseline_removal()
        
        default is [] or None. 

    Remarks
    -------
    Default preprocessing parameters for MALDI-TOF data:
    [('baseline_removal', (1e8, 1e-3)), ('threshold', 10), ('max', 0.01), ('peak_normalize', 1000)]
    '''
    IPython.display.display(IPython.display.HTML('<hr/><h3>每个样本的预处理 Row-wise Preprocessing </h3>'))
    
    if pres is None:
        pres = []

    for _, pre in enumerate(pres):
        pre_type, pre_param = pre
        IPython.display.display(IPython.display.HTML('<h3>' + pre_type + '</h3>'))

        if pre_type == 'baseline_removal':
            X = io.pre.x_baseline_removal(X, lam = pre_param[0], p = pre_param[1])  # use default lambda = 1e8
            print('消除基线飘移: baseline removal (regularization = ' + str(pre_param[0]) + ', residual penalty asymetry = ' + str(pre_param[1]) + ')')
        elif pre_type == 'threshold':
            X = io.pre.x_thresholding(X, pre_param)
            print('阈值预处理（消除背景噪声、满足非负输入条件等）: threshold = ' + str(pre_param))
        elif pre_type in ['max', 'sum', 'rect', 'tri', 'mean']:
            X, X_names = io.pre.x_binning(X, X_names,target_dim=pre_param,flavor=pre_type)
            print('窗函数预处理（' + pre_type +'）: binning window width =  1 / ' + str(pre_param) + ' = ' + str(round(1.0 / pre_param)) )
        elif pre_type == 'diff':
            X = io.pre.diff_dataset(X)
            X_names = X_names[1:] # pop the 1st x label, as diff reduces dim by 1
            print('差分预处理（一阶差分）: first-order difference')
        elif pre_type == 'rowvec_normalize':
            X = io.pre.x_normalize(X, flavor='rowvec', target_max=pre_param)
            print('行向量归一化预处理: row vector normalization (target_max = ' + str(pre_param) + ')')
        elif pre_type == 'peak_normalize':
            X = io.pre.x_normalize(X, flavor='peak', target_max=pre_param)
            print('峰值归一化预处理: peak normalization (target_max = ' + str(pre_param) + ')')

    IPython.display.display(IPython.display.HTML('<br/><br/>预处理后的维度：X.shape = ' + str(X.shape) +'<hr/>'))

    return X, X_names

def analyze_dataset(X, y, X_names, fs_output = '', fs_feature_num = 30, cla_feature_num = 10):
    '''
    This is a general pipeline for data analysis.  
    ordinary feature scaling + feature selection + classifiability analysis + classification 
    NOTE: y must be a vector of 0/1 labels, i.e., only binary classification is supported.

    Parameters
    ----------
    X : an already-preprocessed data matrix
    fs_output : which feature selection result to use for the final classification. 
        Available values: (multi-task fs are not provided here)
        "pearsion-r", 
        "info-gain / mutual information", 
        "chi-squared statistic",
        "anova statistic", 
        "lasso", 
        "elastic net",
        "adaptive lasso",
        "adaptive elastic net"
        Default is '' (empty string) or 'common', which means to use the common intersection of all feature selections.
        If no common features are found, then use elastic net. 
    cla_feature_num : how many features to use for classifiability analysis. 
        0 will disable classifiability analysis.
    '''

    if X_names is None or X_names == []:
        X_names = list(range(X.shape[1]))
    
    IPython.display.display(IPython.display.HTML('<h3>Savitzky-Golay Filter 滤波</h3>'))
    IPython.display.display(IPython.display.HTML('<h4>此处仅做额外的预处理效果预览，并不叠加到后续分析中，需要用户自行判断使用。</br>This preprocessing effect is just for preview. Users should choose the algorithms on demand. </h4>'))
    
    # Savgol filtering
    np.set_printoptions(precision=2)
    X_sg = savgol_filter(X, window_length = 9, polyorder = 3, axis = 1)  # axis = 0 for vertically; axis = 1 for horizontally
    plt.figure(figsize = (40,10))
    plt.scatter(X_names, np.mean(X_sg,axis=0).tolist()) 
    plt.title(u'Averaged Spectrum After Savitzky-Golay Filtering', fontsize=30)
    plt.show()

    IPython.display.display(IPython.display.HTML('<h3>高通滤波|消除基线漂移 Highpass Filter | Baseline Drift Removal</h3>'))
    IPython.display.display(IPython.display.HTML('<h4>此处仅做额外的预处理效果预览，并不叠加到后续分析中，需要用户自行判断使用。</br>This preprocessing effect is just for preview. Users should choose the algorithms on demand. </h4>'))

    # Butterworth filter
    X_f = io.pre.filter_dataset(X, nlc = 0.002, nhc = None)  # axis = 0 for vertically; axis = 1 for horizontally
    plt.figure(figsize = (40,10))
    plt.scatter(X_names, np.mean(X_f,axis=0).tolist()) 
    plt.title('Averaged Spectrum After Butterworth Filter\nIf the filtered result is not good, you will need to finetune the Butterworth highpass filter cutoff freq.', fontsize=30)
    plt.show()


    IPython.display.display(IPython.display.HTML('<hr/><h3>特征缩放 Feature Scaling</h3>'))
    # normalization

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # scaling to [0,1]

    mm_scaler = MinMaxScaler()
    X_mm_scaled = mm_scaler.fit_transform(X)
    print('X_mm_scaled is rescaled to [0,1]. We use X_mm_scaled in DR and FS.')

    IPython.display.display(IPython.display.HTML('<hr/><h2>尝试各类常用分类器(支持多分类)</h2>'))
    _ = metrics.run_multiclass_clfs(X, y)

    IPython.display.display(IPython.display.HTML('<hr/><h2>降维 Dimensionality Reduction</h2>'))

    IPython.display.display(IPython.display.HTML('<h3>无监督降维 Unsupervised Dimensionality Reduction</h3>'))
    unsupervised_dimension_reductions(X_mm_scaled, y)
    print('PCA is (truncated) SVD on centered data (by per-feature mean substraction). If the data is already centered, those two classes will do the same. In practice TruncatedSVD is useful on large sparse datasets which cannot be centered easily.')
    print('MDS (Multidimensional scaling) is a simplification of kernel PCA, and can be extensible with alternate kernels. PCA selects influential dimensions by eigenanalysis of the N data points themselves, while MDS (Multidimensional Scaling) selects influential dimensions by eigenanalysis of the N2 data points of a pairwise distance matrix. This has the effect of highlighting the deviations from uniformity in the distribution. Reference manifold.ipynb')

    IPython.display.display(IPython.display.HTML('<h3>有监督降维 Supervised Dimensionality Reduction</h3>'))
    supervised_dimension_reductions(X_mm_scaled, y)
    print('LDA is like PCA, but focuses on maximizing the seperatibility between categories. \
LDA for two categories tries to maximize distance between group means, meanwhile minimize intra-group variances. \n\
{ (\mu_1 - \mu_2)^2 } \over { s_1^2 + s_2^2 }')
    print('The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix.\nThe fitted model can be used to reduce the dimensionality of the input by projecting it to the most discriminative directions.')
    print('Risk of using LDA: Possible Warning - Variables are collinear. \n\
LDA, like regression techniques involves computing a matrix inversion, which is inaccurate if the determinant is close to 0 (i.e. two or more variables are almost a linear combination of each other). \n\
More importantly, it makes the estimated coefficients impossible to interpret. If an increase in X1 , say, is associated with an decrease in X2 and they both increase variable Y, every change in X1 will be compensated by a change in X2 and you will underestimate the effect of X1 on Y. In LDA, you would underestimate the effect of X1 on the classification. If all you care for is the classification per se, and that after training your model on half of the data and testing it on the other half you get 85-95% accuracy I\'d say it is fine. ')
    
    print('\nPLS: X and Y are decomposed into latent structures in an iterative way. The latent structure corresponding to the most variation of Y is explained by a best latent strcture of X. \nADVANTAGES: Deal with multi-colinearity; Interpretation by data structure')
    
    IPython.display.display(IPython.display.HTML('<hr/><h2>特征选择 Feature Selection</h2>'))
    IPython.display.display(IPython.display.HTML('<p>Reasons for using feature selection:<br/>Improve data processing efficiency; reduce memory usage.<br/>Reducing the number of features, to reduce overfitting and improve the generalization of models.<br/>To gain a better understanding of the features and their relationship to the response variables.</p>'))

    FS_OUTPUT, _, FS_COMMON_IDX = RUN_ALL_FS(X_mm_scaled, y, X_names, N = fs_feature_num, output='all')
    if len(FS_COMMON_IDX) > 0:
        IPython.display.display(IPython.display.HTML('<h3>' + 'Common features selected by all FS methods: ' + str(np.array(X_names)[FS_COMMON_IDX]) + '</h3>'))
    
    if fs_output == 'common' or fs_output == '':
        if len(FS_COMMON_IDX) > 1: # we require at least 2 common features
            # print(X_mm_scaled.shape, FS_COMMON_IDX)
            X_s = X_mm_scaled[:,FS_COMMON_IDX]
        else:
            IPython.display.display(IPython.display.HTML('<h3>' + 'Too few common features. We will use the default elastic net fs for the following procedures.' + '</h3>'))
            X_s = FS_OUTPUT['elastic net']
    else:
        IPython.display.display(IPython.display.HTML('<h3>' + 'Use ' + fs_output + ' for the following procedures.</h3>'))
        X_s = FS_OUTPUT[fs_output]

    if cla_feature_num > 0 and len(set(y)) == 2:
        if cla_feature_num > X_s.shape[1]:
            cla_feature_num = X_s.shape[1] # use all selected features

        IPython.display.display(IPython.display.HTML('<hr/><h2>可分性度量 Classifiablity Analysis(top-'+ str(cla_feature_num) +' selected features)</h2>'))
        IPython.display.display(IPython.display.HTML('若运行报错，请尝试安装R运行环境，并执行 install.packages("ECoL") '))
        IPython.display.display(IPython.display.HTML(metrics.get_html(X_s[:,:cla_feature_num],y)))
    
    IPython.display.display(IPython.display.HTML('<hr/><h2>对筛选后的特征进行分类 Classification on Selected Features</h2><h3>超参数优化及模型选择 Hyper-parameter Optimization （SVM）</h3>'))
    
    _ = metrics.run_multiclass_clfs(X_s, y)

    if len(set(y)) == 2: # show decision boundary for binary classification

        IPython.display.display(IPython.display.HTML('<hr/><h3>支持向量机 SVC</h3>'))

        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-1, 1e-2],'C': [0.001, 0.01, 0.1, 1, 10]},
                            {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10]}]

        # X_s_pca = PCA(n_components=2).fit_transform(X_s)
        _, clf, _ = metrics.grid_search_svm_hyperparams(X_s, y, 0.2, tuned_parameters, verbose = False)
        metrics.plot_svm_boundary(X_s, y, clf)
        print('ACC = ', clf.score(X_s, y))


        IPython.display.display(IPython.display.HTML('<hr/><h3>逻辑回归 (Logistic Regression)</h3>'))
        
        lr = LogisticRegressionCV(max_iter=1000,
                            multi_class='multinomial').fit(X_s, y)
        
        metrics.plot_lr_boundary(X_s, y, lr)
        print('ACC = ', clf.score(X_s, y))

def build_simple_pipeline(X, y, save_path = None):
    '''
    Build a simple pipeline: Standard Scaler + LASSO + PCA + Grid Search - SVM.
    
    Remarks
    -------
    sklearn.pipeline.Pipeline(steps, memory=None):  
    Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods. The final estimator only needs to implement fit. The transformers in the pipeline can be cached using memory argument.
    
    SVC(C=1.0, ...) 与 nuSVC(nu = 0.5, ...) 的区别：前者使用C～[0，inf），后者使用nu～(0,1]，惩罚性的解释不同，优化都可以使用GridSearch方法
    '''

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-1, 1e-2],'C': [0.01, 0.1, 1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000,10000,100000]}]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', SelectFromModel(LassoCV(cv=5), threshold=1e-4)),
        ('pca', PCA(n_components=2)), 
        ('grid_search', GridSearchCV(SVC(), tuned_parameters, cv=5)) ]) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
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
