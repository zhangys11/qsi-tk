import os.path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC # SVC(C=1.0, ...) 与 nuSVC(nu = 0.5, ...) 的区别：前者使用C～[0，inf），后者使用nu～(0,1]，惩罚性的解释不同，优化都可以使用GridSearch方法
from scipy.signal import savgol_filter

from qsi.io.pre import filter_dataset
from . import io
from .vis import supervised_dimension_reductions, unsupervised_dimension_reductions
from .fs import RUN_ALL_FS
from cla import metrics

def analyze(id, x_range = None, y_subset=None, shift = 100, pres = None, fs_output = '', fs_feature_num = 30, cla_feature_num = 10):
    '''
    A complete data analysis flow. 
    1. Load dataset 2. Preprocess 3. General dataset analysis
    
    Parameters
    ----------
    id : dataset id or a full path
    x_range, y_subset : select specific rows and cols for analysis, if you don't wish to use the entire dataset.
    shift : interval between classes in the average wave plot
    
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
        
        default is [] or None. 

    Remarks
    -------
    Default preprocessing parameters for MALDI-TOF data:
    [('threshold', 10), ('peak_normalize', 1000), ('max', 100)]
    '''
    display(HTML('<h2>数据加载 Load Dataset</h2>'))

    if (id in io.DATASET_MAP.keys()):
        X, y, X_names, _, _ = io.load_dataset(id, x_range=x_range, y_subset=y_subset, shift = shift)
    elif os.path.exists(id):
        X, y, X_names, _ = io.open_dataset(id, x_range=x_range, y_subset=y_subset, shift = shift)
    else:
        display(HTML('<h3>数据加载失败，请传入正确的id或文件路径。<br/>Load data failed. Please specific a correct dataset ID or file path.</h3>'))

    display(HTML('<hr/><h2>预处理 Preprocessing </h2>'))
    
    if pres is None:
        pres = []

    for pre in pres:
        pre_type, pre_param = pre        
        display(HTML('<h3>' + pre_type + '</h3>'))

        if pre_type == 'threshold':
            X = io.pre.x_thresholding(X, pre_param)
        elif pre_type in ['max', 'sum', 'rect', 'tri', 'mean']:
            X, X_names = io.pre.x_binning(X, X_names,target_dim=pre_param,flavor=pre_type)
        elif pre_type == 'diff':
            X = io.pre.diff_dataset(X)
            X_names = X_names[1:] # pop the 1st x label, as diff reduces dim by 1
        elif pre_type == 'rowvec_normalize':
            X = io.pre.x_normalize(X, flavor='rowvec', target_max=pre_param)
        elif pre_type == 'peak_normalize':
            X = io.pre.x_normalize(X, flavor='peak', target_max=pre_param)

    display(HTML('预处理后的维度：X.shape = ' + str(X.shape) +'<hr/>'))

    analyze_dataset(X, y, X_names, fs_output, fs_feature_num, cla_feature_num)

def analyze_dataset(X, y, X_names, fs_output = '', fs_feature_num = 30, cla_feature_num = 10):
    '''
    This is a general pipeline for data analysis. 
    ordinary feature scaling + feature selection + classifiability analysis + classification 
    
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
    print('X_mm_scaled is rescaled to [0,1]. We use X_mm_scaled in DR and FS.')

    display(HTML('<hr/><h2>降维 Dimensionality Reduction</h2>'))
    display(HTML('<h3>无监督降维 Unsupervised Dimensionality Reduction</h3>'))
    unsupervised_dimension_reductions(X_mm_scaled, X_names, y)
    display(HTML('<h3>有监督降维 Supervised Dimensionality Reduction</h3>'))
    unsupervised_dimension_reductions(X_mm_scaled, X_names, y)

    display(HTML('<hr/><h2>特征选择 Feature Selection</h2>'))
    
    FS_OUTPUT, _, FS_COMMON_IDX = RUN_ALL_FS(X_mm_scaled, y, X_names, N = fs_feature_num, output='all')
    if fs_output == 'common' or fs_output == '':
        if len(FS_COMMON_IDX) > 0:
            display(HTML('<h3>' + 'Use common features selected by all FS methods: ' + str(X_names(FS_COMMON_IDX)) + '</h3>'))
            X_s = X_mm_scaled[FS_COMMON_IDX]        
        else:
            display(HTML('<h3>' + 'No common features. We will use the default elastic net fs for the following procedures.' + '</h3>'))
            X_s = FS_OUTPUT['elastic net']
    else:
        display(HTML('<h3>' + 'Use ' + fs_output + ' for the following procedures.</h3>'))
        X_s = FS_OUTPUT[fs_output]

    if len(set(y)) == 2:
        if cla_feature_num is None:
            cla_feature_num = X_s.shape[1] # use all selected features

        display(HTML('<hr/><h2>可分性度量 Classifiablity Analysis(top-'+ str(cla_feature_num) +' selected features)</h2>'))
        display(HTML('若运行报错，请尝试安装R运行环境，并执行 install.packages("ECoL") '))
        display(HTML(metrics.get_html(X_s[:,:cla_feature_num],y)))
    
    display(HTML('<hr/><h2>分类 Classification</h2><h3>超参数优化及模型选择 Hyper-parameter Optimization （SVM）</h3>'))
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-1, 1e-2],'C': [0.01, 0.1, 1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000,10000,100000]}]

    print('Use elastic net selected features after PCA as input: ')
    X_s_pca = PCA(n_components=2).fit_transform(X_s)
    best_params, clf, _ = metrics.grid_search_svm_hyperparams(X_s_pca, y, 0.2, tuned_parameters)
    metrics.plot_svm_boundary(X_s_pca, y, clf)


    display(HTML('<hr/><h3>线性分类（逻辑回归模型）Linear Classifier (Logistic Regression)</h3>'))
    
    lr = LogisticRegression(penalty='l2', 
                        tol=0.0001, 
                        C=1.0,                         
                        class_weight=None, 
                        solver='liblinear', # good for small dataset, but requires ovr for multi-class senarios
                        max_iter=100, 
                        multi_class='ovr', # one-vs-rest: a binary problem is fit for each label
                        verbose=0,                        
                        l1_ratio=None).fit(X_s_pca, y)
    
    metrics.plot_lr_boundary(X_s_pca, y, lr)

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