import ackl
import ackl.metrics
assert( '__version__' in ackl.__dict__ and ackl.__version__ >= '1.1.0')

import numpy as np
from IPython.core.display import HTML, display
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from . import io, pipeline

def classify_dataset_with_kernels(file_path, rowwise_pre = [('baseline_removal', (1e7, 1e-2)), ('threshold', 0)], 
                                  pca_ratio = 1.0, scale = False, clfs = ['LinearDiscriminantAnalysis()'],
                                  multi_kernels = [1], multi_kernel_topk = -1,
                                  plots = True, do_cla = False):
    '''
    Analyze a classification dataset with various kernels.

    Parameters
    ----------
    file_path : str
        The path to the dataset file.
    pca_ratio : float
        The ratio of PCA components to keep. Can use 0.99, 0.95, 0.9, 0.85, etc. Default is 1.0 (100%).        
    scale : bool
        Whether to a min-max scaling. If your dataset is non-negative, use False, otherwise use True as some kernels require non-negative input.
    clfs : list of str
        A list of classifiers to try. Default is ['LinearDiscriminantAnalysis()']. Pass 'all' for all available classifiers.
    multi_kernels : lists of int
        Multi-kernel ranks to try. Default is [1].
    multi_kernel_topk : int
        The number of top kernels to use in multi-kernel cases. Only used when multi_kernel > 1. -1 means to use all single kernels for combination. 
    do_cla : bool
        Whether to do classifiability analysis (using the cla package) for each kernel. Default is False.
    '''
    
    display(HTML('<h2>1. 数据加载及预处理</h2><br/>' + file_path + '<br/>'))

    X, y, X_names, labels = io.open_dataset(file_path)
    n_classes = len(labels)
    if len(np.unique(y)) < 2 or n_classes < 2:
        print('Error: less than 2 classes in the dataset. This function only supports classification tasks.')
        return

    X, X_names = pipeline.preprocess_dataset(X, X_names, pres = rowwise_pre)
    
    if plots:
        io.draw_class_average(X, y, X_names, labels=labels, SD=1, shift = (X.max() - X.min()) / n_classes * 2,
                            figsize = (round(n_classes * 1.7) + 3, round(n_classes * .6) + 2 ))
        _ = io.scatter_plot(X, y, labels=labels,
                            figsize = (round(n_classes * .4) + 3, round(n_classes * .25) + 2 ))
        
    pca_desc = ''
    if pca_ratio < 1.0:
        old_dim = X.shape
        X = PCA(n_components = pca_ratio).fit_transform(X) # DR to 99% PCA components
        new_dim = X.shape
        pca_desc = '<p>PCA (keep ' + str(pca_ratio*100) +  '% info) dim change: ' + str(old_dim) + ' -> ' + str(new_dim) + '</p>'
        display(HTML('<h2>1. Preprocessing</h2><hr/>' + pca_desc))

    # if scale:
    #    X = MinMaxScaler().fit_transform(X)
    #    pre_desc = 'sklearn.preprocessing.MinMaxScaler'
    #    display(HTML('<h2>1. Preprocessing</h2><hr/><p>Feature scaling: ' + pre_desc + '</p>' + pca_desc))

    # display(HTML('<h2>2. CLA before kernel' + '</h2><hr/><br/>'))        
    # display(HTML('<p>' + str(ms) + '</p>'))

    display(HTML('<h2>2. Try Kernels' + '</h2><hr/><br/>'))

    KX, dic_test_accs, all_dic_metrics, _ = ackl.metrics.classify_with_kernels(X, y, 
                                            clfs = clfs, embed_title = False, 
                                            do_cla = do_cla, 
                                            multi_kernels = multi_kernels, 
                                            multi_kernel_topk = multi_kernel_topk,
                                            scale = scale, plots = plots, logplot = False, 
                                            output_html = False, verbose = False)

    if do_cla:

        display(HTML('<h2>3. Evaluate Metrics' + '</h2><hr/><br/>'))
        html_str = ackl.metrics.visualize_metric_dicts(all_dic_metrics, plot = True)
        display(HTML( html_str ))

        display(HTML('<h2>4. Run time' + '</h2><hr/><br/>'))
        dic_time = ackl.metrics.time_cost_kernels(X, repeat=10)
        print('Averaged run time: ', dic_time)

    display(HTML('<h3>Summary</h3>'))
    
    ackl.metrics.visualize_kernel_result_dict(dic_test_accs)

    display(HTML('<h3>----------- End of Analysis ----------</h3><br/><br/>'))

    return KX, dic_test_accs