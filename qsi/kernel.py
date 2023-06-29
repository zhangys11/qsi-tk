import pickle
import ackl
import ackl.metrics
assert( '__version__' in ackl.__dict__ and ackl.__version__ > '1.0.5')

import numpy as np
from IPython.core.display import HTML, display
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from . import io, pipeline

def classify_dataset_with_kernels(file_path):
    '''
    Analyze a classification dataset with various kernels.
    '''
    
    display(HTML('<h1>' + file_path + '</h1><hr/><br/>'))

    X, y, X_names, labels = io.open_dataset(file_path)
    n_classes = len(labels)
    if len(np.unique(y)) < 2 or n_classes < 2:
        print('Error: less than 2 classes in the dataset. This function only supports classification tasks.')
        return

    X, X_names = pipeline.preprocess_dataset(X, X_names, pres = [('baseline_removal', (1e7, 1e-2))]) # ('max', 0.2)
    io.draw_class_average(X, y, X_names, labels=labels, SD=1, shift = (X.max() - X.min()) / n_classes * 2,
                          figsize = (round(n_classes * 1.7) + 3, round(n_classes * .6) + 2 ))
    _ = io.scatter_plot(X, y, labels=labels, 
                        figsize = (round(n_classes * .4) + 3, round(n_classes * .25) + 2 ))
    
    old_dim = X.shape
    X = PCA(n_components = 0.99).fit_transform(X) # DR to 99% PCA components
    new_dim = X.shape
        
    X = MinMaxScaler().fit_transform(X)
    pre_desc = 'sklearn.preprocessing.MinMaxScaler'

    display(HTML('<h2>1. Preprocessing</h2><hr/><p>' + '1.1 PCA: ' + str(old_dim) + ' ==> ' + str(new_dim) + '</p><p>1.2 Feature scaling: ' + pre_desc + '</p><br/>'))

    # display(HTML('<h2>2. CLA before kernel' + '</h2><hr/><br/>'))        
    # display(HTML('<p>' + str(ms) + '</p>'))

    display(HTML('<h2>2. Try Kernels' + '</h2><hr/><br/>'))

    pkl = ackl.__version__ + '.pkl'
    if True: # not os.path.isfile(pkl): # if pkl already exist, just reload it

        dics, _ = ackl.metrics.preview_kernels(X, y, calc_metrics = True, embed_title = False, 
                                               scale = False, logplot = False, output_html = False, verbose = False)

        # Persist to a local pickle binary file.
        with open(pkl,'wb') as f:
            pickle.dump(dics, f)

    # Load back
    with open (pkl, 'rb') as f:
        dics = pickle.load(f)

    display(HTML('<h2>3. Evaluate Metrics' + '</h2><hr/><br/>'))
    html_str = ackl.metrics.visualize_metric_dicts(dics, plot = True)
    display(HTML( html_str ))

    display(HTML('<h2>4. Run time' + '</h2><hr/><br/>'))
    dic = ackl.metrics.time_cost_kernels(X, repeat=10)
    print('Averaged run time: ', dic)
        
    display(HTML('<h3>----------- End of Analysis ----------</h3><br/><br/>'))