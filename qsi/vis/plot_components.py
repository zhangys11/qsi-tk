# Use %matplotlib notebook instead of %matplotlib inline to get embedded interactive figures in the IPython notebook.  
# This requires recent versions of matplotlib (1.4+) and IPython (3.0+)
# Need to restart notebook or kernel and call %matplotlib notebook twice
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_components_1d(X, y, labels, use_markers=False, ax=None, legends = None):
    '''
    Plot 1D data points in a 2D space.
    '''

    if X is None or X.shape[1] < 1:
        print('ERROR: X Has no FEATURE/COLUMN! SKIPPING plot_components_1d().')
        return
    
    colors = ['0.8', '0.1', 'red', 'blue', 'black','orange','green','cyan','purple','gray']
    markers = ['o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H']

    if (ax is None):
        fig, ax = plt.subplots()
        
    i=0
    for label in labels:
        cluster = X[np.where(y == label)]
        # print(cluster.shape)

        if use_markers:
              ax.scatter([cluster[:,0]], np.zeros_like([cluster[:,0]]), 
              s=40, marker=markers[i], 
              acecolors='none', 
              edgecolors=colors[i], 
              label= (str(legends[i]) if legends is not None else ("Y = " + str(label) + ' (' + str(len(cluster)) + ')')),
              )
        else: 
              ax.scatter([cluster[:,0]], np.zeros_like([cluster[:,0]]), 
              s=70, facecolors=colors[i],   
              label= (str(legends[i]) if legends is not None else ("Y = " + str(label) + ' (' + str(len(cluster)) + ')')),
              alpha = .4)
        i=i+1

    ax.legend()


def plot_components_2d(X, y, labels = None, use_markers = False, ax=None, legends = None, tags = None):
    '''
    Plot 2D scatter plot of dataset X with labels y.
    '''

    if X.shape[1] < 2:
        print('ERROR: X MUST HAVE AT LEAST 2 FEATURES/COLUMNS! SKIPPING plot_components_2d().')
        return
    
    # Gray shades can be given as a string encoding a float in the 0-1 range
    colors = ['0.9', '0.1', 'red', 'blue', 'green','black','orange','cyan','purple','gray']
    markers = ['o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H']

    if (ax is None):
        fig, ax = plt.subplots()
        
    if (y is None or len(y) == 0):
        labels = [0] # only one class
    if (labels is None):
        labels = set(y)

    i=0        

    for label in labels:
        if y is None or len(y) == 0:
            cluster = X
        else:
            cluster = X[np.where(y == label)]
        # print(cluster.shape)

        if use_markers:
            ax.scatter([cluster[:,0]], [cluster[:,1]], 
                       s=40, 
                       marker=markers[i], 
                       facecolors='none', 
                       edgecolors=colors[i+3],
                       label= (str(legends[i]) if legends is not None else ("Y = " + str(label)  + ' (' + str(len(cluster)) + ')')) )
        else:
            ax.scatter([cluster[:,0]], [cluster[:,1]], 
                       s=70, 
                       facecolors=colors[i],  
                       label= (str(legends[i]) if legends is not None else ("Y = " + str(label) + ' (' + str(len(cluster)) + ')')), 
                       edgecolors = 'black', 
                       alpha = .4) # cmap='tab20'                
        i=i+1
    
    if (tags is not None):
        for j,tag in enumerate(tags):
            ax.annotate(str(tag), (X[j,0] + 0.1, X[j,1] - 0.1))
        
    ax.legend()

    ax.axes.xaxis.set_visible(False) 
    ax.axes.yaxis.set_visible(False)
    
    return ax

def plot_components_3d(X, y, labels=None, legends = None):
    '''
    Plot 3D scatter plot with different colors for different classes.
    '''

    if X.shape[1] < 3:
        print('ERROR: X MUST HAVE AT LEAST 3 FEATURES/COLUMNS! SKIPPING plot_components_3d().')
        return

    if labels is None:
        labels = np.unique(y)

    if legends is None:
        legends = map(lambda x: 'C'+str(x), labels)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')

    colors = ['0.7', '0.1', 'red',  'blue','green', 'black','orange','cyan','purple']

    for idx, l in enumerate(zip(labels, legends)):
        indices = np.where(y == l[0])[0]
        ax.plot(X[indices, 0],
                X[indices, 1],
                X[indices, 2],
                'o',
                alpha = 0.5,
                label= str(l[1]),
                c=colors[idx])

    ax.legend()
