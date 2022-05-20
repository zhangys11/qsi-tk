# Use %matplotlib notebook instead of %matplotlib inline to get embedded interactive figures in the IPython notebook.  
# This requires recent versions of matplotlib (1.4+) and IPython (3.0+)
# Need to restart notebook or kernel and call %matplotlib notebook twice

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# column count of X should >= 3
def plotComponents3D(X, y, labels):

    if X.shape[1] < 3:
        print('ERROR: X MUST HAVE AT LEAST 3 FEATURES/COLUMNS! SKIPPING plotComponents3D().')
        return

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')

    colors = ['gray', 'red', 'blue', 'black','orange','green','cyan','purple']
    
    for idx, l in enumerate(labels):
        indices = np.where(y == l)[0]
        ax.plot(X[indices, 0],
                X[indices, 1],
                X[indices, 2],
                'o', 
                label= "C" + str(l), 
                c=colors[idx])

    ax.legend()