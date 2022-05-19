import matplotlib.pyplot as plt
import numpy as np

def plotComponents1D(X, y, labels, use_markers=False, ax=None, legends = None):
    
    if X is None or X.shape[1] < 1:
        print('ERROR: X Has no FEATURE/COLUMN! SKIPPING plotComponents1D().')
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