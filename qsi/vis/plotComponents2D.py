import matplotlib.pyplot as plt
import numpy as np

def plotComponents2D(X, y, labels = None, use_markers = False, ax=None, legends = None, tags = None):
    
    if X.shape[1] < 2:
        print('ERROR: X MUST HAVE AT LEAST 2 FEATURES/COLUMNS! SKIPPING plotComponents2D().')
        return
    
    # Gray shades can be given as a string encoding a float in the 0-1 range
    colors = ['0.9', '0.1', 'red', 'blue', 'black','orange','green','cyan','purple','gray']
    markers = ['o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H', 'o', 's', '^', 'D', 'H']

    if (ax is None):
        fig, ax = plt.subplots()
        
    i=0
    if (labels is None):
        labels = set(y)
        
    for label in labels:
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