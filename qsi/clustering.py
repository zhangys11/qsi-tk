from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def run_clustering(X, K = None, y = None, X_names = None, show = True):
    '''
    Parameters
    ----------
    y - optional. true labels.
    '''

    if K is None:
        if y is None:
            print('y is none, set K = 2')
            K = 2
        else:
            K = len(set(y)) #  K - num of clusters

    if X_names is None:
        X_names = list(range(X.shape[1]))

    for model in [KMeans(n_clusters = K), 
                  MiniBatchKMeans(n_clusters = K), 
                  AffinityPropagation(),
                  GaussianMixture(n_components=K),
                  SpectralClustering(),
                  AgglomerativeClustering(n_clusters = K, linkage='ward'),
                  # DBSCAN() # no centroids
                 ]:
        clusters = model.fit_predict(X)       

        print('clustering result: ', clusters)
        if y is not None:
            print('true label: ', y)
        
        plt.figure(figsize = (12,4))
        if 'cluster_centers_' in model.__dict__:
            for i in range(len(model.cluster_centers_)):
                plt.plot(X_names, model.cluster_centers_[i], label = 'cluster center ' + str(i+1))
        elif 'means_' in model.__dict__:
            for i in range(len(model.means_)):
                plt.plot(X_names, model.means_[i], label = 'cluster center ' + str(i+1)) # model.covariances_
        elif 'affinity_matrix_' in model.__dict__:
            plt.matshow(model.affinity_matrix_, label='affinity matrix')
        elif isinstance(model, AgglomerativeClustering):
            # HC
            plt.figure(figsize = (10, 10))
            l = linkage(X, 
                               metric='euclidean',
                               method='complete')
            plt.title('Hierarchical Clustering Dendrogram')
            dendr = dendrogram(l, labels = y)
            
        plt.title(str(model))
        plt.legend()
        plt.show()

# run_clustering(X, y)