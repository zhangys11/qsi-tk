import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
import itertools
from tqdm import tqdm
from IPython.core.display import display, HTML

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def draw_regression_plots(yt, yp, title, order=False):
    '''
    yt : ground truth
    yp : prediction
    order : if 'asc' or 'desc', will sort points by yt.
    '''

    if order != False:
        idx = sorted(range(len(yt)), key=lambda k: yt[k]) # 'asc'
        if order == 'desc':
            idx = idx[::-1]
        yp = np.array(yp)[idx] # order yp by yt
        yt = np.array(yt)[idx]
    else:
        yp = np.array(yp)
        yt = np.array(yt)
        
    # scatter plot
    plt.figure()
    plt.title(title + '\n' + 'R2 =' + str(round(r2_score(yt, yp),3)) + ', MSE =' + str(round(mean_squared_error(yt, yp),3)))
    plt.scatter(range(len(yt)), yt, label = 'ground truth')
    plt.scatter(range(len(yp)), yp, label='prediction')
    plt.legend()
    plt.show()

    # diagonal scatter plot 
    plt.figure(figsize = (6,6))
    plt.scatter(yt, yp)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()
    
    # residual plot
    residuals = yp-yt
    plt.figure()
    # print(yp.shape, yt.shape, residuals.shape)
    plt.scatter(yt, residuals)
    plt.xlabel('True values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residuals')
    plt.show()


class GaussianWeightedKNNRegressor(BaseEstimator, RegressorMixin):
    '''
    KNNR augmented by Gaussian distributed weights.
    '''
    
    def __init__(self, n_neighbors=10, sigma=1.0, epsilon=0.5):
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.epsilon = epsilon
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.knn.fit(X_train, y_train)
    
    def _gaussian_weights(self, distances):
        distances = np.where(distances == 0, self.epsilon, distances)  # 将距离为0的位置替换为epsilon
        #print("Distances:", distances)
        #distances /= 100  # 将距离除以100
        #print("Distances:", distances)
        weights = np.exp(-distances**2 / (2 * self.sigma**2))
        #print("Weights:", weights)
        return weights
    
    def predict(self, X):

        predictions=[]
        
        for query_point in X:
            #print("X", X)
            distances, indices = self.knn.kneighbors(query_point.reshape(1, -1))
            weights = self._gaussian_weights(distances/np.max(distances))
            normalized_weights = weights / np.sum(weights)
            weighted_sum = np.sum(normalized_weights * self.y_train[indices.flatten()])
            predictions.append(weighted_sum)
            
        return np.array(predictions)

    def score(self, X, y):
        return r2_score(y, self.predict(X))
    
    def gridsearch_hparam(X_train, y_train, X_val = None, y_val = None, ks = [3, 4, 5, 10, 20], sigmas=[.1, 1, 10,100,1000]):
        
        best_r2 = -np.inf
        best_clf = None

        hparams = [ks, sigmas]
        
        for k, sigma in tqdm(itertools.product(*hparams)):
            
            clf = GaussianWeightedKNNRegressor(k, sigma)

            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, shuffle=True, random_state=0)
            clf.fit(X_train, y_train)
            
            score = r2_score(y_val, clf.predict(X_val))            

            # print(k, sigma, score)
            if score > best_r2:
                best_r2 = score
                best_clf = clf
        
        return best_clf

def run_regressors(X_train, X_val, X_test, y_train, y_val, y_test, 
                   clfs = ['linear regression', 'ridge', 'LASSO', 
                           'SVR(linear)', 'SVR(rbf)', 'SVR(poly)', 
                           'Random Forest Regressor', # 'ANN', 
                           'K-Neighbors Regressor', 'Gaussian Weighted K-Neighbors Regressor'], 
                           X_names = None,
                           order = False,
                           verbose=False):
    
    dic_metrics = {}
    
    for idx, clf_name in enumerate(clfs):
        display(HTML('<h2>' + str(idx+1) + '. ' + str(clf_name) + '</h2>'))
        if clf_name == 'linear regression':

            from sklearn.linear_model import LinearRegression

            lr = LinearRegression()
            yp = lr.fit(X_train, y_train).predict(X_test)
            y_pred_train = lr.predict(X_train)            

        elif clf_name == 'ridge':
            from sklearn.linear_model import Ridge

            hparams = [.1, 10, 1000, 100000, 10000000]
            val_scores = []
            for alpha in hparams:
                ridge = Ridge(alpha = alpha).fit(X_train, y_train)
                val_scores.append(ridge.score(X_val, y_val))

            plt.title('val score ~ alpha')
            plt.xscale('log')
            plt.plot(hparams, val_scores)
            plt.show()

            best_hparam = hparams[np.argmax(val_scores)]
            ridge = Ridge(alpha = best_hparam).fit(X_train, y_train)

            # print('test score:', ridge.score(X_test, y_test))
            yp = ridge.predict(X_test)
            y_pred_train = ridge.predict(X_train)

        elif clf_name == 'LASSO':
            from sklearn.linear_model import Lasso, LassoCV

            hparams = [.001, .01, .1, 10, 100, 1000]
            val_scores = []
            for alpha in hparams:
                lasso = Lasso(alpha = alpha).fit(X_train, y_train)
                val_scores.append(lasso.score(X_val, y_val))

            plt.title('val score ~ alpha')
            plt.xscale('log')
            plt.plot(hparams, val_scores)
            plt.show()

            best_hparam = hparams[np.argmax(val_scores)]
            lasso = Lasso(alpha = best_hparam).fit(X_train, y_train)

            lasso = Lasso(alpha = 0.1).fit(X_train, y_train)
            yp = lasso.predict(X_test)
            y_pred_train = lasso.predict(X_train)

            if verbose:
                from .vis import plot_feature_importance

                fi = np.abs(lasso.coef_)
                plot_feature_importance(fi, X_names, 'LASSO FS Result')

                N = (fi!=0).sum()
                idx = (np.argsort(fi)[-N:])[::-1]
                X_s = X[:, idx]
                print('Important feature Number: ', len(idx))
                print('Important features indices: ', idx)
                if X_names is not None:
                    print('Important features names: ', np.array(X_names)[idx])
                print('Top-'+str(len(idx))+' feature Importance: ', fi[idx])

        elif clf_name.startswith('SVR'): # 'SVR(linear)', 'SVR(rbf)', 'SVR(poly)'

            from sklearn.svm import SVR

            kernel = clf_name[clf_name.find("(")+1:clf_name.find(")")] # clf_name.split('"')[1::2];
            Cs = [ .1, 10, 100, 1000, 10000]
            val_scores = []
            for C in Cs:
                svr = SVR(kernel = kernel, C=C).fit(X_train, y_train)
                val_scores.append(svr.score(X_val, y_val))

            plt.title('val score ~ C')
            plt.xscale('log')
            plt.plot(Cs, val_scores)
            plt.show()

            best_hparam = Cs[np.argmax(val_scores)]
            svr = SVR(kernel = kernel, C=best_hparam)
            yp = svr.fit(X_train, y_train).predict(X_test)
            y_pred_train = svr.predict(X_train)

        elif clf_name == 'Random Forest Regressor':
            from sklearn.ensemble import RandomForestRegressor

            # Each run is random. The result is very unstable
            Ns = [1, 2, 3, 5, 10, 20, 50, 100]
            val_scores = []
            for N in Ns:
                rf_regressor = RandomForestRegressor(n_estimators = N, max_depth = 2).fit(X_train, y_train)
                val_scores.append(rf_regressor.score(X_val, y_val))

            plt.title('val score ~ trees')
            plt.xscale('log')
            plt.plot(Ns, val_scores)
            plt.show()

            best_hparam = Ns[np.argmax(val_scores)]
            rf_regressor = RandomForestRegressor(n_estimators = best_hparam, max_depth = 2)
            rf_regressor.fit(X_train, y_train)

            y_pred_train = rf_regressor.predict(X_train)
            yp = rf_regressor.predict(X_test)

        elif clf_name == 'ANN':
            from keras.models import Sequential
            from keras.layers import Dense
            
            # create ANN model
            model = Sequential()
            
            # Defining the Input layer and FIRST hidden layer, both are same!
            model.add(Dense(units=10, input_dim=X_train.shape[1], kernel_initializer='normal', activation='sigmoid'))

            # Defining the Second layer of the model
            # after the first layer we don't have to specify input_dim as keras configure it automatically
            # model.add(Dense(units=3, kernel_initializer='normal', activation='tanh'))
            
            # The output neuron is a single fully connected node 
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))
            
            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='rmsprop')
            if verbose:
                model.summary()

            # Fitting the ANN to the Training set
            model.fit(X_train, np.array(y_train), batch_size = 2, epochs = 500, verbose=verbose)
            y_pred_train = model.predict(X_train).flatten()
            yp = model.predict(X_test).flatten()

        elif clf_name=='K-Neighbors Regressor':
            from sklearn.neighbors import KNeighborsRegressor

            # param_grid = {
            #     'n_neighbors': [1, 2, 3, 4, 5, 10],
            #     'metric': ['l1', 'l2', 'chebyshev', 'cosine'], # l1 - manhattan / cityblock, l2 - euclidean 
            #}

            Ns = [1, 2, 3, 4, 5, 10, 20]
            val_scores = []
            for N in Ns:
                knr = KNeighborsRegressor(n_neighbors = N).fit(X_train, y_train)
                val_scores.append(knr.score(X_val, y_val))

            plt.title('val score ~ neighbors')
            # plt.xscale('log')
            # .set_major_locator(MaxNLocator(integer=True))
            plt.plot(Ns, val_scores)
            plt.show()

            best_hparam = Ns[np.argmax(val_scores)]
            knr = KNeighborsRegressor(n_neighbors = best_hparam).fit(X_train, y_train) 

            y_pred_train = knr.predict(X_train)
            yp = knr.predict(X_test)

        elif clf_name == 'Gaussian Weighted K-Neighbors Regressor':
            
            gwknr = GaussianWeightedKNNRegressor.gridsearch_hparam(X_train, y_train, X_val, y_val)
            best_hparam = gwknr.n_neighbors, gwknr.sigma

            y_pred_train = gwknr.predict(X_train)
            yp = gwknr.predict(X_test)
        
        else:
            print('Undefined regression model: ' + clf_name)

        dic_metrics[clf_name] = round(r2_score(y_test, yp), 3), round(mean_squared_error(y_test, yp),3) # save R2 and MSE to dict
        draw_regression_plots(y_test, yp, title = clf_name, order=order)
        if verbose:
            print('Training set R2: ', r2_score(y_train, y_pred_train))
            print('Test set R2:', r2_score(y_test, yp))

    display(HTML('<h2>Summary</h2>'))
    tbl_html = '<table><tr><th>Regressor</th><th>R2</th><th>MSE</th></tr>'
    for k,v in dic_metrics.items():
        tbl_html += '<tr><td>'+str(k)+'</td><td>'+str(v[0])+'</td><td>'+str(v[1])+'</td></tr>'
    display(HTML( tbl_html + '</table>'))
