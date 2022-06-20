import os
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab, matplotlib
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sqlalchemy import false
from .data import SMOTE

from .vis import *
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/data/"

DATASET_MAP = {'s4_formula': ('7341_C1.csv', ',', False ,'7341_C1 desc.txt'),
's3_formula': ('B3.csv', ',', False ,'B3 desc.txt'),
's4_formula_c2': ('7341_C2.csv', ',', True ,'7341_C2 desc.txt'),
'milk_tablet_candy': ('734b.csv',',', False,'734b desc.txt'),
'vintage': ('7344.txt','\t', False,'7344 desc.txt'),
'vintage_c2': ('7344_C03.csv',',',True,'7344_C03 desc.txt'),
'beimu': ('754a_C2S_Beimu.txt',',', True,'754a_C2S_Beimu desc.txt'),
'shihu': ('754b_C2S_Shihu.txt',',',True,'754b_C2S_Shihu desc.txt'),
'huangqi_rm': ('7044X_RAMAN.csv',',', True,'7044X_RAMAN desc.txt'),
'huangqi_uv': ('7143X_UV.csv',',',True,'7143X_UV desc.txt'),
'cheese': ('Cheese_RAMAN.csv',',',True,'Cheese_RAMAN desc.txt'),}

def get_available_datasets():
    return list( DATASET_MAP.keys() )

def DataSetIdToPath(id):
    return DATASET_MAP[id]

def LoadDataSet(id):
    
    path, delimiter, has_y, path_desc = DataSetIdToPath(id)
    X, y, X_names = PeekDataSet(DATA_FOLDER + path, delimiter, has_y)
    
    if os.path.exists (DATA_FOLDER + path_desc):
        f=open(DATA_FOLDER + path_desc, "r", encoding = 'UTF-8')
        desc = f.read()
        f.close()

        print (desc)
        
    else:
        desc = ''
    
    return X, y, X_names, desc

def OpenDataSet(path, delimiter = ',', has_y = True):
    '''
    Parameters
    ----------
    has_y : boolean, whether there is y column, usually the 1st column.
    '''

    # path = "github/data/754a_C2S_Beimu.txt"
    data = pd.read_csv(path, delimiter=delimiter) # ,header=None

    cols = data.shape[1]

    if has_y:

        # convert from pandas dataframe to numpy matrices
        X = data.iloc[:,1:cols].values # .values[:,::10]
        y = data.iloc[:,0].values.ravel() # first col is y label
        # use map(float, ) to convert the string list to float list
        X_names = list(map(float, data.columns.values[1:])) # list(map(float, data.columns.values[1:])) # X_names = np.array(list(data)[1:])

    else:

        X = data.values
        y = None

        # use map(float, ) to convert the string list to float list
        X_names = list(map(float, data.columns.values)) # X_names = np.array(list(data)[1:])

    return X, y, X_names

def ScatterPlot(X, y):    

    pca = PCA(n_components=2) # keep the first 2 components
    X_pca = pca.fit_transform(X)
    plotComponents2D(X_pca, y)
    plt.show()

def Draw_Average (X, X_names):

    matplotlib.rcParams.update({'font.size': 16})

    plt.figure(figsize = (20,5))

    plt.plot(X_names, X.mean(axis = 0), "k", linewidth=1, label= 'averaged waveform $± 3\sigma$ (310 samples)') 
    plt.errorbar(X_names, X.mean(axis = 0), X.std(axis = 0)*3, 
                color = "dodgerblue", linewidth=3, 
                alpha=0.1)  # X.std(axis = 0)
    plt.legend()

    plt.title(u'Averaged Spectrum')
    # plt.xlabel(r'Wavenumber $(cm^{-1})$')
    # plt.ylabel('Raman signal')
    plt.yticks([])
    # plt.xticks([])
    plt.show()

    matplotlib.rcParams.update({'font.size': 12})


def Draw_All (X, X_names, titles = None):

    matplotlib.rcParams.update({'font.size': 16})

    if titles is None:
        titles = range(len(X))

    for x, title in zip(X, titles):

        plt.figure(figsize = (20,5))

        plt.plot(X_names, x, "k", linewidth=1)

        plt.title(title)
        plt.yticks([])
        plt.show()

    matplotlib.rcParams.update({'font.size': 12})


def Draw_Class_Average (X, y, X_names, SD = 1, shift = 200):
    '''
    Parameter
    ---------
    SD : Integer, show the +/- n-std errorbar. When SD = 0, will not show. 
    shift : y-direction shift to disperse waveforms of different classes.
    '''

    matplotlib.rcParams.update({'font.size': 18})

    plt.figure(figsize = (24,10))

    for c in set(y):    
        Xc = X[y == c]
        yc = y[y == c] 

        if SD == 0:
            plt.plot(X_names, np.mean(Xc,axis=0) + c*shift, label= 'Class ' + str(c) + ' (' + str(len(yc)) + ' samples)') 
        
        else: # show +/- std errorbar
            plt.errorbar(X_names, Xc.mean(axis = 0) + shift*c, Xc.std(axis = 0)*SD, 
                        color = ["blue","red","green","orange"][c], 
                        linewidth=1, 
                        alpha=0.2,
                        label= 'Class ' + str(c) + ' (' + str(len(yc)) + ' samples)' + ' mean ± 1 SD',
                        )  # X.std(axis = 0)
            plt.scatter(X_names, np.mean(Xc,axis=0).tolist() + c*shift, 
                    color = ["blue","red","green","orange"][c],
                    s=1 
                    ) 

        plt.legend()

    plt.title(u'Averaged Spectrums for Each Category\n')
    # plt.xlabel(r'$ cm^{-1} $') # depending on it is Raman or MS
    plt.ylabel('Intensity')
    plt.yticks([])
    plt.show()

    matplotlib.rcParams.update({'font.size': 10})

def PeekDataSet(path,  delimiter = ',', has_y = True):

    X, y, X_names = OpenDataSet(path, delimiter=delimiter, has_y = has_y)

    if y is None:
        Draw_Average (X, X_names)
    else:
        Draw_Class_Average (X, y, X_names)

    ScatterPlot(X, y)
    
    return X, y, X_names

def Upsample(target_path, X, y, X_names, folds = 3, d = 0.5):
    '''
    Upsample a dataset by SMOTE.

    Parameters
    ----------
    X_names : the labels for each X feature
    folds : expand to N folds
    d : sampling distance in SMOTE
    '''

    SMOTE.expand_dataset(X, y, X_names, target_path, d, folds)

def SaveDataSet(targe_path, X, y, X_names):
    '''
    Save X, y to a local csv file.

    Parameters
    ----------
    X_names : the labels for each X feature
    '''

    dfX = pd.DataFrame(X)
    dfX.columns = X_names

    dfY = pd.DataFrame(y)
    dfY.columns = ['label']

    df = pd.concat([dfY, dfX], axis=1)
    df = df.sort_values(by=['label'], ascending=True)
    df.to_csv(targe_path, index=False) # don't create the index column