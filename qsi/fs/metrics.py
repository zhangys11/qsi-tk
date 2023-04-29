from math import log
import numpy as np

def ic(y, y_pred, k):
    '''
    information criterion for model selection.
    returns AIC, BIC and AICC (corrected AIC).

    Example
    -------
    y_pred = np.array([[0.69910715, 0.30089285],
    [0.50212636, 0.49787364],
    [0.50212636 ,0.49787364]])

    y = [1,1,0]

    ic(y,y_pred,k = 100)
    '''
    
    y_oh = np.eye(len(set(y)))[y] # convert to one-hot encoding. make sure set(y) has all the labels.
    n = len(y)
    resid = (y_oh - y_pred).flatten()
    sse = sum(resid ** 2)
    AIC = n*log(sse/n) + 2*k
    BIC = n*log(sse/n) + log(n)*k  
    AICC = n*log(sse/n) + (n+k)/(1-(k+2)/n)
    
    return AIC, BIC, AICC