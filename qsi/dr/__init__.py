# Dimensionality Reduction Algorithms
# MF-based DR algorithms are provided in mf.py
# In this file, we provide DCT & FFT -based DR

import cv2
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.fftpack
import numpy as np
from .metrics import calculate_recon_error, DRMetrics
from IPython.core.display import HTML, display
from sklearn.preprocessing import MinMaxScaler

def dct(x, K = None, flavor = 'fftpack'):
    '''
    Parameters
    ----------
    K : The DCT components to be kept. 
        If K = None, will try a range of different K values; 
        otherwise will show the DR and Recovered signal with specified K.
    flavor : the underlying DCT implementation, cv2 or fftpack
    
    Remarks
    -------
    For `scipy.fftpack.dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False)`, 
    $$
    y[k] = 2 \sum_{i=0}^{M-1} x[i] cos({{(2i+1)\pi k} \over {2M} }), 0 <= k < M
    $$
    If norm='ortho', y[k] is multiplied by a scaling factor f:
    $$
    f = \sqrt{1/(4M)}
    $$, if k = 0,
    $$
    f = \sqrt{1/(2M)}
    $$,  otherwise.
    
    '''

    # OpenCV DCT & IDCT input requires even dimensions, while scipy.fftpack does not.
    if flavor == 'cv2' and len(x) % 2 == 1:
        print('Cut x to be even length. OpenCV DCT & IDCT input requires even dimensions.')
        x = x[:-1]

    if K is None:

        R = range(2,len(x),round(len(x) / 20))
        Errs = []
        for K in tqdm(R):

            if flavor == 'cv2':
                x_dst = cv2.dct(x)
            else:
                x_dst = scipy.fftpack.dct(x, norm = 'ortho')

            x_dst[K:] = 0

            if flavor == 'cv2':
                x_dst_r = cv2.idct(x_dst)
            else:
                x_dst_r = scipy.fftpack.idct(x_dst, norm = 'ortho')
            
            Err = norm(x_dst_r - x)**2
            Errs.append(Err)    

        plt.figure(figsize=(16,3))
        plt.plot(list(R), Errs)
        plt.title ('MSE ~ K (kept DCT components)')
        plt.show()

    elif K >= 2:

        plt.figure(figsize=(16,3))
        plt.plot(x)
        plt.title('original signal')
        plt.show()

        if flavor == 'cv2':
            x_dst = cv2.dct(x)
        else:
            x_dst = scipy.fftpack.dct(x, norm = 'ortho')
        plt.figure(figsize=(16,3))
        plt.plot(x_dst)
        plt.title('DCT')
        plt.show()

        print(K ,' / ', len(x_dst), ' = ', K/len(x_dst))

        x_dst[K:] = 0

        if flavor == 'cv2':
            x_dst_r = cv2.idct(x_dst)
        else:
            x_dst_r = scipy.fftpack.idct(x_dst, norm = 'ortho')

        plt.figure(figsize=(16,3))
        plt.plot(x_dst_r)
        plt.title('IDCT')
        plt.show()


def dataset_dct_row_wise(X, K = 2, flavor = 'fftpack'):
    '''
    Returns
    -------
    mse, ms, and relative-mse (mse/ms)
    '''

    XE = X.copy()
    if flavor == 'cv2':    
        if XE.shape[1] % 2 == 1:
            XE = XE[:,:-1]            

    D = np.zeros(XE.shape)

    for i in tqdm(range(len(XE))):
        x = XE[i]

        if flavor == 'cv2':
            x_dst = cv2.dct(x)
        else:
            x_dst = scipy.fftpack.dct(x, norm = 'ortho')

        x_dst[K:] = 0

        if flavor == 'cv2':
            x_dst_r = cv2.idct(x_dst)
        else:
            x_dst_r = scipy.fftpack.idct(x_dst, norm = 'ortho')
        
        D[i]= x_dst_r.flatten()

    return calculate_recon_error(XE, D)


def dataset_dct(X, tq = 0.99, flavor = 'fftpack'):
    '''
    Parameters
    ----------
    tq : threshold quantile. only DCT components > tq*max are kept.

    Use 2D DCT to do DR and Recovery. 
    DCT is optimal for human eye: the distortions introduced occur at the highest frequencies only, neglected by human eye as noise.
    DCT can be performed by simple matrix operations: Image is first transformed to DCT space and dimensionality reduction is achieved during inverse transform by discarding the transform coefficients corresponding to highest frequencies.
    Computing DCT is not data-dependent, unlike PCA that needs the eigenvalue decomposition of data covariance matrix, which is why DCT is orders of magnitude cheaper to compute than PCA.
        
    2D-DCT is applied to the entire data matrix. features/row has a similar meaning to k in dimensionality reduction.
    The 2D DCT (two-dimensional discrete cosine transformation) is also conducted for comparison, as it is behind the widely used lossy image compression method, e.g. JPEG. DCT transforms the original data to the frequency domain. By setting a certain threshold, only the non-trivial frequency components (usually the low frequencies) above the threshold are kept. The 2D IDCT (inverse DCT) is used to reconstruct the data. Although DCT is not based on matrix factorization, it has the same concepts of compression ratio and reconstruction error, and can be compared with PCA, NMF, and LAE.
    '''

    XE = X.copy()    
    if XE.shape[1] % 2 == 1:
        XE = XE[:,:-1]

    # scale to [0,1]
    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    XE_mm_scaled = mm_scaler.fit_transform(XE)

    if flavor == 'cv2':  
        dst2D = cv2.dct(XE_mm_scaled)
    else:
        dst2D = scipy.fftpack.dct(scipy.fftpack.dct(XE_mm_scaled, norm='ortho').T, norm='ortho').T
        # dst2D = scipy.fftpack.dct(scipy.fftpack.dct(XE.T, norm='ortho').T, norm='ortho')

    print('MAX / MIN', 
        np.max(dst2D), 
        np.min(dst2D))

    s = '<table><tr> <th>THRESHOLD</th> <th>Features per row</th> <th>Non-zero percentage</th> <th>MSE</th> <th>Relative error</th></tr>'

    # Threshold
    for i in (range(0, 10, 1)):
        
        thresh = i/100.0
        dst2D_thresh = dst2D * (abs(dst2D) >=(thresh*np.max(dst2D)))
                
        if flavor=='cv2':
            idst2D = cv2.idct(dst2D_thresh)
        else:
            idst2D = scipy.fftpack.idct(scipy.fftpack.idct(dst2D_thresh.T, norm='ortho').T, norm='ortho')
        
        # inverse transformation by the same scaler
        original_data_unscaled = mm_scaler.inverse_transform(XE_mm_scaled)
        decoded_data_unscaled = mm_scaler.inverse_transform(idst2D)

        mse, ms, rmse = calculate_recon_error(original_data_unscaled, decoded_data_unscaled)

        features_per_row = np.sum( dst2D_thresh != 0.0 ) / (XE.shape[0]*1.0)
        percent_nonzeros = features_per_row / (XE.shape[1]*1.0)
        
        s += '<tr>'
        s += ('<td>' + str(thresh) + '</td>' 
        + '<td>' + str(features_per_row) + '</td>' 
        + '<td>' + str(percent_nonzeros) + '</td>' 
        + '<td>' + str(mse) + '</td>'
        + '<td>' + str(rmse) + '</td>')
        
        #print(thresh, '\n', 
        #      features_per_row, percent_nonzeros, 
        #      mse, ms, mse/ms)
        s += '</tr>'
        
    s += '</table>'
    display(HTML(s))

    if tq > 0 and tq < 1:        

        dst2D_thresh = dst2D * (abs(dst2D) >=(tq * np.max(dst2D)))
        
        # idst2D = scipy.fftpack.idct(scipy.fftpack.idct(dst2D_thresh, norm='ortho').T, norm='ortho').T
        if flavor=='cv2':
            idst2D = cv2.idct(dst2D_thresh)
        else:
            idst2D = scipy.fftpack.idct(scipy.fftpack.idct(dst2D_thresh.T, norm='ortho').T, norm='ortho')
        
        # inverse transformation by the same scaler
        original_data_unscaled = mm_scaler.inverse_transform(XE_mm_scaled)
        decoded_data_unscaled = mm_scaler.inverse_transform(idst2D)

        features_per_row = np.sum( dst2D_thresh != 0.0 ) / (XE_mm_scaled.shape[0]*1.0)
        percent_nonzeros = features_per_row / (XE_mm_scaled.shape[1]*1.0)

        print(features_per_row, percent_nonzeros)
        drm = DRMetrics(original_data_unscaled, idst2D, decoded_data_unscaled)
        drm.report()