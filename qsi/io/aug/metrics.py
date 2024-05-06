import scipy
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


# def JS_divergence(p,q):
#     M=(p+q)/2
#     return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)

def calculate_pairwise_metrics(px,py):
    KL = scipy.stats.entropy(px, py)
    JS = scipy.spatial.distance.jensenshannon(px,py) # JS_divergence(px, py)
    pearson = px.corr(py,method="pearson")
    spearman = px.corr(py,method="spearman")
    kendall = px.corr(py,method="kendall")
    similarity = 1 - scipy.spatial.distance.cosine(px,py)
    r2=r2_score(px,py)
    mse=mean_squared_error(px,py)

    ### SSIM and PSNR are orginally used on images. We tile/stack the 1D signal to 2D array and then apply the two metrics.  
    n = 20 # len(px)
    SSIM=structural_similarity(np.tile(px, (n,1)).astype(int),np.tile(py, (n,1)).astype(int))
    PSNR=peak_signal_noise_ratio(np.tile(px, (n,1)),np.tile(py, (n,1)))
    
    return KL,JS,pearson,spearman,kendall,similarity,r2,mse,SSIM, PSNR

def calculate_metrics(dataset_raw,dataset_syn):
    KL=[]
    JS=[]
    pearson=[]
    spearman=[]
    kendall=[]
    similarity=[]
    r2=[]
    mse=[]
    SSIM=[]
    PSNR=[]


    for i in range(dataset_raw.shape[0]):
        for j in range(dataset_syn.shape[0]):
            px = dataset_raw.iloc[i,:]
            py = dataset_syn.iloc[j,:]
            result=calculate_pairwise_metrics(px,py)

            KL.append(result[0])
            JS.append(result[1])
            pearson.append(result[2])
            spearman.append(result[3])
            kendall.append(result[4])
            similarity.append(result[5])
            r2.append(result[6])
            mse.append(result[7])
            SSIM.append(result[8])
            PSNR.append(result[9])


    KL_mean=np.mean(KL)
    JS_mean=np.mean(JS)
    pearson_mean=np.mean(pearson)
    spearman_mean=np.mean(spearman)
    kendall_mean=np.mean(kendall)
    similarity_mean=np.mean(similarity)
    r2_mean=np.mean(r2)
    mse_mean=np.mean(mse)
    SSIM_mean = np.mean(SSIM)
    PSNR_mean = np.mean(PSNR)

    return [KL_mean,JS_mean,pearson_mean,spearman_mean,kendall_mean,similarity_mean,r2_mean,mse_mean,SSIM_mean, PSNR_mean]
    