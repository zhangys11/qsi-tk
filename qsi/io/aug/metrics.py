import scipy
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, f1_score,precision_score, recall_score, accuracy_score
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from . import SMOTE, Gaussian, GMM, KDE, VAE, TVAE, MDN, GAN, DCGAN, ctGAN

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

def calculate_clf_metrics(clf, df_test):
    '''
    clf - a classifier
    df_test - a pandas dataframe. The last column is target var (y).
    '''
    y_pred = clf.predict(df_test.iloc[:,:-1])
    accuracy = accuracy_score(df_test.iloc[:,-1], y_pred)
    return accuracy,f1_score(df_test.iloc[:,-1], y_pred, average='macro'),precision_score(df_test.iloc[:,-1], y_pred, average='macro'),recall_score(df_test.iloc[:,-1], y_pred, average='macro')

def expand_dataset(X,y,nobs, models=['Guassian','GMM','KDE','MDN','VAE','TVAE','GAN','DCGAN','CTGAN']):  
    '''
    Use this function to perform data aug on the class with less samples.

    Parameters
    ----------
    X, y - samples of the smaller class
    nobs - new samples to be generated
    models - always use all models for the current version
    '''
    
    x_new_VAE,y_new_VAE=VAE.expand_dataset(X,y,nobs, verbose=True)
    x_new_TVAE,y_new_TVAE=TVAE.expand_dataset(X,y,nobs,epochs=20,embedding_dim=64,compress_dims=(64, 64),decompress_dims=(32, 32))
    x_new_CTGAN,y_new_CTGAN=ctGAN.expand_dataset(X,y,nobs,epochs=20, embedding_dim=32, generator_dim=(128, 30, 16), discriminator_dim=(64, 16))    
    x_new_Gaussian,y_new_Gaussian=Gaussian.expand_dataset(X,y,nobs)
    x_new_GMM,y_new_GMM=GMM.expand_dataset(X,y,nobs,n_gaussians=10)
    x_new_KDE,y_new_KDE=KDE.expand_dataset(X,y,nobs)
    x_new_MDN,y_new_MDN=MDN.expand_dataset(X,y,nobs,n_gaussians=5,epochs=100,n_hidden=15)
    x_new_GAN,y_new_GAN=GAN.expand_dataset(X,y,nobs,epochs = 100,BATCH_SIZE=16,noise_dim=(100))
    x_new_DCGAN,y_new_DCGAN=DCGAN.expand_dataset(X,y,nobs,epochs = 100,BATCH_SIZE=16,noise_dim=(100))

    X_list=[x_new_Gaussian,x_new_GMM,x_new_KDE,x_new_MDN,x_new_VAE,x_new_TVAE,x_new_GAN,x_new_DCGAN,x_new_CTGAN]
    y_list=[y_new_Gaussian,y_new_GMM,y_new_KDE,y_new_MDN,y_new_VAE,y_new_TVAE,y_new_GAN,y_new_DCGAN,y_new_CTGAN]
    
    return X_list,y_list, models

def evaluation_model(X_list,models,df_test,target_y): # df_test[df_test['Label']==0]
    data_org=test_set
    data_org=data_org[data_org['Label']==0]
    data_org.drop('Label',axis=1,inplace=True)
    data_org[data_org<0.0001]=1
    data_org=data_org.reset_index(drop=True)

    plt.figure(figsize=(24,8))
    plt.plot(data_org.columns,data_org.mean(),c='r',label='original data')
    plt.xticks(range(0,2000,300),fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(prop={'size':20})
    plt.show()

    eva_result=pd.DataFrame()
    for (i,j) in zip(X_list,model_name_list):
        print(j)
        data_sam=i
        data_sam.columns=data_sam.columns.astype(str)

        plt.figure(figsize=(24,8))
        plt.plot(data_sam.columns,data_sam.mean(),c='r',label=j)
    #     plt.plot(data_org.columns,data_org.mean(),c='g',label='original data')
        plt.xticks(range(0,2000,300),fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(prop={'size':20})
        plt.show()

        data_sam=data_sam.reset_index(drop=True)
        data_sam[data_sam<0.0001]=1
        ### 评估生成数据
        metric=calculate_metrics(data_org,data_sam)
    #     print(metric)
        model_name=j
        eva_result[model_name]=metric
        
        eva_result.index=['KL','JS','pearson','spearman','kendall','similarity','r2','mse','SSIM']#,'PSNR']
    return eva_result

def metric_generate_data(data_all):
    test_acc=[]
    test_f1=[]
    test_pre=[]
    test_rec=[]

    # 创建一个SVM分类器对象
    clf = svm.SVC(C=10,kernel='rbf')
    clf.fit(data_all.iloc[:,:-1],data_all.iloc[:,-1])
    # print('SVM')
    acc,f1,pre,rec=test_metric(clf)
    test_acc.append(acc)
    test_f1.append(f1)
    test_pre.append(pre)
    test_rec.append(rec)

    clf = xgb.XGBClassifier(eval_metric=['logloss','auc','error'],n_estimators=400)#,learning_rate=0.1,reg_lambda=0.001,reg_alpha=0,max_depth=10)
    clf.fit(data_all.iloc[:,:-1],data_all.iloc[:,-1])
    # print('CGB')
    acc,f1,pre,rec=test_metric(clf)
    test_acc.append(acc)
    test_f1.append(f1)
    test_pre.append(pre)
    test_rec.append(rec)

    ## RF
    clf = RandomForestClassifier(n_estimators=500, random_state=50)#min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)
    clf.fit(data_all.iloc[:,:-1],data_all.iloc[:,-1])
    # print('RF')
    acc,f1,pre,rec=test_metric(clf)
    test_acc.append(acc)
    test_f1.append(f1)
    test_pre.append(pre)
    test_rec.append(rec)

    ## Adaboost
    clf = AdaBoostClassifier(n_estimators=300,random_state=37)  # adaboost
    clf = clf.fit(data_all.iloc[:,:-1],data_all.iloc[:,-1])  # 拟合训练集
    # print('adaboost')
    acc,f1,pre,rec=test_metric(clf)
    test_acc.append(acc)
    test_f1.append(f1)
    test_pre.append(pre)
    test_rec.append(rec)

    ## GBDT
    clf = GradientBoostingClassifier(n_estimators=250,random_state=42)  # gbdt
    clf = clf.fit(data_all.iloc[:,:-1],data_all.iloc[:,-1])  # 拟合训练集
    # print('GBDT')
    acc,f1,pre,rec=test_metric(clf)
    test_acc.append(acc)
    test_f1.append(f1)
    test_pre.append(pre)
    test_rec.append(rec)
    
    return test_acc,test_f1,test_pre,test_rec

def classification_run(train_set,X_list,y_list):
    org_acc,org_f1,org_pre,org_rec=metric_generate_data(train_set)
    metric_all=pd.DataFrame()
    for i in tqdm(range(9)):
        synthetic_data=pd.concat([X_list[i],y_list[i]],axis=1)
    #     synthetic_data.columns=df.columns
        data_all=pd.concat([synthetic_data,train_set],axis=0)
        test_acc,test_f1,test_pre,test_rec=metric_generate_data(data_all)
        acc=model_name_list[i]+' acc'
        f1=model_name_list[i]+' f1'
        pre=model_name_list[i]+' precision'
        rec=model_name_list[i]+' recall'
        metric_all[acc]=test_acc
        metric_all[pre]=test_pre
        metric_all[rec]=test_rec
        metric_all[f1]=test_f1
    metric_all['org_acc']=org_acc
    metric_all['org_precision']=org_pre
    metric_all['org_recall']=org_rec
    metric_all['org_f1']=org_f1
    metric_all.index=['SVM','XGBoost','RF','Adaboost','GBDT']
    return metric_all

