from qsi import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLassoCV
from tqdm import tqdm

def load_sample_dataset():

    XA, yAc, X_names, _, labels = io.load_dataset('yogurt_fermentation_a', x_range = list(range(100,1400)), shift=400)
    n_classes = len(labels)

    # Convert to one-hot encoding to support MT Lasso 
    yAe = np.eye(n_classes)[yAc]

    XB, yBc, X_names, _, labels = io.load_dataset('yogurt_fermentation_b', x_range = list(range(100,1400)), shift=400)

    # Convert to one-hot encoding to support MT Lasso 
    yBe = np.eye(n_classes)[yBc]

    return XA, yAe, XB, yBe, X_names, labels

def weighted_mtlasso(XA, yAe, XB, yBe, X_names, labels):
    '''
    The complete sample code is provided in qsi.fs.mt.ipynb
    '''

    # 合并数据集AB
    XMT = np.vstack((XA, XB))

    dic_acc = {}
    best_hparam = None
    best_T1_acc = 0
    best_cm = ()

    # 存储T1的top-3和top-5准确率
    dic_top3_acc = {}
    dic_top5_acc = {}

    for w in tqdm([.01, .05, .1, .25, .5, 1]):

        y_T1 = np.vstack((yAe, yBe))  # T1: time
        y_T2 = np.array([0] * len(yAe) + [1] * len(yBe)) * w  # T2: brand

        # 划分整体训练集和测试集
        X_train, X_test, y_T1_train, y_T1_test, y_T2_train, y_T2_test = train_test_split(
            XMT, y_T1, y_T2, test_size=0.2, random_state=42
        )

        # 初始化多任务Lasso模型，用多任务LassoCV找到最优的acc
        model = MultiTaskLassoCV(max_iter=1000, tol=1e-2, selection="cyclic")

        # 拟合模型
        model.fit(X_train, np.column_stack((y_T1_train, y_T2_train)))

        # 在测试集上进行预测
        yp = model.predict(X_test)

        # 计算混淆矩阵和准确率
        cm_T1 = confusion_matrix(y_T1_test.argmax(axis=1), yp[:, :-1].argmax(axis=1))
        acc_T1 = accuracy_score(y_T1_test.argmax(axis=1), yp[:, :-1].argmax(axis=1))
        top3_acc_T1 = top_k_accuracy_score(y_T1_test.argmax(axis=1), yp[:, :-1], k=3)
        top5_acc_T1 = top_k_accuracy_score(y_T1_test.argmax(axis=1), yp[:, :-1], k=5)

        cm_T2 = confusion_matrix(y_T2_test / w, yp[:, -1] / w > 0.5)  # use default 0.5 threshold
        acc_T2 = accuracy_score(y_T2_test / w, yp[:, -1] / w > 0.5)

        dic_acc[round(w, 2)] = (round(acc_T1, 3), round(acc_T2, 3))
        dic_top3_acc[round(w, 2)] = round(top3_acc_T1, 3)
        dic_top5_acc[round(w, 2)] = round(top5_acc_T1, 3)

        if best_T1_acc < acc_T1:
            best_hparam = round(w, 2)
            best_cm = (cm_T1, cm_T2)
            best_T1_acc = acc_T1

    print(dic_acc)
    print('Best hyperparameter:', best_hparam)

    print("Best test accuracy on T1:", dic_acc[best_hparam][0])
    print("Best test accuracy on T2:", dic_acc[best_hparam][1])