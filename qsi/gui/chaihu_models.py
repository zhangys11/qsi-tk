import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, r2_score

# import warnings
# warnings.filterwarnings('ignore')

def train_models():
    print("Training models...")

    data=pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + "/static/dataset.xlsx")
    # data = data.sample(len(data), random_state=10)
    x=data.iloc[:,:-4]
    y=data.iloc[:,-4]
    s = StandardScaler()
    s.fit(x)
    x = s.transform(x)
    lda = LDA(n_components=2)
    lda.fit(x, y)
    x = lda.transform(x)
    df_dr = pd.DataFrame(x)

    data1 = pd.concat([df_dr, data.iloc[:, -4:]], axis=1)
    data1 = data1.sample(len(data1), random_state=10)

    a=int(round(0.75*len(data),0))
    train_x = data1.iloc[:a, :2]
    train_y1 = data1.iloc[:a, -3]
    train_y2 = data1.iloc[:a, -2]
    train_y3 = data1.iloc[:a, -1]
    test_x = data1.iloc[a:, :2]
    test_y1 = data1.iloc[a:, -3]
    test_y2 = data1.iloc[a:, -2]
    test_y3 = data1.iloc[a:, -1]

    model_1 = SVR(C=1000, kernel='rbf')
    model_2 = SVR(C=800, kernel='rbf')
    model_3 = SVR(C=1200, kernel='rbf')
    model_ssa = model_1.fit(train_x, train_y1)
    model_ssc = model_2.fit(train_x, train_y2)
    model_ssd = model_3.fit(train_x, train_y3)
    ssa = model_ssa.predict(test_x)
    ssc = model_ssc.predict(test_x)
    ssd = model_ssd.predict(test_x)
    print('r2 ssa', r2_score(test_y1, ssa))
    print('r2 ssc', r2_score(test_y2, ssc))
    print('r2 ssd', r2_score(test_y3, ssd))

    data2 = pd.concat([df_dr,y],axis=1)
    data2 = data2.sample(len(data2), random_state=8)

    train_x=data2.iloc[:a,:-1]
    train_y=data2.iloc[:a,-1]
    test_x = data2.iloc[a:, :-1]
    test_y = data2.iloc[a:, -1]

    model_classifier=SVC(C=10,kernel='rbf')
    model_classifier.fit(train_x,train_y)
    predict = model_classifier.predict(test_x)
    a = np.sum(test_y == predict)
    print('acc',a / len(test_y))
    C = confusion_matrix(test_y, predict)
    print('confusion matrix:',C)
    model_classifier = SVC(C=10, kernel='rbf', probability=True)
    model_classifier.fit(train_x, train_y)

    return s, lda, model_ssa,model_ssc,model_ssd, model_classifier


def predict_SSa(X):
    pre_ssa=model_ssa.predict(X)
    return pre_ssa

def predict_SSc(X):
    pre_ssc = model_ssc.predict(X)
    return pre_ssc

def predict_SSd(X):
    pre_ssd = model_ssd.predict(X)
    return pre_ssd

def predict_class(X):
    pre_class=model_classifier.predict_proba(X)
    a=pre_class.argmax()
    place_list=['Gansu','Inner Mongolia(wild)','Inner Mongolia(cultivated)','Inner Mongolia(black)','Shaanxi']
    place=place_list[a]
    i_1=pre_class[0][3]
    i_2=pre_class[0][1]
    i_3=pre_class[0][2]
    g=pre_class[0][0]
    s=pre_class[0][4]
    proba=[]
    proba.append(i_1)
    proba.append(i_2)
    proba.append(i_3)
    proba.append(g)
    proba.append(s)
    return proba, place

def get_html(X):
    '''
    Generate a summary report in HTML format
    '''

    ssa = predict_SSa(X)
    ssc = predict_SSc(X)
    ssd = predict_SSd(X)
    pro, place = predict_class(X)

    html = '<table class="table table-striped" bgcolor="white">'

    tr = '<tr bgcolor="white"><th colspan="3" style="center">Predicted origin 产地预测:  ' + str(place) + '</th><tr>'  # <th> Value </th><th> Details </th>

    html += tr

    tr = '<tr><td colspan=3>' + str('Inner Mongolia 内蒙 (wild 野生:' 
                                       + str(round(pro[0]*100,2))+str('%') 
                                       + ', '+'cultivated 种植:'+str(round(pro[1]*100,2)) + str('%') + ', '
                                       + 'black bupleurum 黑柴胡:'+str(round(pro[2]*100,2)) + str('%')+'), ' 
                                       + '<br/>Gansu 甘肃:'+str(round(pro[3]*100,2))+str('%')+', ' 
                                       + '<br/>Shaanxi 陕西:'+str(round(pro[4]*100,2)))+str('%') + '</td></tr>'
    html += tr

    tr = '<tr><th> Saikosaponin A, 柴胡皂苷A, SSa (mg/L)</th><th>Saikosaponin C, 柴胡皂苷C, SSc (mg/L)</th><th>Saikosaponin D, 柴胡皂苷D, SSd (mg/L)</th></tr>'  # <th> Value </th><th> Details </th>
    html += tr

    tr = '<tr>' \
         '<td> ' + str(round(float(ssa),2) ) + '</td>' \
         '<td> ' + str(round(float(ssc), 2)) + '</td>' \
         '<td> ' + str(round(float(ssd), 2)) + '</td>' \
         '</tr>'
    html += tr
    html += "</table>"

    return html

def load_file(pathname):
    '''
    Load data from a csv file
    '''
    X = pd.read_excel(pathname)
    if len(X)!=1:
        print('The model only supports single sample analysis.')
    else:
        X = s.transform(X)
        X = lda.transform(X)
    return X

def analyze_file(fn):
    if os.path.isfile(fn) is False:
        return 'File ' + fn + ' does not exist.'

    X= load_file(fn)
    return get_html(X)

def analyze_probs(fn):
    if os.path.isfile(fn) is False:
        return 'File ' + fn + ' does not exist.'

    X = load_file(fn)
    probs,_ = predict_class(X)
    a = str('wild')+str('  ')+str(round(probs[0]*100,2))+str("%")
    b = str('cultivated') + str('  ') + str(round(probs[1] * 100,2)) + str("%")
    c = str('black') + str('  ') + str(round(probs[2] * 100,2)) + str("%")
    f=f"{a}<br/>{b}<br/>{c}"
    d = str(round(probs[3] * 100,2)) + str("%")
    e = str(round(probs[4] * 100,2)) + str("%")
    pro=[f,d,e]
    return pro

# if __name__ == '__main__':
s, lda, model_ssa,model_ssc,model_ssd, model_classifier=train_models()