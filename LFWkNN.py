# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:40:22 2020

@author: user
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier as knn
import os
from collections import Counter
from sklearn.metrics import  precision_score, f1_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from feature_weight import feature_weight
os.getcwd()
os.chdir('C:/Users/user/Google 드라이브/Lecture/advanced machine learning/project/data/')
import warnings
warnings.filterwarnings(action='ignore')


#%%


def KNN(X_train, Y_train, X_test, Y_test, label) :
    knn_clf = knn(n_neighbors=5)
    knn_clf.fit(X_train, Y_train)
    
    y_pred = knn_clf.predict(X_test)
    return y_pred



#%%
def weighted_l2(weight, row0, row1):
    sub = row0-row1
    return np.sqrt((weight*sub*sub).sum()) 

#%%

# local knn : 각 train sample 마다 적절한 K를 할당해주기 위해 k 값마다 local k accuracy를 할당해 주는 과정 

def local_k_acc(X_train,X_train_k,Y_train):
    col=list(X_train.columns)
    col2=list(X_train.columns)
    col2.append('dist')
    for i in range(1,101):
        k=i
        global weight
        k2=[]
        for ii in range(len(X_train)):
            point=np.array(X_train.iloc[ii,0:len(col)])
            X_train['dist'] = [weighted_l2(weight, point, np.array(sample)) for idx, sample in X_train.iterrows()]
            sort_train = X_train.sort_values(by=['dist'],axis=0)
            kk=sort_train.iloc[range(k+1)]
            kk=kk.iloc[1:,:]
            kk_inx=list(kk.index)
            
            train_class=[]
            X_train=X_train.iloc[:,0:len(col)]
            
            for j in (kk_inx):
                train_class.append(Y_train[j])
                
            cnt=Counter(train_class)
            mode=cnt.most_common(1)
            mode_val=list(dict(cnt).values())
            mode1=mode[0][0]
            mode_ok=cnt[mode1]
            k2.append(round(int(mode_ok)/int(sum(mode_val)),2))
        
        
        X_train_k[str(k)+'k']=k2
    #X_train_k.drop(['sepal length','sepal width','petal length','petal width','dist'], axis=1, inplace=True)  
    X_train_k.drop(col2, axis=1, inplace=True)
    return(X_train_k,X_train)

#%%

# local knn : global k accuracy를 구하고 위의 local k accuracy에 더해주는 과정

def eval_acc(X_train):
    global_train=[]
    
    for jj in range(1,101):
        clf = knn(n_neighbors=jj)
        clf.fit(X_train, Y_train)
        global_train.append(round(clf.score(X_train,Y_train),2))
        
    for z in range(0,len(X_train_k)):
        X_train_k.iloc[z,:]=X_train_k.iloc[z,:]+global_train
    
    return(X_train_k)

#%%

# local knn : global acc와 local acc의 합을 통해 각 train sample 마다 최적의 k값을 추출하는 과정

def sample_k(X_train_k):
    sample_k=[]
    for zz in range(0,len(X_train_k)):
        inx=list(X_train_k.columns) 
        a=list(X_train_k.iloc[zz,:])
        aa=a.index(max(a))
        sample_k.append(inx[aa][0:1])
    
    X_train_k['sample_k']=sample_k
    
    return(X_train_k)

#%%

# local knn : test 데이터가 들어왔을 때 가장 거리가 가까운 3개의 train sample들의 local k 값을 확인 하고 3개중 다수의 k값을 할당하는 과정

def test_k(X_test, X_train, X_train_k):
    col=list(X_train.columns)
    
    k3=[]
    for k in range(0,len(X_test)):
        point=list(X_test.iloc[k,0:len(col)])
        X_train['dist'] = pairwise_distances([point],X_train,metric='euclidean')[0] 
        sort_train = X_train.sort_values(by=['dist'],axis=0)
        kk=sort_train.iloc[range(3)]
        kk_inx=list(kk.index)
        
        test_k=[]
        X_train=X_train.iloc[:,0:len(col)]
        for j in (kk_inx):
            test_k.append(X_train_k.loc[j]['sample_k'])
            
        cnt=Counter(test_k)
        mode=cnt.most_common(1)
        mode1=mode[0][0]
        k3.append(mode1)
        
    X_test['test_k']=k3
    return(X_test,X_train,X_train_k)

#%%
    
# local knn : 위의 과정들을 종합하여 test data에 대한 최종 class를 도출하는 과정   
    
def fin_class(X_test, X_train ):
    col=list(X_train.columns)
    fin_class=[]
    global weight
    for k in range(0,len(X_test)):
        point=list(X_test.iloc[k,0:len(col)])
        X_train['dist'] = [weighted_l2(weight, point, np.array(sample)) for idx, sample in X_train.iterrows()]
        sort_train = X_train.sort_values(by=['dist'],axis=0)
        kk=sort_train.iloc[range(int(X_test['test_k'].iloc[k]))]
        kk_inx=list(kk.index)
        
        local_class=[]
        X_train=X_train.iloc[:,0:len(col)]
        
        for j in (kk_inx):
            local_class.append(Y_train.loc[j])
        
        cnt=Counter(local_class)
        mode=cnt.most_common(1)
        mode1=mode[0][0]
        fin_class.append(mode1)    
            
    X_test['fin_class']=fin_class
    return(X_test, X_train)

#%%
# iris local Knn

balanced_data = pd.read_csv('imbalanced.csv')

X=balanced_data.iloc[:,0:-1]
Y=balanced_data.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1)
normal_knn_output = KNN(X_train, Y_train, X_test, Y_test, label=[0,1])

weight = feature_weight(X_train, Y_train).get_weight()



X_train_k=X_train


X_train_k, X_train = local_k_acc(X_train,X_train_k, Y_train)

X_train_k=eval_acc(X_train)

X_train_k=sample_k(X_train_k)

X_test,X_train,X_train_k = test_k(X_test, X_train, X_train_k)

X_test, X_train = fin_class(X_test, X_train)



print('normal f1_score :{}'.format(f1_score(normal_knn_output, Y_test)))
print('normal precision_score :{}'.format(precision_score(normal_knn_output, Y_test)))
print('normal confusion matrix : {}'.format(confusion_matrix(normal_knn_output, Y_test).T))
print('-----------------------------------------------------------------------------')
print('LKWkNN f1_score : {}'.format(f1_score(X_test['fin_class'], Y_test)))
print('LKWkNN precision_score : {}'.format(precision_score(X_test['fin_class'], Y_test)))
print('LKWkNN confusion matrix : {}'.format(confusion_matrix(X_test['fin_class'], Y_test).T))
confusion_matrix(X_test['fin_class'], Y_test, labels=[0,1])



