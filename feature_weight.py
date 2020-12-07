# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:22:29 2020

@author: Injo Kim

Seoultech 2020-autumn semeseter 
Advacned Mahicne Learning term project 
Local Feature Weighted kNN 
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn.metrics import f1_score, precision_score
from collections import Counter

#%%
class feature_weight :
    def __init__(self, data, target) :
        self.data, self.target = data, target

    def global_accuracy(self) :
        """
        Calculate Global Knn accuracy
        """
        self.original_acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), self.data, self.target, cv=5).mean()
        
        
    def feature_iteration(self) :
        """
        Calculate each accuarcy that remove one column
        """
        self.global_accuracy()
        self.weight = []
        
        weight_append = self.weight.append
        
        for col in self.data :
            #remove one column
            column = list(self.data.columns)
            column.remove(col)
            
            #make new dataset that remove one column
            adjusted_data = self.data[column]
            weight_append((cross_val_score(KNeighborsClassifier(n_neighbors=5), adjusted_data, self.target, cv=5).mean()))
            
    def get_weight(self) :
        """
        Calculate weight using paper
        """
        self.feature_iteration()
        for idx in range(len(self.weight)) :
            self.weight[idx] = 1-(self.weight[idx]-self.original_acc)
            
        sum_of_disc = sum(self.weight)
        for idx in range(len(self.weight)) :
            self.weight[idx] = self.weight[idx]/sum_of_disc
            
        return self.weight
    
#%%
class weighted_distance :
    def __init__(self, data, target) :
        self.data = data
        self.target = target
        self.k = 5
        self.answer_index = []
        self.weight = feature_weight(self.data, self.target).get_weight()
        
    def make_train_test(self) :
        global seed
        X_train, X_test , y_train, y_test = train_test_split(self.data, self.target, random_state=seed)
        self.train_index, self.test_index = X_train.index, X_test.index 
        return self.train_index, self.test_index
        
        
    def weighted_l2(self, row0, row1) :
        """
        calculate weighted euclidean distance using feature weight
        """
        sub = row0-row1
        return np.sqrt((self.weight*sub*sub).sum())  
    

    def weighted_distance_matrix(self) :
        """ 
        get symmetric distance matrix using weighted eculidean distance
        """
        self.make_train_test()
        
        dist_matrix = np.zeros((len(self.test_index), len(self.train_index)))
        mat_data = np.array(self.data)
        
        # Calculate the weighted distance for each sample to create a symmetric matrix
        for idx_0, value_0 in enumerate(mat_data[self.test_index]) :
            for idx_1, value_1 in enumerate(mat_data[self.train_index]) :
                dist_matrix[idx_0, idx_1] = self.weighted_l2(value_0, value_1)
                
        dist_matrix = pd.DataFrame(dist_matrix, index=self.test_index, columns=self.train_index)
        

        return dist_matrix

    
    def class_determination(self) :
        distance_table = self.weighted_distance_matrix()
        
        final = [list(sample.sort_values()[0:self.k].index) for idx, sample in distance_table.iterrows()] 
        
        class_vector = pd.DataFrame([list(self.target[idx]) for idx in final], index=self.test_index)
        return class_vector
    
    
    def predict(self, data= None) :
        if len(data) == None :
          data = self.class_determination()
        
        else : 
          None

        prediction = [Counter(sample).most_common(n=1)[0][0] for idx, sample in data.iterrows()]
        prediction = pd.DataFrame(prediction, index=self.test_index)
      
        return prediction
    
#%%
def data_select(name) :
    if name == 'iris' :
        dataset = datasets.load_iris()
        data, target = pd.DataFrame(dataset.data, columns=dataset.feature_names), dataset.target
        
    else  :
        dataset = pd.read_csv('data/'+ name +'.csv')
        data, target = dataset.iloc[:,0:-1], dataset.iloc[:,-1]
        
    return data, target
        
#%%
if __name__ == '__main__':

    """
    Only touch
    """
    data_name = 'blood'
    seed = 42

    """
    normal kNN
    """
    data, target = data_select(data_name)
    knn = KNeighborsClassifier(n_neighbors=5)
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=seed)
    knn.fit(X_train, y_train)
    
    if data_name == 'iris' :
        print('normal precision socre : {}'.format(precision_score(knn.predict(X_test), y_test, average='macro')))
        print('normal f1 socre : {}'.format(f1_score(knn.predict(X_test), y_test, average='macro')))        
    else :
        print('normal precision socre : {}'.format(precision_score(knn.predict(X_test), y_test)))
        print('normal f1 socre : {}'.format(f1_score(knn.predict(X_test), y_test)))
    
    
    """
    DCT kNN
    """
    wd = weighted_distance(data, target)
    predict_vote = wd.class_determination()
    predict = wd.predict(predict_vote)
    
    if data_name =='iris' :
        print('DCT precision socre : {}'.format(precision_score(predict, y_test, average='macro')))
        print('DCT f1 socre : {}'.format(f1_score(predict, y_test, average='macro')))
    else :
        print('DCT precision socre : {}'.format(precision_score(predict, y_test)))
        print('DCT f1 socre : {}'.format(f1_score(predict, y_test)))
