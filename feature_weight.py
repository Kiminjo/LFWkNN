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
import time

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
        
        for col in self.data :
            #remove one column
            column = list(self.data.columns)
            column.remove(col)
            
            #make new dataset that remove one column
            adjusted_data = self.data[column]
            self.weight.append(cross_val_score(KNeighborsClassifier(n_neighbors=5), adjusted_data, self.target, cv=5).mean())
            
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


class weighted_distance :
    def __init__(self, data, target) :
        self.data = data
        self.target = target
        self.k = 5
        
    def make_train_test(self) :
        X_train, X_test , y_train, y_test = train_test_split(self.data, self.target)
        train_index, test_index = X_train.index, X_test.index 
        return train_index, test_index
        
        
    def weighted_l2(self, row0, row1) :
        """
        calculate weighted euclidean distance using feature weight
        """
        weight = feature_weight(self.data, self.target).get_weight()
        sub = row0-row1
        return np.sqrt((weight*sub*sub).sum())  
    

    def weighted_distance_matrix(self) :
        """ 
        get symmetric distance matrix using weighted eculidean distance
        """
        
        dist_matrix = np.zeros((len(self.target), len(self.target)))
        mat_data = np.array(self.data)
        
        # Calculate the weighted distance for each sample to create a symmetric matrix
        for idx_0, value_0 in enumerate(mat_data) :
            for idx_1, value_1 in enumerate(mat_data) :
                dist_matrix[idx_0, idx_1] = self.weighted_l2(value_0, value_1)
                
        return dist_matrix
    
    def split_matrix(self) :
        dist_mat = self.weighted_distance_matrix()
        train_index, test_index = self.make_train_test()
        train_dist_matrix = dist_mat[train_index]
        
        return train_dist_matrix
    
#    def class_determination(self) :


iris = datasets.load_iris()
data, target = pd.DataFrame(iris.data, columns=iris.feature_names), iris.target

start = time.time()
wd = weighted_distance(data, target)
dist_matrix = wd.weighted_distance_matrix()
print('executive time : {}'.format(time.time()-start))


