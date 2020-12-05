# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:26:27 2020

@author: Injo Kim

Seoultech 2020-atumn semester
Advnanced Machine learning term project
Local feature weighted kNN
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from feature_weight import feature_weight
import time


class Get_local_k :
    def __init__(self) :
        iris = datasets.load_iris()
        self.data, self.target = pd.DataFrame(iris.data, columns=iris.feature_names), iris.target
        self.k_list = [1,3,5,7,9,11]
        self.global_acc_ls = []
        self.local_k_df = []
        
        
    def global_acc(self) :
        X_train, X_val, y_train, y_val = train_test_split(self.data, self.target)
        
        for k in self.k_list :
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            self.global_acc_ls.append(accuracy_score(knn.predict(X_val), y_val))
            
            
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
        self.global_acc()
        
        dist_matrix = np.zeros((len(self.target), len(self.target)))
        mat_data = np.array(self.data)
        
        # Calculate the weighted distance for each sample to create a symmetric matrix
        for idx_0, value_0 in enumerate(mat_data) :
            for idx_1, value_1 in enumerate(mat_data) :
                dist_matrix[idx_0, idx_1] = self.weighted_l2(value_0, value_1)
                
        for k_idx, k in enumerate(self.k_list) :
            acc_list = []
            
            # select k points with small weighted distance for each sample
            for idx_0, row in enumerate(dist_matrix) :
                idx_list = []
                value_list =[]
                
                for idx_1, value in enumerate(row) :
                    idx_list.append(idx_1)
                    value_list.append(value)
                    
                sort_df = pd.DataFrame((idx_list, value_list), index=['index', 'value']).T
                sort_df = sort_df.sort_values(by=['value'])
                sort_df = list(sort_df.iloc[1:k+1, :].index)

                acc_list.append((self.target[idx_0]== [self.target[i] for i in sort_df]).mean())
            
            # combine local k accuracy and global k accuracy 
            self.local_k_df.append(acc_list+self.global_acc_ls[k_idx])
            
        self.local_k_df = pd.DataFrame(np.array(self.local_k_df).T, columns=['k=1', 'k=3', 'k=5', 'k=7', 'k=9', 'k=11'])
        
        
    def get_train_k(self) :
        """
        get train samples local k 
        """
        self.weighted_distance_matrix()
        
        random.seed(0)
        val_idx = random.sample(list(self.local_k_df.index), int(len(self.local_k_df)*0.3))

        train_idx = list(self.local_k_df.index)
        for idx in val_idx :
            train_idx.remove(idx)
            
        train_local_k_df, test_local_k_df = self.local_k_df.iloc[train_idx, :], self.local_k_df.iloc[val_idx, :]
        
        train_local_k_df['fitted_k'] = 0

        for idx, row in train_local_k_df.iterrows() :
            row = list(row)
            train_local_k_df.loc[idx, 'fitted_k'] = self.k_list[row.index(max(row))]
            
        # Run up to 480seconds(almost 8 min) at this point
        return train_local_k_df
         
    

    

lk = Get_local_k()

start = time.time()
local_k = lk.get_train_k()
print('실행시간 :{}'.format(time.time()-start))