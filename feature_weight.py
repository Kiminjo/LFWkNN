# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:22:29 2020

@author: Injo Kim

Seoultech 2020-autumn semeseter 
Advacned Mahicne Learning term project 
Local Feature Weighted kNN 
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class feature_weight :
    def __init__(self, data, target) :
        self.data, self.target = data, target

    def global_accuracy(self) :
        """
        Calculate Global Knn accuracy
        """
        global_X_train, global_X_val, global_y_train, global_y_val = train_test_split(self.data, self.target)
    
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(global_X_train, global_y_train)
        self.original_acc = accuracy_score(knn.predict(global_X_val), global_y_val)
        
        
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
            X_train, X_val, y_train, y_val = train_test_split(adjusted_data, self.target)
            
            #learning using Knn
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)
            self.weight.append(accuracy_score(knn.predict(X_val), y_val))
            
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

                


