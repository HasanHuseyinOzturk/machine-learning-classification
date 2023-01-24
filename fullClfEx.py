# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn.datasets import load_iris

irisData = load_iris()#Veri setini değişkene ekledik
X = irisData.data#Giris verileri
y = irisData.target#Sınıfı yani çıkış değerleri

def ml(X, y, model=[]):   
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
    
    for i in range(len(model)):
        if(model[i] == MLPClassifier):
            models = model[i](max_iter=5000)
        elif(model[i] == DecisionTreeClassifier):
            models = model[i](max_depth=5)
        elif(model[i] == KNeighborsClassifier):
            models = model[i](n_neighbors=10, n_jobs=-1)
        elif(model[i] == RandomForestClassifier):
            models = model[i](random_state=0, n_estimators=5000)
        elif(model[i] == GradientBoostingClassifier):
            models = model[i](random_state=0, n_estimators = 5000, learning_rate=0.1)
        else:
            models = model[i](random_state=0,n_estimators = 5000)
            
        models.fit(X_train, y_train)
        y_pred = models.predict(X_test)
        rmse = (np.sqrt(MSE(y_test, y_pred)))
        print("RMSE:{} ".format(models),rmse)
        
        
model = [DecisionTreeClassifier,
          KNeighborsClassifier,
          MLPClassifier,
          RandomForestClassifier,
          GradientBoostingClassifier]


ml(X, y, model=model)
