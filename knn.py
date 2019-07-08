# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:34:45 2019

@author: dell
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def knnDemo(X,y,n):
    # creates the classifier and fits it to the data
    res = 0.05
    k1 = knn(n_neighbors=n,p=2,metric='minkowski')
    k1.fit(X,y)
    
    # sets up the grid to plot the decision bundary
    x1_min,x1_max = X[:,0].min() -1,X[:,0].max() + 1
    x2_min,x2_max = X[:,1].min() -1,X[:,1].max() + 1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,res),np.arange(x2_min,x2_max,res))
    # make the decision bundary prediction
    z = k1.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    
    # creats the color map
    camp_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
    camp_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
    
    # plot the decision surface
    plt.contourf(xx1,xx2,z,alpha = 0.4,cmap = camp_light)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    # plot the samples
    for idx,c1 in enumerate(np.unique(y)):
        plt.scatter(X[:,0],X[:,1],c=y,cmap= camp_bold)
    
    plt.show()                            
    
                            

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
X1 = iris_x[:,0:3:2]
X2 = iris_x[:,0:2]
X3 = iris_x[:,1:3]
knnDemo(X2,iris_y,15)