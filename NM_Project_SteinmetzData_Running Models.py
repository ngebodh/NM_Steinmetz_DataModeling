# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:20:21 2020

@author: TheBeast
"""



# =============================================================================
#                         Clear Variables
# =============================================================================

# Clear variables before running
from IPython import get_ipython
get_ipython().magic('reset -sf')


# =============================================================================
#                       Import Libraries 
# =============================================================================


import numpy as np
import os
import scipy
from scipy import signal
import os, requests
import plotly.express as px
from plotly.offline import plot

from matplotlib import rcParams 
from matplotlib import pyplot as plt



from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# =============================================================================
#                   Import Feature Matrix
# =============================================================================
# 'D:\Neuromatch\Project\Analysis' 'NM_Steinmetz_Features.mat'
feat_mat_loc= r'D:\Neuromatch\Project\Analysis\NM_Steinmetz_Features.mat'

# feat_mat_loc_imp=os.listdir(feat_mat_loc)

from scipy.io import loadmat
Feature_Mat = loadmat(feat_mat_loc)

feat_names=[]
for i in range(np.size(Feature_Mat['All_feat_name'])):
    feat_names.append(Feature_Mat['All_feat_name'][0][i][0])
    

# =============================================================================
#               Train Test Split
# =============================================================================

# del X, y, y_train, y_test
feat_pick =[1,5,9,13,17,21,25,29,33,37,41]  # Visual Only
# feat_pick =[2,6,10,14,18,22,26,30,34,38,42] # Thala
feat_pick =[4,8,12,16,20,24,28,32,36,40,44] # Motor
# feat_pick =[17,21,25,29,37] 
# X = Feature_Mat['All_feat_cat'][:,feat_pick]


X = Feature_Mat['All_feat_cat'][:,0:-2]
y= Feature_Mat['All_feat_cat'][:,-1]

print(f"Shape of X mat: {X.shape}")
print(f"Shape of Y mat: {y.shape}")
indx=[y!=-2]
# y= y[indx]
# X =X[indx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)





# =============================================================================
#                     SVM Classifier
# =============================================================================


clf = svm.SVC(kernel='rbf', C=1)
scores = cross_val_score(clf, X, y, cv=10)
scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# clf.score(X_test, y_test)




#print(taskname)
num_iteration=100
from sklearn.pipeline import make_pipeline
import csv
import os
import numpy as np
import sys
import math
import datetime
from sklearn import preprocessing, svm
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

import random as rn
from sklearn.model_selection import GridSearchCV 


# param_grid = {'C': [0.1, 1, 10, 100, 1000],  
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#               'kernel': ['rbf']}  

# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
# grid = make_pipeline(preprocessing.StandardScaler(),svm.SVC())

# grid.fit(X_train, y_train) 

# grid_predictions = grid.predict(X_test) 
# print(classification_report(y_test, grid_predictions)) 




p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state= 1)
    clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1, kernel = 'rbf'))
    scores1 = cross_val_score(clf, X_train, y_train, cv=cv)

    print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
    p_task_svm = scores1.mean() + p_task_svm

    acc_list_task.append(scores1.mean()*100)
    std_list_task.append(scores1.std() * 100)

print(" \n Score total " + taskname +" using SVM: " + repr(p_task_svm/num_iteration) )  
print(acc_list_task)
print(f"Mean accuracy over iterations:{np.round(np.mean(acc_list_task),3)}% (+/- {np.round(np.std(acc_list_task),3)})")



### Shuffle the targets 

# y_train_shuffle = y_train

# np.random.shuffle(y_train_shuffle)

# AA=rn.sample(list(y_train), len(list(y_train)))
p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state= 1)
    clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1, kernel = 'rbf'))
    scores1 = cross_val_score(clf, X_train,rn.sample(list(y_train), len(list(y_train))), cv=cv)

    print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
    p_task_svm = scores1.mean() + p_task_svm

    acc_list_task.append(scores1.mean()*100)
    std_list_task.append(scores1.std() * 100)

print(" \n Score total " + taskname +" using SVM: " + repr(p_task_svm/num_iteration) )  
print(acc_list_task)
print(f"Mean accuracy over iterations:{np.round(np.mean(acc_list_task),3)}% (+/- {np.round(np.std(acc_list_task),3)})")






# =============================================================================
#                   Decision Tree Classsifier
# =============================================================================

from sklearn.tree import DecisionTreeClassifier

p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=3, test_size=0.25, random_state= 1)
    clf = make_pipeline(preprocessing.StandardScaler(), DecisionTreeClassifier())
    scores1 = cross_val_score(clf, X_train, y_train, cv=cv)

    print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
    p_task_svm = scores1.mean() + p_task_svm

    acc_list_task.append(scores1.mean()*100)
    std_list_task.append(scores1.std() * 100)

print(" \n Score total " + taskname +" using SVM: " + repr(p_task_svm/num_iteration) )  
print(acc_list_task)
print(f"Mean accuracy over iterations:{np.round(np.mean(acc_list_task),3)}% (+/- {np.round(np.std(acc_list_task),3)})")






### Shuffle the targets 

p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=3, test_size=0.25, random_state=1)
    clf = make_pipeline(preprocessing.StandardScaler(), DecisionTreeClassifier())
    scores1 = cross_val_score(clf, X_train, rn.sample(list(y_train), len(list(y_train))), cv=cv)

    print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
    p_task_svm = scores1.mean() + p_task_svm

    acc_list_task.append(scores1.mean()*100)
    std_list_task.append(scores1.std() * 100)

print(" \n Score total " + taskname +" using SVM: " + repr(p_task_svm/num_iteration) )  
print(acc_list_task)
print(f"Mean accuracy over iterations:{np.round(np.mean(acc_list_task),3)}% (+/- {np.round(np.std(acc_list_task),3)})")




# =============================================================================
#                   Random Forest Classifier
# =============================================================================

from sklearn.ensemble import RandomForestClassifier

p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state=1)
    clf = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier())
    scores1 = cross_val_score(clf, X_train, y_train, cv=cv)

    print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
    p_task_svm = scores1.mean() + p_task_svm

    acc_list_task.append(scores1.mean()*100)
    std_list_task.append(scores1.std() * 100)

print(" \n Score total " + taskname +" using SVM: " + repr(p_task_svm/num_iteration) )  
print(acc_list_task)
print(f"Mean accuracy over iterations:{np.round(np.mean(acc_list_task),3)}% (+/- {np.round(np.std(acc_list_task),3)})")




p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state=1)
    clf = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier())
    scores1 = cross_val_score(clf, X_train, rn.sample(list(y_train), len(list(y_train))), cv=cv)

    print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
    p_task_svm = scores1.mean() + p_task_svm

    acc_list_task.append(scores1.mean()*100)
    std_list_task.append(scores1.std() * 100)

print(" \n Score total " + taskname +" using SVM: " + repr(p_task_svm/num_iteration) )  
print(acc_list_task)
print(f"Mean accuracy over iterations:{np.round(np.mean(acc_list_task),3)}% (+/- {np.round(np.std(acc_list_task),3)})")


# =============================================================================
#                         AdaBoost Classifier
# =============================================================================

from sklearn.ensemble import  AdaBoostClassifier


clf = AdaBoostClassifier(n_estimators=3)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state= 1)
    clf = make_pipeline(preprocessing.StandardScaler(), AdaBoostClassifier( DecisionTreeClassifier(),
    n_estimators=600,))
    scores1 = cross_val_score(clf, X_train, y_train, cv=cv)

    # print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
    p_task_svm = scores1.mean() + p_task_svm

    acc_list_task.append(scores1.mean()*100)
    std_list_task.append(scores1.std() * 100)

print(" \n Score total " + taskname +" using SVM: " + repr(p_task_svm/num_iteration) )  
print(acc_list_task)
print(f"Mean accuracy over iterations:{np.round(np.mean(acc_list_task),3)}% (+/- {np.round(np.std(acc_list_task),3)})")




### Shuffle the targets 

clf = AdaBoostClassifier(n_estimators=3)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state=1)
    clf = make_pipeline(preprocessing.StandardScaler(), AdaBoostClassifier( DecisionTreeClassifier(),
    n_estimators=600,))
    scores1 = cross_val_score(clf, X_train, rn.sample(list(y_train), len(list(y_train))), cv=cv)

    # print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
    p_task_svm = scores1.mean() + p_task_svm

    acc_list_task.append(scores1.mean()*100)
    std_list_task.append(scores1.std() * 100)

print(" \n Score total " + taskname +" using SVM: " + repr(p_task_svm/num_iteration) )  
print(acc_list_task)
print(f"Mean accuracy over iterations:{np.round(np.mean(acc_list_task),3)}% (+/- {np.round(np.std(acc_list_task),3)})")









# =============================================================================
#                         Multi-layer perceptron (MLP) 
# =============================================================================

from sklearn.neural_network import MLPClassifier

p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state= 1)
    clf = make_pipeline(preprocessing.StandardScaler(),
                        MLPClassifier(solver='adam', alpha=1e-5,max_iter=900,
                        hidden_layer_sizes=(5,4), random_state=1))
    scores1 = cross_val_score(clf, X_train, y_train, cv=cv)

    # print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
    p_task_svm = scores1.mean() + p_task_svm

    acc_list_task.append(scores1.mean()*100)
    std_list_task.append(scores1.std() * 100)

print(" \n Score total " + taskname +" using SVM: " + repr(p_task_svm/num_iteration) )  
print(acc_list_task)
print(f"Mean accuracy over iterations:{np.round(np.mean(acc_list_task),3)}% (+/- {np.round(np.std(acc_list_task),3)})")




clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(100, 10, 5,2), random_state=1)
scores = cross_val_score(clf, X_train, y_train, cv=10)
scores

clf.fit(X, y)




# =============================================================================
#               Try Deep Net 
# =============================================================================





# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
from keras.layers import Dropout
from keras.utils import normalize


y_train_deep = to_categorical(y_train,num_classes=4)
y_test_deep = to_categorical(y_test,num_classes=4)

X_train_deep = normalize(X_train, axis=1)
X_test_deep = normalize(X_test, axis=1)



model = Sequential()
model.add(Dense(6000, activation='relu', input_dim=len(X_train[0,:])))
model.add(Dropout(0.5))
# model.add(Dense(80, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(70, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(5000, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(4000, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4000, activation='relu'))
# model.add(Dropout(0.5))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))

# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))

model.add(Dense(4, activation="softmax"))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# build the model
history = model.fit(X_train_deep, y_train_deep, validation_data=(X_test_deep,y_test_deep), epochs=150)


pred_train= model.predict(X_train_deep)
scores = model.evaluate(X_train_deep,y_train_deep, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
scores = model.evaluate(X_train_deep,y_train_deep, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 

# =============================================================================
# 
# =============================================================================


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA




fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()






from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2


target_names=feat_names
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()


































