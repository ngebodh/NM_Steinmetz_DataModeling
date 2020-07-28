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

# del X, y
X = Feature_Mat['All_feat_cat'][:,0:-2]
# X = Feature_Mat['All_feat_cat'][:,0:-2]
y= Feature_Mat['All_feat_cat'][:,-1]

print(f"Shape of X mat: {X.shape}")
print(f"Shape of Y mat: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)




# =============================================================================
#                     SVM Classifier
# =============================================================================


clf = svm.SVC(kernel='rbf', C=0.75)
scores = cross_val_score(clf, X, y, cv=10)
scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# clf.score(X_test, y_test)




#print(taskname)
num_iteration=3
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



p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state= None)
    clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1, kernel = 'rbf'))
    scores1 = cross_val_score(clf, X_train, y_train, cv=cv)

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
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state= None)
    clf = make_pipeline(preprocessing.StandardScaler(), DecisionTreeClassifier())
    scores1 = cross_val_score(clf, X_train, y_train, cv=cv)

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
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state= None)
    clf = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier())
    scores1 = cross_val_score(clf, X_train, y_train, cv=cv)

    # print("Accuracy: %0.3f (+/- %0.3f)" % (scores1.mean(), scores1.std()))
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

p_task_svm = 0
acc_list_task = []
std_list_task = []
taskname='Random'
for iteration in range(num_iteration):
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state= None)
    clf = make_pipeline(preprocessing.StandardScaler(), AdaBoostClassifier( DecisionTreeClassifier(max_depth=12),
    n_estimators=600,))
    scores1 = cross_val_score(clf, X_train, y_train, cv=cv)

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
    cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state= None)
    clf = make_pipeline(preprocessing.StandardScaler(),
                        MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5,2), random_state=1))
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


y_train = to_categorical(y_train)


# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
from keras.layers import Dropout



model = Sequential()
model.add(Dense(500, activation='relu', input_dim=32))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# build the model
history = model.fit(X_train, y_train, epochs=50)


pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 





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


































