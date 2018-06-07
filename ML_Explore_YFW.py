# -*- coding: utf-8 -*-
########################################################################
#import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pydot
import graphviz
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
#########################################################################
#import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
import numpy as np
import pydot
import graphviz
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
#################################  Data import  + (Manipualtion)  ################################

# data import

flt = DataFrame(pd.read_excel(fl1,header=None))

# Split into Train & Test dataset
X = flt.iloc[:,0:57]
Y = flt.iloc[:,57]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#################################  Explore ML methods  ################################
'''   Pros & Cons
1. KNN :      
2. Logistic : less sensitive to sample size
3. Decision Tree: senstive to sample size
4. SVM 
5. NB
'''

'''Comparing different models at a time 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()




'''
#################################  Decision Tree  ################################
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
tuned_parameters = {"criterion": ["gini", "entropy"],
              "min_samples_split": list(range(2,11)),
              "max_depth": list(range(1,11)),
              "min_samples_leaf": list(range(1,11)),
              "max_leaf_nodes": list(range(2,11)),
              }
clf = tree.DecisionTreeClassifier()
grid = GridSearchCV(estimator= clf, param_grid=tuned_parameters)
grid.fit(X_train, y_train)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_params_)
#{'min_samples_split': 2, 'max_leaf_nodes': 6, 'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 10}

clf_best = tree.DecisionTreeClassifier(**grid.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)

# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()

# Precision, Recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#################################  Logistic Regression  ################################

from sklearn import linear_model

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

# We start with initializing our classifier.
clf = linear_model.LogisticRegression()
# Declare parameters to tune
penalty = ['l1','l2']
c = [0.001,0.01,0.1,1,10,100,1000]
parameters = {'penalty': penalty,'C':c }  
# Tune model using cross-validation ()
nfold = 3
clf = GridSearchCV(clf, parameters, cv=nfold)
clf.fit(X_train, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

clf_best = linear_model.LogisticRegression(**clf.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)

# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()

# Precision, Recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 

#################################       KNN         ################################
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
 # KNN - Binary
# Create the knn model.
# Look at 3 closest neighbors.

from sklearn.neighbors import KNeighborsClassifier
cv_scores = []
for k in list(range(1,11,1)):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
print cv_scores   # We find K = 3 providing the best results

knn = KNeighborsClassifier(n_neighbors=(cv_scores.index(max(cv_scores))+1))

y_pred = knn.fit(X_train, y_train).predict(X_test)
score = knn.score(X_test, y_test)

# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
knn_matrix = confusion_matrix(y_test, y_pred.round())
print 'Confusion matrix:\n',knn_matrix
#tn, fp, fn, tp = knn_matrix.ravel().ravel()
# Precision, Recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred.round())) 

#################################    SVM        ################################
from sklearn import svm
'''
Comparing different SVM kernels - 
http://scikit-learn.org/stable/auto_examples/exercises/plot_iris_exercise.html
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()
'''

# Find the optimized model 
from sklearn.svm import SVR
import numpy as np
parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
svr = svm.SVR()
clf = GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
clf.best_params_

print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

clf_best = linear_model.LogisticRegression(**clf.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)

# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()

# Precision, Recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#################################      Naive bayes         ################################
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
y_pred = clf.fit(X_train, y_train).predict(X_test)
score = clf.score(X_test, y_test)

# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()

# Precision, Recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#################################      impact of various data pre-processing techniques - KNN + Normalization       ################################
'''
1. 
KNN - Normalization
Logistic - Normalization
2. PCA 
'''
from sklearn.datasets import load_breast_cancer
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals.six import StringIO
from sklearn.grid_search import GridSearchCV 
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
import numpy as np
import pydot
import os
# Apply standardization (or Z-score normalization) to features, you could compare the performance between w/o normalization
# Apply Scaling to X_train and X_test
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

knn = neighbors.KNeighborsClassifier()

# Declare parameters to tune
n_neighbors = list(range(1,21))
distances = ['euclidean','manhattan','jaccard']
parameters = {'n_neighbors': n_neighbors,'metric': distances}  
# Tune model using cross-validation ()
nfold = 10
clf = GridSearchCV(knn, parameters, cv=nfold)
clf.fit(X_train_std, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_


# Now you can predict new values using best model derived from cross validation and test your performance on the test set
y_pred = clf.predict(X_test_std)
score = clf.score(X_test_std, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
# output classification report
print 'Classification report:\n',classification_report(y_test, y_pred)

############### 
tn, fp, fn, tp = clf_matrix.ravel(); print 'tn, fp, fn, tp: ', tn, fp, fn, tp 
sum = tn + fp + fn + tp 
print 'fp expected cost(1), fn expected cost(10): \n', fp/float(sum) * 1 , fn/float(sum) * 10 



#################################      impact of various data pre-processing techniques - Logistic Regression  + Normalization    ################################

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

clf = linear_model.LogisticRegression()
# Declare parameters to tune
penalty = ['l1','l2']
c = [0.001,0.01,0.1,1,10,100,1000]
parameters = {'penalty': penalty,'C':c }  
# Tune model using cross-validation ()
nfold = 3
clf = GridSearchCV(clf, parameters, cv=nfold)
clf.fit(X_train_std, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

# Now you can predict new values using best model derived from cross validation and test your performance on the test set
y_pred = clf.predict(X_test_std)
score = clf.score(X_test_std, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
# output classification report
print 'Classification report:\n',classification_report(y_test, y_pred)
#########
tn, fp, fn, tp = clf_matrix.ravel(); print 'tn, fp, fn, tp: ', tn, fp, fn, tp 
sum = tn + fp + fn + tp 
print 'fp expected cost(1), fn expected cost(10): \n', fp/float(sum) * 1 , fn/float(sum) * 10


#################################      impact of various data pre-processing techniques - SVM  + PCA    ################################

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation
pca = PCA(n_components=20)# adjust yourself
pca.fit(X_train)
X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)
clf = SVC()
clf.fit(X_t_train, y_train)
print 'score', clf.score(X_t_test, y_test)
print 'pred label', clf.predict(X_t_test)
# Now you can predict new values using best model derived from cross validation and test your performance on the test set
y_pred = clf.predict(X_t_test)
score = clf.score(X_t_test, y_test)
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
# output classification report
print 'Classification report:\n',classification_report(y_test, y_pred)
########################
tn, fp, fn, tp = clf_matrix.ravel(); print 'tn, fp, fn, tp: ', tn, fp, fn, tp 
sum = tn + fp + fn + tp 
print 'fp expected cost(1), fn expected cost(10): \n', fp/float(sum) * 1 , fn/float(sum) * 10

#################################     building cost-sensitive prediction models(10:1)       ################################
'''
http://nbviewer.jupyter.org/github/albahnsen/CostSensitiveClassification/blob/master/doc/tutorials/tutorial_edcs_credit_scoring.ipynb
'''
# Load classifiers and split dataset in training and testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Fit the classifiers using the training dataset
classifiers = {"KN": {"f": KNeighborsClassifier()},
               "DT": {"f": DecisionTreeClassifier()},
               "LR": {"f": LogisticRegression()},
               "NB": {"f": BernoulliNB()}}

for model in classifiers.keys():
    # Fit
    classifiers[model]["f"].fit(X_train, y_train)
    # Predict
    classifiers[model]["c"] = classifiers[model]["f"].predict(X_test)
    classifiers[model]["p"] = classifiers[model]["f"].predict_proba(X_test)
    classifiers[model]["p_train"] = classifiers[model]["f"].predict_proba(X_train)

# Evaluate the performance
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
measures = {"f1": f1_score, "pre": precision_score, 
            "rec": recall_score, "acc": accuracy_score}
results = pd.DataFrame(columns=measures.keys())

# Evaluate each model in classifiers
for model in classifiers.keys():
    results.loc[model] = [measures[measure](y_test, classifiers[model]["c"]) for measure in measures.keys()]

print results





#################################     Use best practices when evaluating the models       ################################
'''
(accuracy, precision, recall, f-measure, AUC, average misclassification cost…), 
show some visual indications of model performance (ROC curves, lift charts).
'''

from sklearn import linear_model

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

# We start with initializing our classifier.
clf = linear_model.LogisticRegression()
# Declare parameters to tune
penalty = ['l1','l2']
c = [0.001,0.01,0.1,1,10,100,1000]
parameters = {'penalty': penalty,'C':c }  
# Tune model using cross-validation ()
nfold = 3
clf = GridSearchCV(clf, parameters, cv=nfold)
clf.fit(X_train, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

clf_best = linear_model.LogisticRegression(**clf.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)

# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()

# Precision, Recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(np.array(y_test), y_pred)
roc_auc = auc(fpr, tpr)
print("The Area Under the ROC Curve : %f" % roc_auc)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve'+'\n'+ttl)
plt.legend(loc="lower right")
plt.show()


# LIFT Contrast 
y_pred = DataFrame(knn.fit(X_train, y_train).predict_proba(X_test))[1]

lift_knn = []
x_knn = []
for i in np.arange(0,1.1,0.1):
    
    y_pred1 = np.where(y_pred >= i, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred1).ravel()
    lift_knn.append((float(tp)/(tp+fn))/(float(tp+fp)/(tn + fp + fn + tp)))
    x_knn.append(float(tp+fp)/(tn + fp + fn + tp))

y_pred_tree = DataFrame(clf_best.fit(X_train, y_train).predict_proba(X_test))[1]
lift_dt = []
x_dt = []
for i in np.arange(0,1.0,0.1):
    y_pred1 = np.where(y_pred_tree >= i, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred1).ravel()
    lift_dt.append((float(tp)/(tp+fn))/(float(tp+fp)/(tn + fp + fn + tp)))
    x_dt.append(float(tp+fp)/(tn + fp + fn + tp))
plt.title('Lift curve')
plt.plot(x_knn,lift_knn, 'b',label = 'KNN')
plt.plot(x_dt,lift_dt, 'g',label = 'Logistic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [1, 1],'r--')
plt.xlim([0, 1])
plt.ylabel('Lift')
plt.xlabel('Percentage of test instances')
plt.show()



#################################     Baggin, Boosting, Voting      ################################
https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/ 


#################################    Bagging      ################################

# Bagged Decision Trees for Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# Random Forest
import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
clf = GridSearchCV(clf, param_grid=param_grid)
clf.fit(X_train, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

clf_best = RandomForestClassifier(**clf.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()

# Precision, Recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 

################################################################# Extra Trees

# Extra Trees
import pandas
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier()
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
clf = GridSearchCV(clf, param_grid=param_grid)
clf.fit(X_train, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

clf_best = ExtraTreesClassifier(**clf.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()

# Precision, Recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 

################################################################# AdaBoost

###   Boosting   ###
# AdaBoost Classification
'''
You can tune the parameters to optimize the performance of algorithms, I’ve mentioned below the key parameters for tuning:

n_estimators: It controls the number of weak learners.
learning_rate:Controls the contribution of weak learners in the final combination. There is a trade-off between learning_rate and n_estimators.
base_estimators: It helps to specify different ML algorithm.
'''
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
param_grid = {'n_estimators': [3, 5, 8,10, 16, 32, 50, 100, 150, 200],
              'learning_rate': [0.6,0.8,0.9,1.0]}


# run grid search
clf = GridSearchCV(clf, param_grid=param_grid)
clf.fit(X_train, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

clf_best = AdaBoostClassifier(**clf.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix



################################################################# Stochastic Gradient Boosting Classification
#  Stochastic Gradient Boosting Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier()
param_grid = {'n_estimators': [3, 5, 8,10, 16, 32, 50, 100, 150, 200, 250, 300],
              'learning_rate': [0.4,0.5,0.6,0.8,0.9,1.0]}


# run grid search
clf = GridSearchCV(clf, param_grid=param_grid)
clf.fit(X_train, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

clf_best = GradientBoostingClassifier(**clf.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix



### Voting ####
#################################################
# Extra Trees
import pandas
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier()
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
clf = GridSearchCV(clf, param_grid=param_grid)
clf.fit(X_train, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

clf_best = ExtraTreesClassifier(**clf.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()

# Precision, Recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 

################################################################# AdaBoost

###   Boosting   ###
# AdaBoost Classification
'''
You can tune the parameters to optimize the performance of algorithms, I’ve mentioned below the key parameters for tuning:

n_estimators: It controls the number of weak learners.
learning_rate:Controls the contribution of weak learners in the final combination. There is a trade-off between learning_rate and n_estimators.
base_estimators: It helps to specify different ML algorithm.
'''
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
param_grid = {'n_estimators': [3, 5, 8,10, 16, 32, 50, 100, 150, 200],
              'learning_rate': [0.6,0.8,0.9,1.0]}


# run grid search
clf = GridSearchCV(clf, param_grid=param_grid)
clf.fit(X_train, y_train)
print "Best estimator found by grid search:",clf.best_estimator_
print "Best parameters found by grid search:",clf.best_params_
print clf.best_score_

clf_best = AdaBoostClassifier(**clf.best_params_)
y_pred = clf_best.fit(X_train, y_train).predict(X_test)
score = clf_best.score(X_test, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix



################################################################# Stochastic Gradient Boosting Classification
#  Stochastic Gradient Boosting Classification
'''
Briefly, gradient boosted decision trees work by sequentially fitting a series of decision trees to the data; 
each tree is asked to predict the error made by the previous trees, 
and is often trained on slightly perturbed versions of the data. 
'''
import pandas
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

clf1 = GradientBoostingClassifier()
param_grid = {'n_estimators': [3, 5, 8,10, 16, 32, 50, 100, 150, 200, 250, 300],
              'learning_rate': [0.4,0.5,0.6,0.8,0.9,1.0]}


# run grid search
clf1 = GridSearchCV(clf1, param_grid=param_grid)
clf1.fit(X_train, y_train)
print "Best estimator found by grid search:",clf1.best_estimator_
print "Best parameters found by grid search:",clf1.best_params_
print clf.best_score_

clf1_best = GradientBoostingClassifier(**clf1.best_params_)
y_pred = clf1_best.fit(X_train, y_train).predict(X_test)
score = clf1_best.score(X_test, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix

# AdaBoost Classification

import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

clf2 = AdaBoostClassifier()
param_grid = {'n_estimators': [3, 5, 8,10, 16, 32, 50, 100, 150, 200],
              'learning_rate': [0.6,0.8,0.9,1.0]}


# run grid search
clf2 = GridSearchCV(clf2, param_grid=param_grid)
clf2.fit(X_train, y_train)
print "Best estimator found by grid search:",clf2.best_estimator_
print "Best parameters found by grid search:",clf2.best_params_
print clf2.best_score_

clf2_best = AdaBoostClassifier(**clf2.best_params_)
y_pred = clf2_best.fit(X_train, y_train).predict(X_test)
score = clf2_best.score(X_test, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix

# Extra Trees
import pandas
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier

clf3 = ExtraTreesClassifier()
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
clf3 = GridSearchCV(clf3, param_grid=param_grid)
clf3.fit(X_train, y_train)
print "Best estimator found by grid search:",clf3.best_estimator_
print "Best parameters found by grid search:",clf3.best_params_
print clf3.best_score_

clf3_best = ExtraTreesClassifier(**clf3.best_params_)
y_pred = clf3_best.fit(X_train, y_train).predict(X_test)
score = clf3_best.score(X_test, y_test)
# out prediction accuracy
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()


####
'''
Best Model we have so far: Stochastic Gradient Boosting, Extra Trees, Extra Trees
'''
eclf = VotingClassifier(estimators=[('XGB', clf1_best), ('ADA', clf2_best), ('extr', clf3_best)],
                        voting='soft',
                        weights=[3, 2, 1])
results = model_selection.cross_val_score(eclf, X_test, y_test, cv=kfold)
print(results.mean())

####
eclf = VotingClassifier(estimators=[('XGB', clf1_best), ('ADA', clf2_best), ('extr', clf3_best)])
param_grid = {"voting": ['soft', 'hard'],
              "weights": [[1, 1, 1],[1,2,3],[2,3,1],[3,2,1],[0,0,1],[0,1,0],[1,0,0]]
}
eclf = GridSearchCV(eclf, param_grid=param_grid)
eclf.fit(X_train, y_train)
print "Best estimator found by grid search:",eclf.best_estimator_
print "Best parameters found by grid search:",eclf.best_params_


#eclf_best = VotingClassifier(**eclf.best_params_)

eclf_best = VotingClassifier(estimators=[('XGB', clf1_best), ('ADA', clf2_best), ('extr', clf3_best)],
                        voting='soft',
                        weights=[1, 2, 3])
results = model_selection.cross_val_score(eclf_best, X_test, y_test, cv=kfold)
y_pred = eclf_best.fit(X_train, y_train).predict(X_test)
score = eclf_best.score(X_test, y_test)
print(results.mean())
print 'Accuracy:', score
# output confusion matrix
clf_matrix = confusion_matrix(y_test, y_pred)
print 'Confusion matrix:\n',clf_matrix
#tn, fp, fn, tp = clf_matrix.ravel().ravel()
################################################# XGBoost
#https://anaconda.org/conda-forge/xgboost 
#ttps://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
'''
XGBoost on the other hand, builds really short and simple decision trees iteratively. 
Each tree is called a "weak learner" for their high bias. 
XGBoost starts by creating a first simple tree which has poor performance by itself. 
It then builds another tree which is trained to predict what the first tree was not able to, 
and is itself a weak learner too. The algorithm goes on by sequentially building more weak learners, 
each one correcting the previous tree until a stopping condition is reached, 
such as the number of trees (estimators) to build.
'''
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
                             
optimized_GBM.fit(X_train, y_train)
optimized_GBM.grid_scores_  #params: {'max_depth': 3, 'min_child_weight': 1},
#xgdmat = xgb.DMatrix(X_train, y_train)
y_pred = optimized_GBM.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
cross_val_score(optimized_GBM, X_train, y_train) 


from sklearn.metrics import accuracy_score
# RandomizedSearchCV to find the optimal hyperparameters. 
from xgboost.sklearn import XGBRegressor  
import scipy.stats as st
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)
params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}
xgbreg = XGBRegressor()  
from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)  
gs.fit(X_train, y_train)  
print gs.best_params_
print gs.best_score_

######
from xgboost.sklearn import XGBClassifier  
from xgboost.sklearn import XGBRegressor

xclas = XGBClassifier()  # and for classifier  
xclas.fit(X_train, y_train)  
y_pred = xclas.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
cross_val_score(xclas, X_train, y_train) 

#################################################
#http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/ 
'''
The EnsembleVoteClassifier implements "hard" and "soft" voting. 
In hard voting, we predict the final class label as the class label that has been predicted most frequently by the classification models. 
In soft voting, we predict the class labels by averaging the class-probabilities (only recommended if the classifiers are well-calibrated).

'''
#################################################       Stacking        #################################################
#https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np

# I Sample Stacking  
clf1 = ExtraTreesClassifier()
clf2 = RandomForestClassifier()
clf3 = AdaBoostClassifier()
gb = GradientBoostingClassifier()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=gb)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['ExtraTree', 
                       'Random Forest', 
                       'AdaBoost',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_train, y_train, 
                                              cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

# II Demo prob as meta-feautres
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=gb)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_train, y_train, 
                                              cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

# III Stacked Classification and GridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier

# Initializing models

clf1 = ExtraTreesClassifier()
clf2 = RandomForestClassifier()
clf3 = AdaBoostClassifier()
gb = GradientBoostingClassifier()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=gb)

params = {'meta-gradientboostingclassifier__n_estimators': [3, 5, 8,10, 16, 32, 50, 100, 150, 200, 250, 300],
          'meta-gradientboostingclassifier__learning_rate': [0.4,0.5,0.6,0.8,0.9,1.0],
          'adaboostclassifier__n_estimators':[3, 5, 8,10, 16, 32, 50, 100, 150, 200],
          'adaboostclassifier__learning_rate':[0.6,0.8,0.9,1.0],
          'randomforestclassifier__max_depth':[3, None],
          'randomforestclassifier__max_features':[1, 3, 10],
          'randomforestclassifier__min_samples_split':[2, 3, 10],
          'randomforestclassifier__min_samples_leaf':[1, 3, 10],
          'randomforestclassifier__bootstrap':[True, False],
          'randomforestclassifier__bootstrap':["gini", "entropy"],
          'extratreesclassifier__max_depth':[3, None],
          'extratreesclassifier__max_features':[1, 3, 10],
          'extratreesclassifier__min_samples_split':[2, 3, 10],
          'extratreesclassifier__min_samples_leaf':[1, 3, 10],
          'extratreesclassifier__bootstrap':[True, False],
          'extratreesclassifier__bootstrap':["gini", "entropy"]
}
          
'''          
GradientBoostingClassifier()
param_grid = {'n_estimators': [3, 5, 8,10, 16, 32, 50, 100, 150, 200, 250, 300],
              'learning_rate': [0.4,0.5,0.6,0.8,0.9,1.0]}
AdaBoostClassifier()
param_grid = {'n_estimators': [3, 5, 8,10, 16, 32, 50, 100, 150, 200],
              'learning_rate': [0.6,0.8,0.9,1.0]}
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
'''

grid = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=5,
                    refit=True)
grid.fit(X_train, y_train)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)






############################# Mix - http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/
import pandas as pd
from sklearn.grid_search import GridSearchCV

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
    
    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': mean(scores),
                 'std_score': std(scores),
            }
            return pd.Series(dict(params.items() + d.items()))
                      
        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort([sort_by], ascending=False)
        
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]


from sklearn import datasets



from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC

models1 = { 
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params1 = { 
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ]
}

helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_train, y_train, scoring='f1', n_jobs=-1)
helper1.score_summary(sort_by='min_score')
