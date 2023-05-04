#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Maged Eid - SEP 787 - Course Final Project
# Load libraries (Classification)
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


# In[2]:


data = pd.read_csv('C:\ALL\McMaster M.Eng. Study\SEP 787 - Machine Learning (Classification Models)\BankNote_Authentication.csv')


# In[3]:


print(data.head())


# In[4]:


# Data Exploration
print('Number of Rows: ', data.shape[0])
print('Number of Columns: ', data.shape[1], '\n')
print('SubSet of Data:\n ', data.head(), '\n')

#labeling
labels = ['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Target']
data.columns = labels
print('Columns Names:', data.columns, '\n')
print('Data Describe:\n ', data.describe(), '\n')
print('Data Information:'); print(data.info())


# In[5]:


# Data Split (Training 75% of the data & Test 25% of the data)
x = data.drop('Target', axis=1).values
y = data['Target'].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print(X_train.shape)
print(y_train.shape)

# Data histograms
data.hist()
plt.show()


# In[6]:


# KNN Algorithm
knn = KNeighborsClassifier(n_neighbors=4)

# cross-validation
cv_results = cross_validate(knn, X_train, y_train, cv=5)

# fitting training data
start_time = time.time()
knn.fit(X_train, y_train)
training_time = time.time() - start_time

# predicted data
start_time = time.time()
y_predicted = knn.predict(X_test)
testing_time = time.time() - start_time

# KNN Results

# Classification Report
print("Classification Report")
print(metrics.classification_report(y_test, y_predicted))

# Accuracy Score matrix
print('Accuracy of KNN Algorithm: '
      , metrics.accuracy_score(y_test, y_predicted)*100)

# F1 Score
f1_score_knn = metrics.f1_score(y_test, y_predicted, average='micro')

# Confusion matrix
cm_knn = metrics.confusion_matrix(y_test, y_predicted)

# recall
recall_knn = metrics.recall_score(y_test, y_predicted)

# Print computational times
print(f"Training time: {training_time:.4f} seconds")
print(f"Testing time: {testing_time:.4f} seconds")

# Print cross-validation results
print("Cross Validation Results:")
print(f"Test Scores: {cv_results['test_score']}")
print(f"Mean Test Score: {np.mean(cv_results['test_score'])}")

# Heatmap confusion matrix 
sns.heatmap(cm_knn, annot=True, fmt=".0f", linewidths=3, square=True, cmap='Blues', color="#cd1076")
plt.ylabel('actual label')
plt.xlabel('predicted label')

# show F1 Score and Recall 
plt.title(f'F1 Score [KNN Algorithm]: {f1_score_knn:.2f}\n'
          f'Recall [KNN Algorithm]: {recall_knn:.2f}', size=14, color='black')
plt.show()

# predict probabilities for test data
y_prob = knn.predict_proba(X_test)

# calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[7]:


# Naive Bayes Algorithm
gnb = GaussianNB()

# cross-validation
cv_results = cross_validate(gnb, X_train, y_train, cv=5)

# fitting training data
start_time = time.time()
gnb.fit(X_train, y_train)
training_time = time.time() - start_time

# predicted data
start_time = time.time()
y_predicted = gnb.predict(X_test)
testing_time = time.time() - start_time

# Classification Report
print("Classification Report")
print(metrics.classification_report(y_test, y_predicted))

# Accuracy Score matrix
print('Accuracy of Naive Bayes Algorithm: '
      , metrics.accuracy_score(y_test, y_predicted)*100)

# F1 Score
f1_score_NB = metrics.f1_score(y_test, y_predicted, average='micro')

# Confusion matrix
cm_mnb = metrics.confusion_matrix(y_test, y_predicted)

# recall
recall_NB = metrics.recall_score(y_test, y_predicted)

# Print computational times
print(f"Training time: {training_time:.4f} seconds")
print(f"Testing time: {testing_time:.4f} seconds")

# Print cross-validation results
print("Cross Validation Results:")
print(f"Test Scores: {cv_results['test_score']}")
print(f"Mean Test Score: {np.mean(cv_results['test_score'])}")

# Heatmap confusion matrix 
sns.heatmap(cm_mnb, annot=True, fmt=".0f", linewidths=3, square=True, cmap='Blues', color="#cd1076")
plt.ylabel('actual label')
plt.xlabel('predicted label')

# show F1 Score and Recall 
plt.title(f'F1 Score [Naive Bayes Algorithm]: {f1_score_NB:.2f}\n'
          f'Recall [NB Algorithm]: {recall_NB:.2f}', size=14, color='black')
plt.show()

# predict probabilities for test data
y_prob = gnb.predict_proba(X_test)

# calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[8]:


# Decision Tree Algorithm
dtc = DecisionTreeClassifier()

# cross-validation
cv_results = cross_validate(dtc, X_train, y_train, cv=5)

# fitting training data
start_time = time.time()
model = dtc.fit(X_train, y_train)
training_time = time.time() - start_time

# predicted data
start_time = time.time()
y_predicted = dtc.predict(X_test)
testing_time = time.time() - start_time

# Decision Tree Results
print('Classification Report:')
print(metrics.classification_report(y_test, y_predicted))

# Accuracy Score matrix
print('Accuracy of Decision Tree Algorithm: '
      , metrics.accuracy_score(y_test, y_predicted)*100)

# F1 Score
f1_score_DTC = metrics.f1_score(y_test, y_predicted, average='micro')

# Confusion matrix
cm_dtc = metrics.confusion_matrix(y_test, y_predicted)

# recall
recall_DTC = metrics.recall_score(y_test, y_predicted)

# Heatmap confusion matrix 
sns.heatmap(cm_dtc, annot=True, fmt=".0f", linewidths=3, square=True, cmap='Blues', color="#cd1076")
plt.ylabel('actual label')
plt.xlabel('predicted label')

# show F1 Score and Recall 
plt.title(f'F1 Score [Decision Tree Algorithm]: {f1_score_DTC:.2f}\n'
          f'Recall [DTC Algorithm]: {recall_DTC:.2f}', size=14, color='black')
plt.show()

# print training and testing times
print('Training Time: ', training_time)
print('Testing Time: ', testing_time)

# Print cross-validation results
print("Cross Validation Results:")
print(f"Test Scores: {cv_results['test_score']}")
print(f"Mean Test Score: {np.mean(cv_results['test_score'])}")

# predicted probabilities
y_prob = dtc.predict_proba(X_test)[:,1]

# calculate false positive rate, true positive rate, and threshold
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# calculate area under curve
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Decision Tree Algorithm')
plt.legend(loc="lower right")
plt.show()


# In[9]:


# AdaBoost Algorithm
adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)

# cross-validation
cv_results = cross_validate(adaboost, X_train, y_train, cv=5)

# fitting training data
start_time = time.time()
model = adaboost.fit(X_train, y_train)
training_time = time.time() - start_time

# predicted data
start_time = time.time()
y_predicted = adaboost.predict(X_test)
testing_time = time.time() - start_time

# AdaBoost Results
print('Classification Report:')
print(metrics.classification_report(y_test, y_predicted))

# Accuracy Score matrix
print('Accuracy of AdaBoost Algorithm: '
      , metrics.accuracy_score(y_test, y_predicted)*100)

# F1 Score
f1_score_AB = metrics.f1_score(y_test, y_predicted, average='micro')

# Confusion matrix
cm_ab = metrics.confusion_matrix(y_test, y_predicted)

# recall
recall_AB = metrics.recall_score(y_test, y_predicted)

# Heatmap confusion matrix 
sns.heatmap(cm_ab, annot=True, fmt=".0f", linewidths=3, square=True, cmap='Blues', color="#cd1076")
plt.ylabel('actual label')
plt.xlabel('predicted label')

# show F1 Score and Recall 
plt.title(f'F1 Score [AdaBoost Algorithm]: {f1_score_AB:.2f}\n'
          f'Recall [AdaBoost Algorithm]: {recall_AB:.2f}', size=14, color='black')
plt.show()

# print training and testing times
print('Training Time: ', training_time)
print('Testing Time: ', testing_time)

# Print cross-validation results
print("Cross Validation Results:")
print(f"Test Scores: {cv_results['test_score']}")
print(f"Mean Test Score: {np.mean(cv_results['test_score'])}")

# predicted probabilities
y_prob = adaboost.predict_proba(X_test)[:,1]

# calculate false positive rate, true positive rate, and threshold
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)

# calculate area under curve
roc_auc = metrics.auc(fpr, tpr)

# plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - AdaBoost Algorithm')
plt.legend(loc="lower right")
plt.show()


# In[10]:


plt.figure(figsize=(12, 6))
model_f1_score = [f1_score_knn, f1_score_NB, f1_score_DTC, f1_score_AB]
recalls = [recall_knn, recall_NB, recall_DTC, recall_AB]
model_name = ['KNN', 'Naive Bayes', 'Decision Tree', 'AdaBoost']
recall_name = ['KNN', 'Naive Bayes', 'Decision Tree', 'AdaBoost']

# Barplot f1 score
sns.barplot(x=model_f1_score, y=model_name, palette='magma')
plt.title('Algorithms F1 Score')
plt.show()

# barplot recall 
sns.barplot(x=recalls, y=recall_name, palette='magma')
plt.title('Algorithms Recall')
plt.show()


# In[11]:


#Kind Regards, Maged Eid, SEP 787. Thank You.

