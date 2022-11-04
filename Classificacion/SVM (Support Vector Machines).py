# https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/coursera/ML0101EN/ML0101EN-Clas-SVM-cancer-py-v1.ipynb?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY3Rpb24iOiJsdGkvcmVzdWx0cyIsInVzZXJuYW1lIjoicGVyZWx0b2JpYXMiLCJvcHRpb25zIjp7Imxpc19vdXRjb21lX3NlcnZpY2VfdXJsIjoiaHR0cHM6Ly9hcGkuY291cnNlcmEub3JnL2FwaS9vbkRlbWFuZEx0aU91dGNvbWVzLnYxIiwibGlzX3Jlc3VsdF9zb3VyY2VkaWQiOiJvbmRlbWFuZH44NzliNWJlZjU5NjIwOTNhMTU5YmQyNDhmMjI0MjNiZSF-OFVqZU1rLW1FZWl0NGc0R3N4RTRkZyF-dWVWSWohflNpYXBWQlVGRWV5MmpCTEU2NWdnWHciLCJvYXV0aF9jb25zdW1lcl9rZXkiOiJtYWNoaW5lX2xlYXJuaW5nX3dpdGhfcHl0aG9uIiwic2NvcmUiOiIxLjAifSwiaWF0IjoxNjMxNTg3MzYxLCJleHAiOjE2MzE2MDg5NjF9.IljVkfMVM6Ul9bYpUACCA3ZUOCDXzPrTm-lTvz4xV-g&d=grading.labs.cognitiveclass.ai&lti=true
# wget -O cell_samples.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head(9))
print(cell_df.shape)

# Como las posibles soluciones son 2 o 4 se ponen esas dos opciones
# la sintacios es y = .....  kind='scatter' , primer categoria en este caso x='Clump' , segunda categoria y=UnifShape
ax = cell_df[cell_df['Class'] == 4][0:698].plot(
    kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
cell_df[cell_df['Class'] == 2][0:698].plot(
    kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)
plt.show()

# Data pre-processing and selection: pasas todos los valores a int

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
print(X[0:5])
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
print(y[0:7])

#Train and Test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

# Modeling (SVM with Scikit-learn)

# 1.Linear 2.Polynomia 3.Radial basis function (RBF) 4.Sigmoid (DIFERENTS TYPES OF FUCTIONS)
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print(yhat[0:5])

# Evaluation


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


print(f1_score(y_test, yhat, average='weighted'))
print(jaccard_score(y_test, yhat,pos_label=2))

clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train) 
yhat2 = clf2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))