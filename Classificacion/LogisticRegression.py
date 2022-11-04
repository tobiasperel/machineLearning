# https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/coursera/ML0101EN/ML0101EN-Clas-Logistic-Reg-churn-py-v1.ipynb?lti=true
#wget -O ChurnData.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import log_loss
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()

churn_df = churn_df[['tenure', 'age', 'address', 'income',
                    'ed', 'employ', 'equip',   'callcard', 'wireless', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
# print(churn_df.head())
print("The data length is: " + str(churn_df.shape))

X = np.asarray(churn_df[['tenure', 'age', 'address',
            'income', 'ed', 'employ', 'equip']])
# print(X[0:5])
y = np.asarray(churn_df['churn'])
#print(y [0:5])

# Normalize the data:

X = preprocessing.StandardScaler().fit(X).transform(X)
# print(X[0:5])

# Train / Test dataset


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

# Modeling (Logistic Regression with Scikit-learn)

# c = inverse of regularization strength, tiene que ser un float y cuanto mas chico mejor
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
print(LR)

yhat = LR.predict(X_test)  # aca predecimos los valores en valor binario 0/1
print(yhat)
yhat_prob = LR.predict_proba(X_test)
print("yhat_prob = " + str(yhat_prob))  # con probabilidad

# Evaluation

# se pone 0 salvo que todas las respuestas sean 1 o algo asi, siempre poner 0
print(jaccard_score(y_test, yhat, pos_label=0))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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


print(confusion_matrix(y_test, yhat, labels=[1, 0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))


#log loss

log_loss(y_test, yhat_prob)


LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))
