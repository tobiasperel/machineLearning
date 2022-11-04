#https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/coursera/ML0101EN/ML0101EN-Clas-Decision-Trees-drug-py-v1.ipynb?lti=true
#wget -O drug200.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv
#hay que descargar esto --> https://graphviz.org/download/
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree


my_data = pd.read_csv("drug200.csv", delimiter=",")
print(my_data[0:5])
print(my_data.shape)  # size of the data

#Pre-processing

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:3])
print(X[0:5][1])
print(X[0:5][1][3])

#Esto es muy interesante e importante, aca pasas todos los datos que no sean numericos a numero. Por ejemplo
#si sos mujer sos 1 y si sos hombre 0.

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print('-----------------------------------------')
print(X[0:5])

y = my_data["Drug"] # y pasa a ser el valor de la droga
print(y[0:5])

#Setting up the Decision Tree
# Aca se le pone que haga test con el 30% de la base de datos y lo de random 3 no tengo mucha idea 
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape)) 
print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))

#Modeling

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4) # aca yo lo empezas a divir creo, el max_depth de 4 para infinito no cambia
drugTree.fit(X_trainset,y_trainset)

#Prediction (si mal no entiendo esta es la parte donde predecis como tal)

predTree = drugTree.predict(X_testset) 
print (predTree [0:5])
print (y_testset [0:5])

#Evaluation 
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#Visualisation

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5] # como son 6 columnas de la 0-6
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
