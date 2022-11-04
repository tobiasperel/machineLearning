#https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/coursera/ML0101EN/ML0101EN-Reg-Mulitple-Linear-Regression-Co2-py-v1.ipynb?lti=true
#wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()

# summarize the data
df.describe()


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] 


msk = np.random.rand(len(df)) < 0.8  # aca lo que se hace es darle el 80% de la base de datos para entrenar
# y el 20% para probar, 80% testing, 20% trading y con la funcion np.random.rand(len(df)) le damos datos aleatorios
train = cdf[msk] # aca le decimos que haga train con el msk 
test = cdf[~msk] # aca que testie con lo restante, por eso tiene "~"


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])  #aca sacas la funcion
regr.fit (x, y)   
# The coefficients
print ('Coefficients: ', regr.coef_) # m <-- m.x+b 

y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

#Practice # 
print("----------------------------Practice---------------------------------------------")
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)
y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))