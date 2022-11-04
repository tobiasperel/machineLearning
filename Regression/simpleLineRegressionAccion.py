import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline

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

'''
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue') #esto es por si queres mostrar los valores
plt.xlabel("Engine size") # le decis cual eje es cual
plt.ylabel("Emission")
plt.show()
'''

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])  #aca sacas la funcion
regr.fit (train_x, train_y)   
# The coefficients
print ('Coefficients: ', regr.coef_) # m <-- m.x+b 
print ('Intercept: ',regr.intercept_) # termino independiente <-- b

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r') # el -r significa ques es de color rojo 
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x) # aca se calcula el error de la funcion

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )