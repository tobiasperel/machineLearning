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


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] # aca le pones los valores que queres tener en cuenta
#cdf.head(22)
print(cdf.head(22)) #cantidad de rows que queres ver

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']] 
viz.hist()  # esto es como para prepararlo pero no lo efectuas pero solo con show se muetra
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB") #que toma el eje x
plt.ylabel("Emission") #que toma el eje y
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
plt.show()
