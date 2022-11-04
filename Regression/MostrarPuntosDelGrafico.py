# https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/coursera/ML0101EN/ML0101EN-Reg-NoneLinearRegression-py-v1.ipynb?lti=true
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("china_gdp.csv")
df.head(10)

plt.figure(figsize=(8, 5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
# a partir de esto uno hace la funcion que vea mas prudente en base a los puntos que tiene