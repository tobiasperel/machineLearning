import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'length': [1.5, 0.5, 1.2, 0.9, 3],
    'width': [0.7, 0.2, 0.15, 0.2, 1.1]
    })
hist = df.hist(bins=3) # genereas el grafico 
plt.show()
print(df[0:3])
print(df.shape) # row y columnas



