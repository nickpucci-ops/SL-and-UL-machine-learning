import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv('x.csv')
y = pd.read_csv('y.csv')
z = pd.read_csv('z.csv')

#2D scatter plots
plt.scatter(x, z, alpha = 0.5)
plt.show()
