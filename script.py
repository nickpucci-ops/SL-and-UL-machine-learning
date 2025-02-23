import pandas as pd
import matplotlib.pyplot as plt
import csv

x = pd.read_csv('x.csv', header=None)
y = pd.read_csv('y.csv')
z = pd.read_csv('z.csv', header=None)

#2D scatter plots
plt.scatter(x, z, label='x vs z', alpha = 0.5)
plt.show()
