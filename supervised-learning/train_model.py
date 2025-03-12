import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

#training data
x = pd.read_csv('x.csv', header=None).values.flatten()#.values.flatten => shapes as 50 rows
y = pd.read_csv('y.csv', header=None).values.flatten()
z = pd.read_csv('z.csv', header=None).values.flatten()


X = np.column_stack((x,y)) #combine x and y into the feature matrix 'X' (50, 2)

#scale the features and the target due to the large z values
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)  # Scale x, y
scaler_z = StandardScaler(with_mean=True, with_std=True)
z_scaled = scaler_z.fit_transform(z.reshape(-1, 1)).ravel()
