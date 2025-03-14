#training script for supervised learning with polynomial ridge regression (not submitted)
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

#training data
x = pd.read_csv('data/x.csv', header=None).values.flatten()#.values.flatten => shapes as 50 rows
y = pd.read_csv('data/y.csv', header=None).values.flatten()
z = pd.read_csv('data/z.csv', header=None).values.flatten()

X = np.column_stack((x,y)) #combine x and y into the feature matrix 'X' (50, 2)

#scale the features and the target due to the large z values
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)  # Scale x, y
scaler_z = StandardScaler(with_mean=True, with_std=True)
z_scaled = scaler_z.fit_transform(z.reshape(-1, 1)).ravel()

#Implement polynomial features
poly = PolynomialFeatures(degree=3)  # (1, x, y, x^2, xy, y^2, x * y^2, x^2 * y, y^3, x^3 )
X_poly = poly.fit_transform(X_scaled)  #(50, 10)

alpha_constant = 100.0
#traing model with ridge regression, saving trained model, scalars, and polynomial transformer
model_ridge = Ridge(alpha=alpha_constant)#alpha is a constant that multiplies the L2 term (the penalty) and controls the regularization strength
model_ridge.fit(X_poly, z_scaled)

model_no_ridge = LinearRegression()#alpha=0 is equivalent to linear regression
model_no_ridge.fit(X_poly, z_scaled)

trained_data_ridge = {
    'model': model_ridge,
    'scaler_X': scaler_X,
    'scaler_z': scaler_z,
    'poly': poly
}
with open('trained_models/trained_model_ridge.pkl', 'wb') as f:
    pickle.dump(trained_data_ridge, f)


trained_data_no_ridge = {
    'model': model_no_ridge,
    'scaler_X': scaler_X,
    'scaler_z': scaler_z,
    'poly': poly
}
with open('trained_models/trained_model_no_ridge.pkl', 'wb') as f:
    pickle.dump(trained_data_no_ridge, f)
