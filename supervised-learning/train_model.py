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

#Implement polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)  # (1, x, y, x^2, xy, y^2)
X_poly = poly.fit_transform(X_scaled)  #(50, 6)

alpha_constant = 10.0
#traing model with ridge regression, saving trained model, scalars, and polynomial transformer
model = Ridge(alpha=alpha_constant)#alpha is a constant that multiplies the L2 term and controls the regularization strength, alpha = 0 is linear regression
model.fit(X_poly, z_scaled)

trained_data = {
    'model': model,
    'scaler_X': scaler_X,
    'scaler_z': scaler_z,
    'poly': poly
}
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(trained_data, f)
