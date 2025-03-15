# supervised_learning.py (for submission)
import numpy as np
import pandas as pd
import pickle

with open('trained_model_ridge_best.pkl', 'rb') as f:
    trained_data_ridge = pickle.load(f)

model_ridge = trained_data_ridge['model']
scaler_X_ridge = trained_data_ridge['scaler_X']
scaler_z_ridge = trained_data_ridge['scaler_z']
poly_ridge = trained_data_ridge['poly']

x = pd.read_csv('x.csv', header=None).values.flatten()
y = pd.read_csv('y.csv', header=None).values.flatten()
X = np.column_stack((x, y))

X_scaled_ridge = scaler_X_ridge.transform(X)
X_poly_ridge = poly_ridge.transform(X_scaled_ridge)

z_pred_scaled_ridge = model_ridge.predict(X_poly_ridge)
z_predicted_ridge = scaler_z_ridge.inverse_transform(z_pred_scaled_ridge.reshape(-1, 1)).ravel()

pd.DataFrame(z_predicted_ridge).to_csv('z-predicted.csv', header=False, index=False)
