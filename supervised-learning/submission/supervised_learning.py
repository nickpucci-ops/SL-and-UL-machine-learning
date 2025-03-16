# supervised_learning.py (for submission)
import numpy as np
import pandas as pd
import pickle

#loading model
with open('trained_model_best.pkl', 'rb') as f:
    trained_data = pickle.load(f)

model = trained_data['model']
scaler_X = trained_data['scaler_X']
scaler_z = trained_data['scaler_z']
poly = trained_data['poly']

#reading x and y
x = pd.read_csv('x.csv', header=None).values.flatten()
y = pd.read_csv('y.csv', header=None).values.flatten()
X = np.column_stack((x, y))

#scaling and applying model to generate predictions
X_scaled = scaler_X.transform(X) #apply scalar value provided by trained model to transform the data for fitting
X_poly = poly.transform(X_scaled) #apply the polynomial features provided by trained model to transform the SCALED data for fitting

z_pred_scaled = model.predict(X_poly) #make predictions for z based on poly scaled transformed data 
z_predicted = scaler_z.inverse_transform(z_pred_scaled.reshape(-1, 1)).ravel() # unscale the data using the provided scaler z from trained model to match actual z value scale

#saved z predictions
pd.DataFrame(z_predicted).to_csv('z-predicted.csv', header=False, index=False)
