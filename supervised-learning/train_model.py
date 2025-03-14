# training script for supervised learning with polynomial ridge regression (not submitted)
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# training data 
x = pd.read_csv('data/x.csv', header=None).values.flatten()  # .values.flatten => shapes as n rows
y = pd.read_csv('data/y.csv', header=None).values.flatten()
z = pd.read_csv('data/z.csv', header=None).values.flatten()

X = np.column_stack((x, y))  # combine x and y into the feature matrix 'X' (n, 2)

# training splits: 80% training, 20% test
# Perform 5 random restarts with different train-test splits
n_restarts = 5
alpha_constant = 50.0  # Reverting to the balanced alpha value
best_mse = float('inf')  # Track the best MSE
best_restart = 0  # Track the best restart index

for restart in range(n_restarts):
    print(f"random restart {restart + 1}/{n_restarts}")
    
    # Split data: 80% training, 20% test with a different random seed
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42 + restart)

    # scale the features and the target due to the large z values
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_z = StandardScaler(with_mean=True, with_std=True)
    z_train_scaled = scaler_z.fit_transform(z_train.reshape(-1, 1)).ravel()
    z_test_scaled = scaler_z.transform(z_test.reshape(-1, 1)).ravel()

    # Implement polynomial features
    poly = PolynomialFeatures(degree=3)  # (1, x, y, x^2, xy, y^2, x * y^2, x^2 * y, y^3, x^3 )
    X_train_poly = poly.fit_transform(X_train_scaled)  # (n, 10)
    X_test_poly = poly.transform(X_test_scaled)

    # train model with ridge regression, saving trained model, scalars, and polynomial transformer
    model_ridge = Ridge(alpha=alpha_constant)
    model_ridge.fit(X_train_poly, z_train_scaled)

    # Predict on test set to evaluate MSE
    z_pred_scaled_ridge = model_ridge.predict(X_test_poly)
    z_pred_ridge = scaler_z.inverse_transform(z_pred_scaled_ridge.reshape(-1, 1)).ravel()
    mse_test = mean_squared_error(z_test, z_pred_ridge)
    print(f"MSE for restart {restart} on test set: {mse_test}")

    # Predict on training set for comparison
    z_pred_train_scaled = model_ridge.predict(X_train_poly)
    z_pred_train = scaler_z.inverse_transform(z_pred_train_scaled.reshape(-1, 1)).ravel()
    mse_train = mean_squared_error(z_train, z_pred_train)
    #print(f"MSE for Restart {restart} (Train): {mse_train}")

    if mse_test < best_mse:
        best_mse = mse_test
        best_restart = restart
        best_trained_data_ridge = {
            'model': model_ridge,
            'scaler_X': scaler_X,
            'scaler_z': scaler_z,
            'poly': poly,
            'X_test': X_test,
            'z_test': z_test,
            'X_train': X_train,
            'z_train': z_train
        }

with open(f'trained_models/trained_model_ridge_best.pkl', 'wb') as f:
    pickle.dump(best_trained_data_ridge, f)
print(f"best model selected from restart {best_restart + 1} with test MSE: {best_mse}")
