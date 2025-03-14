# training script for supervised learning with polynomial ridge regression (not submitted)
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# training data
x = pd.read_csv('data/x_sample.csv', header=None).values.flatten()  # .values.flatten => shapes as 50 rows
y = pd.read_csv('data/y_sample.csv', header=None).values.flatten()
z = pd.read_csv('data/z_sample.csv', header=None).values.flatten()

X = np.column_stack((x, y))  # combine x and y into the feature matrix 'X' (50, 2)

# training splits: 80% training, 20% test
# Perform 5 random restarts with different train-test splits
n_restarts = 5
alpha_constant = 100.0 
best_mse = float('inf')  # Track the best MSE
best_restart = 0  # Track the best restart index

for restart in range(n_restarts):
    print(f"Random Restart {restart + 1}/{n_restarts}")
    
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
    X_train_poly = poly.fit_transform(X_train_scaled)  # (50, 10)
    X_test_poly = poly.transform(X_test_scaled)

    # traing model with ridge regression, saving trained model, scalars, and polynomial transformer
    model_ridge = Ridge(alpha=alpha_constant)  # alpha is a constant that multiplies the L2 term (the penalty) and controls the regularization strength
    model_ridge.fit(X_train_poly, z_train_scaled)

    # Predict on test set to evaluate MSE
    z_pred_scaled_ridge = model_ridge.predict(X_test_poly)
    z_pred_ridge = scaler_z.inverse_transform(z_pred_scaled_ridge.reshape(-1, 1)).ravel()
    mse = mean_squared_error(z_test, z_pred_ridge)
    print(f"MSE for Restart {restart}: {mse}")

    # Track the best model based on MSE
    if mse < best_mse:
        best_mse = mse
        best_restart = restart
        best_trained_data_ridge = {
            'model': model_ridge,
            'scaler_X': scaler_X,
            'scaler_z': scaler_z,
            'poly': poly,
            'X_test': X_test,  # test data saved for visualization
            'z_test': z_test
        }

# Save only the best model
with open(f'trained_models/trained_model_ridge_best.pkl', 'wb') as f:
    pickle.dump(best_trained_data_ridge, f)
print(f"Best model: {best_restart + 1} with MSE: {best_mse}")
