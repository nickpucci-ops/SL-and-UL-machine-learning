# training script for supervised learning with polynomial ridge regression (not submitted)
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

# training data
x = pd.read_csv('data/x.csv', header=None).values.flatten()  # .values.flatten => shapes as 50 rows
y = pd.read_csv('data/y.csv', header=None).values.flatten()
z = pd.read_csv('data/z.csv', header=None).values.flatten()

X = np.column_stack((x, y))  # combine x and y into the feature matrix 'X' (50, 2)

# training splits: 80% training, 20% test
# 5 random restarts with different train-test splits
n_restarts = 5
alpha_constant = 100.0  # Keep your alpha value

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

    trained_data_ridge = {
        'model': model_ridge,
        'scaler_X': scaler_X,
        'scaler_z': scaler_z,
        'poly': poly,
        'X_test': X_test,  # test data saved for visualization
        'z_test': z_test
    }
    with open(f'trained_models/trained_model_ridge_restart_{restart}.pkl', 'wb') as f:
        pickle.dump(trained_data_ridge, f)
