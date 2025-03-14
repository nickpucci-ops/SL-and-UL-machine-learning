# visualize trained model results with predicted z values (not submitted)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from train_model import alpha_constant
from sklearn.metrics import mean_squared_error

x = np.loadtxt('data/x_sample.csv', delimiter=',')
y = np.loadtxt('data/y_sample.csv', delimiter=',')
z_actual = np.loadtxt('data/z_sample.csv', delimiter=',')

# loading ridge model (best model)
with open('trained_models/trained_model_ridge_best.pkl', 'rb') as f:
    trained_data_ridge = pickle.load(f)
model_ridge = trained_data_ridge['model']
scaler_X_ridge = trained_data_ridge['scaler_X']
scaler_z_ridge = trained_data_ridge['scaler_z']
poly_ridge = trained_data_ridge['poly']
X_test = trained_data_ridge['X_test']
z_test = trained_data_ridge['z_test']

X = np.column_stack((x, y))

# Scale features and transform to polynomial features (full dataset)
X_scaled_ridge = scaler_X_ridge.transform(X)
X_poly_ridge = poly_ridge.transform(X_scaled_ridge)

# Predict on full dataset
z_pred_scaled_ridge = model_ridge.predict(X_poly_ridge)
z_predicted_ridge = scaler_z_ridge.inverse_transform(z_pred_scaled_ridge.reshape(-1, 1)).ravel()

# Predict on test set for MSE
X_test_scaled = scaler_X_ridge.transform(X_test)
X_test_poly = poly_ridge.transform(X_test_scaled)
z_pred_test_scaled = model_ridge.predict(X_test_poly)
z_pred_test = scaler_z_ridge.inverse_transform(z_pred_test_scaled.reshape(-1, 1)).ravel()
mse_test = mean_squared_error(z_test, z_pred_test)
print(f"best model based on mse test set: {mse_test}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Predicted z vs x and Actual z vs x (Ridge)
ax1.scatter(x, z_actual, color='blue', alpha=0.5, label='Actual z vs x (full dataset)')
ax1.scatter(X_test[:, 0], z_test, color='blue', alpha=0.5, label='Actual z vs x (test)')
ax1.scatter(X_test[:, 0], z_pred_test, color='red', alpha=0.5, label=f'Pred z (Ridge, alpha={alpha_constant}) vs x')
ax1.set_xlabel('x')
ax1.set_ylabel('z')
ax1.set_title('z vs x (Actual vs Predicted)')
ax1.legend()

# Plot 2: Predicted z vs y and Actual z vs y (Ridge)
ax2.scatter(y, z_actual, color='green', alpha=0.5, label='Actual z vs y (full dataset)')
ax2.scatter(X_test[:, 1], z_test, color='green', alpha=0.5, label='Actual z vs y (test)')
ax2.scatter(X_test[:, 1], z_pred_test, color='red', alpha=0.5, label=f'Pred z (Ridge, alpha={alpha_constant}) vs y')
ax2.set_xlabel('y')
ax2.set_ylabel('z')
ax2.set_title('z vs y (Actual vs Predicted)')
ax2.legend()

# Plot 3: Actual vs Predicted z (test set)
ax3.scatter(z_pred_test, z_test, color='red', alpha=0.5, label=f'Ridge, alpha={alpha_constant}')
ax3.plot([z_test.min(), z_test.max()], [z_test.min(), z_test.max()], 'k--', label='Ideal Fit')
ax3.set_xlabel('Predicted z')
ax3.set_ylabel('Actual z')
ax3.set_title('Actual vs Predicted z (PolyRidge Regression on Test Set)')
ax3.legend()

plt.tight_layout()
plt.show()
