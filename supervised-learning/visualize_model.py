# visualize_model.py (not submitted)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from train_model import alpha_constant

x = np.loadtxt('data/x_sample.csv', delimiter=',')
y = np.loadtxt('data/y_sample.csv', delimiter=',')
z_actual = np.loadtxt('data/z_sample.csv', delimiter=',')

#loading ridge model
with open('trained_models/trained_model_ridge.pkl', 'rb') as f:
    trained_data_ridge = pickle.load(f)
model_ridge = trained_data_ridge['model']
scaler_X_ridge = trained_data_ridge['scaler_X']
scaler_z_ridge = trained_data_ridge['scaler_z']
poly_ridge = trained_data_ridge['poly']
#loading no ridge model
with open('trained_models/trained_model_no_ridge.pkl', 'rb') as f:
    trained_data_no_ridge = pickle.load(f)
model_no_ridge = trained_data_no_ridge['model']
scaler_X_no_ridge = trained_data_no_ridge['scaler_X']
scaler_z_no_ridge = trained_data_no_ridge['scaler_z']
poly_no_ridge = trained_data_no_ridge['poly']

X = np.column_stack((x, y))

# Scale features and transform to polynomial features
X_scaled_ridge = scaler_X_ridge.transform(X)
X_poly_ridge = poly_ridge.transform(X_scaled_ridge)

X_scaled_no_ridge = scaler_X_no_ridge.transform(X)
X_poly_no_ridge = poly_no_ridge.transform(X_scaled_no_ridge)

# Predict
z_pred_scaled_ridge = model_ridge.predict(X_poly_ridge)
z_predicted_ridge = scaler_z_ridge.inverse_transform(z_pred_scaled_ridge.reshape(-1, 1)).ravel()

z_pred_scaled_no_ridge = model_no_ridge.predict(X_poly_no_ridge)
z_predicted_no_ridge = scaler_z_no_ridge.inverse_transform(z_pred_scaled_no_ridge.reshape(-1, 1)).ravel()

#old visualization, delete later once done testing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Predicted z vs x and y (Ridge model)
ax1.scatter(x, z_predicted_ridge, color='blue', alpha=0.5, label='vs x')
ax1.scatter(y, z_predicted_ridge, color='green', alpha=0.5, label='vs y')
ax1.set_xlabel('x or y')
ax1.set_ylabel('Predicted z')
ax1.set_title(f'Predicted z vs x and y (Degree 3, alpha={alpha_constant})')
ax1.legend()

# Plot 2: Actual vs Predicted z (No Ridge vs Ridge)
if z_actual is not None:
    ax2.scatter(z_predicted_no_ridge, z_actual, color='purple', alpha=0.5, label='No Ridge')
    ax2.scatter(z_predicted_ridge, z_actual, color='red', alpha=0.5, label=f'Ridge (alpha={alpha_constant})')
    ax2.plot([z_actual.min(), z_actual.max()], [z_actual.min(), z_actual.max()], 'k--', label='Ideal Fit')
    ax2.set_xlabel('Predicted z')
    ax2.set_ylabel('Actual z')
    ax2.set_title('Actual vs Predicted z (No Ridge vs Ridge)')
    ax2.legend()
else:
    ax2.text(0.5, 0.5, 'No z.csv available\nCannot plot actual vs predicted',
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Actual vs Predicted z (Unavailable)')

plt.tight_layout()
plt.show()

#new visualizations
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Predicted z vs x and Actual z vs x (Ridge and No Ridge)
ax1.scatter(x, z_actual, color='blue', alpha=0.5, label='Actual z vs x')
ax1.scatter(x, z_predicted_ridge, color='red', alpha=0.5, label=f'Pred z (Ridge, alpha={alpha_constant}) vs x')
ax1.set_xlabel('x')
ax1.set_ylabel('z')
ax1.set_title('z vs x (Actual vs Predicted)')
ax1.legend()

# Plot 2: Predicted z vs y and Actual z vs y (Ridge and No Ridge)
ax2.scatter(y, z_actual, color='green', alpha=0.5, label='Actual z vs y')
ax2.scatter(y, z_predicted_ridge, color='red', alpha=0.5, label=f'Pred z (Ridge, alpha={alpha_constant}) vs y')
ax2.set_xlabel('y')
ax2.set_ylabel('z')
ax2.set_title('z vs y (Actual vs Predicted)')
ax2.legend()

# Plot 3: Actual vs Predicted z (No Ridge vs Ridge)
ax3.scatter(z_predicted_no_ridge, z_actual, color='purple', alpha=0.5, label='No Ridge')
ax3.scatter(z_predicted_ridge, z_actual, color='red', alpha=0.5, label=f'Ridge (alpha={alpha_constant})')
ax3.plot([z_actual.min(), z_actual.max()], [z_actual.min(), z_actual.max()], 'k--', label='Ideal Fit')
ax3.set_xlabel('Predicted z')
ax3.set_ylabel('Actual z')
ax3.set_title('Actual vs Predicted z (No Ridge vs Ridge)')
ax3.legend()

plt.tight_layout()
plt.show()
