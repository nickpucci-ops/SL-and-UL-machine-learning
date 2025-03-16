# visualize trained model results with predicted z values (not submitted)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from train_model import alpha_constant

# loading ridge model (best model)
with open('trained_models/trained_model_best.pkl', 'rb') as f:
    trained_data = pickle.load(f)
model = trained_data['model']
scaler_X = trained_data['scaler_X']
scaler_z = trained_data['scaler_z']
poly = trained_data['poly']
X_test = trained_data['X_test']
z_test = trained_data['z_test']
#X_train = trained_data['X_train']
#z_train = trained_data['z_train']

#read data
x = np.loadtxt('data/x.csv', delimiter=',')
y = np.loadtxt('data/y.csv', delimiter=',')
z_actual = np.loadtxt('data/z.csv', delimiter=',')
X = np.column_stack((x, y))


# Scale features and transform to polynomial features (full dataset)
X_scaled = scaler_X.transform(X)
X_poly = poly.transform(X_scaled)

# Predict on full dataset
z_pred_scaled = model.predict(X_poly)
z_predicted = scaler_z.inverse_transform(z_pred_scaled.reshape(-1, 1)).ravel()

# Predict on test set for MSE
X_test_scaled = scaler_X.transform(X_test)
X_test_poly = poly.transform(X_test_scaled)
z_pred_test_scaled = model.predict(X_test_poly)
z_pred_test = scaler_z.inverse_transform(z_pred_test_scaled.reshape(-1, 1)).ravel()
#mse_test = mean_squared_error(z_test, z_pred_test)
#print(f"MSE on test set for best model: {mse_test}")


# Predict on full dataset for MSE
#mse_full = mean_squared_error(z_actual, z_predicted_ridge)
#print(f"MSE on full dataset for best model: {mse_full}")

#scaling and applying model to generate predictions
#X_scaled = scaler_X.transform(X)
#X_poly = poly.transform(X_scaled)

#z_pred_scaled = model.predict(X_poly)
#z_predicted = scaler_z.inverse_transform(z_pred_scaled.reshape(-1, 1)).ravel()

#configure subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Predicted z vs x and Actual z vs x
ax1.scatter(x, z_actual, color='blue', alpha=0.3, label='Actual z vs x (full dataset)')
ax1.scatter(X_test[:, 0], z_test, color='cyan', alpha=0.7, marker='^', label='Actual z vs x (test)')
ax1.scatter(x, z_predicted, color='red', alpha=0.3, marker='o', label=f'Pred z (alpha={alpha_constant}) vs x')
ax1.set_xlabel('x')
ax1.set_ylabel('z')
ax1.set_title('z vs x (Actual vs Predicted)')
ax1.legend()

# Plot 2: Predicted z vs y and Actual z vs y
ax2.scatter(y, z_actual, color='green', alpha=0.3, label='Actual z vs y (full dataset)')
ax2.scatter(X_test[:, 1], z_test, color='lime', alpha=0.7, marker='^', label='Actual z vs y (test)')
ax2.scatter(y, z_predicted, color='red', alpha=0.3, marker='o', label=f'Pred z (alpha={alpha_constant}) vs y')
ax2.set_xlabel('y')
ax2.set_ylabel('z')
ax2.set_title('z vs y (Actual vs Predicted)')
ax2.legend()

# Plot 3: Actual vs Predicted z (test set)
ax3.scatter(z_pred_test, z_test, color='red', alpha=0.7, marker='o', label=f'alpha={alpha_constant}')
ax3.plot([z_test.min(), z_test.max()], [z_test.min(), z_test.max()], 'k--', label='Ideal Fit')
ax3.set_xlabel('Predicted z')
ax3.set_ylabel('Actual z')
ax3.set_title('Actual vs Predicted z (PolyRidge Regression on Test Set)')
ax3.legend()

plt.tight_layout()
plt.show()
