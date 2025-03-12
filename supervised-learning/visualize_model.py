# visualize_model.py (not submitted)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from train_model import alpha_constant

x = np.loadtxt('x.csv', delimiter=',')
y = np.loadtxt('y.csv', delimiter=',')

with open('trained_model.pkl', 'rb') as f:
    trained_data = pickle.load(f)
model = trained_data['model']
scaler_X = trained_data['scaler_X']
scaler_z = trained_data['scaler_z']
poly = trained_data['poly']

X = np.column_stack((x, y))
X_scaled = scaler_X.transform(X)
X_poly = poly.transform(X_scaled)

z_pred_scaled = model.predict(X_poly)
z_predicted = scaler_z.inverse_transform(z_pred_scaled.reshape(-1, 1)).ravel()
z_actual = np.loadtxt('z.csv', delimiter=',')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Predicted z vs x and y
ax1.scatter(x, z_predicted, color='blue', alpha=0.5, label='vs x')
ax1.scatter(y, z_predicted, color='green', alpha=0.5, label='vs y')
ax1.set_xlabel('x or y')
ax1.set_ylabel('Predicted z')
ax1.set_title('Predicted z vs x and y (Degree 2, alpha=' + str(alpha_constant) + ')')
ax1.legend()

# Plot 2: Actual vs Predicted z (if z.csv is available)
if z_actual is not None:
    ax2.scatter(z_predicted, z_actual, color='red', alpha=0.5)
    ax2.plot([z_actual.min(), z_actual.max()], [z_actual.min(), z_actual.max()], 'k--', label='Ideal Fit')
    ax2.set_xlabel('Predicted z')
    ax2.set_ylabel('Actual z')
    ax2.set_title('Actual vs Predicted z (Degree 2, alpha=' + str(alpha_constant) + ')')
    ax2.legend()
else:
    ax2.text(0.5, 0.5, 'No z.csv available\nCannot plot actual vs predicted', 
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Actual vs Predicted z (Unavailable)')

plt.tight_layout()
plt.show()
