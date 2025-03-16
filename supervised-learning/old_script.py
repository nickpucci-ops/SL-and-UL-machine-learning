import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


x = pd.read_csv('x.csv', header=None).values.flatten()#.values.flatten => shapes as 50 rows
y = pd.read_csv('y.csv', header=None).values.flatten()
z = pd.read_csv('z.csv', header=None).values.flatten()


#Polynomial Regression

X = np.column_stack((x,y)) #combine x and y into the feature matrix (50, 2)

# Scale features and target (necessary for polynomial regression)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)  # Scale x, y

scaler_z = StandardScaler(with_mean=True, with_std=True)
z_scaled = scaler_z.fit_transform(z.reshape(-1, 1)).ravel()  # Scale z (billions)

# Create 2x2 subplots for visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

#plot 1
ax1.scatter(x, z, color='blue', alpha=0.5)
ax1.set_xlabel('x'); ax1.set_ylabel('z'); ax1.set_title('x vs z')

#plot 2
ax2.scatter(y, z, color='green', alpha=0.5)
ax2.set_xlabel('y'); ax2.set_ylabel('z'); ax2.set_title('y vs z')

poly2 = PolynomialFeatures(degree=2)# degree 2 => (1, x, y, x^2, xy, y^2)
X_poly2 = poly2.fit_transform(X_scaled) #transform to (50, 6) for new columns

#ridge regression
model2 = Ridge(alpha=1.0)#alpha is a constant that multiplies the L2 term and controls the regularization strength, alpha = 0 is linear regression
model2.fit(X_poly2, z_scaled)


# Predict (scaled) and unscale for degree 2
z_pred2_scaled = model2.predict(X_poly2) #We want to train the model on scaled data first, since we are dealing with large discrepancies
z_pred2 = scaler_z.inverse_transform(z_pred2_scaled.reshape(-1, 1)).ravel() #then we match the trained scaled data to our original interpretation

#plot 3: Actual vs Predicted z (degree 2, unscaled, Predicted z on x, Actual z on y)
ax3.scatter(z_pred2, z, color='red', alpha=0.5)  # Predicted z on x, Actual z on y
ax3.plot([z.min(), z.max()], [z.min(), z.max()], 'k--')  # 45-degree line
ax3.set_xlabel('Predicted z'); ax3.set_ylabel('Actual z')
ax3.set_title('Actual vs Predicted z (Degree 2, alpha=1.0)')

#degree 3
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X_scaled)

model3 = Ridge(alpha=100.0)  # Stronger regularization constant for more terms
model3.fit(X_poly3, z_scaled)

# Predict (scaled) and unscale for degree 3
z_pred3_scaled = model3.predict(X_poly3)
z_pred3 = scaler_z.inverse_transform(z_pred3_scaled.reshape(-1, 1)).ravel()

#plot 4: Actual vs Predicted z (degree 3, unscaled, Predicted z on x, Actual z on y)
ax4.scatter(z_pred3, z, color='purple', alpha=0.5)  # Predicted z on x, Actual z on y
ax4.plot([z.min(), z.max()], [z.min(), z.max()], 'k--')  # 45-degree line
ax4.set_xlabel('Predicted z'); ax4.set_ylabel('Actual z')
ax4.set_title('Actual vs Predicted z (Degree 3, alpha=100.0) (Potential Overfit)')

plt.tight_layout()
plt.show()
print("Saving plot 3 (Degree 2) to z-predicted.csv")
pd.DataFrame(z_pred2).to_csv('z-predicted.csv', header=False, index=False)
