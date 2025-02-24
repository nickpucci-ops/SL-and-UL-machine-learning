
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


x = pd.read_csv('x.csv', header=None).values.flatten()#.values.flatten => shapes as 50 rows
y = pd.read_csv('y.csv', header=None).values.flatten()
z = pd.read_csv('z.csv', header=None).values.flatten()


#Display first two subplots (x vs z, y vs z)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

ax1.scatter(x, z, color='blue', alpha=0.5)
ax1.set_xlabel('x'); ax1.set_ylabel('z')
ax1.set_title('x vs z')

ax2.scatter(y, z, color='green', alpha=0.5)
ax2.set_xlabel('y'); ax2.set_ylabel('z')
ax2.set_title('y vs z')

plt.tight_layout()
plt.show()

#Polynomial Regression

X = np.column_stack((x,y)) #combine x and y into the feature matrix (50, 2)

poly = PolynomialFeatures(degree=2) # degree 2 => (1, x, y, x^2, xy, y^2)
X_poly = poly.fit_transform(X)# transform to (50, 6) for new columns

#Using Ridge regression
model = Ridge(alpha=1000.0)#alpha is a constant that multiplies the L2 term and controls the regularization strength, alpha = 0 is linear regression
model.fit(X_poly, z)

z_pred = model.predict(X_poly) #Predict the z values for all 50 samples

# Visualize predictions vs actual
plt.scatter(z, z_pred, color='red', alpha=0.5)
plt.plot([z.min(), z.max()], [z.min(), z.max()], 'k--')  # 45-degree line
plt.xlabel('Actual z'); plt.ylabel('Predicted z')
plt.title('Actual vs Predicted z')
plt.show() 


plt.scatter(z_pred, z, color='red', alpha=0.5)
plt.plot([z.min(), z.max()], [z.min(), z.max()], 'k--')  # 45-degree line
plt.xlabel('Actual z'); plt.ylabel('Predicted z')
plt.title('Actual vs Predicted z')
plt.show()  
