import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 100
x = np.random.uniform(-10, 10, n_samples)
y = np.random.uniform(-10, 10, n_samples)

#cubic relationship with noise
z = x**3 + y**3 + 2 * x**2 * y + x * y**2 + np.random.normal(0, 50, n_samples)

pd.DataFrame(x).to_csv('x_sample.csv', header=False, index=False)
pd.DataFrame(y).to_csv('y_sample.csv', header=False, index=False)
pd.DataFrame(z).to_csv('z_sample.csv', header=False, index=False)

print("Generated")
