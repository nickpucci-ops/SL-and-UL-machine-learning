#generate more x, y, z samples for testing purposes (not submitted)
import numpy as np
import pandas as pd

np.random.seed(32)

n_samples = 100
x = np.sort(np.random.uniform(-10, 10, n_samples))
y = np.sort(np.random.uniform(-10, 10, n_samples))

#cubic relationship with noise (amplified weight with 4x^3, 4y^3, and cross terms 12x^2y, 12xy^2)
z = 4 * x**3 + 4 * y**3 + 12 * x**2 * y + 12 * x * y**2 + np.random.normal(0, 2500, n_samples)

pd.DataFrame(x).to_csv('data/x_sample.csv', header=False, index=False)
pd.DataFrame(y).to_csv('data/y_sample.csv', header=False, index=False)
pd.DataFrame(z).to_csv('data/z_sample.csv', header=False, index=False)
print("generated")
