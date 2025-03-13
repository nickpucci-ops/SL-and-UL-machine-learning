# visualize x and y against z (not submitted)
import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('x_sample.csv', delimiter=',')
y = np.loadtxt('y_sample.csv', delimiter=',')
z = np.loadtxt('z_sample.csv', delimiter=',')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(x, z, color='blue', alpha=0.5, label='z vs x')
ax1.set_xlabel('x')
ax1.set_ylabel('Actual z')
ax1.set_title('Actual z vs x')
ax1.legend()


ax2.scatter(y, z, color='green', alpha=0.5, label='z vs y')
ax2.set_xlabel('y')
ax2.set_ylabel('Actual z')
ax2.set_title('Actual z vs y')
ax2.legend()

plt.tight_layout()
plt.show()
