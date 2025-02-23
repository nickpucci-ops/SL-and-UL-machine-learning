import matplotlib.pyplot as plt
import pandas as pd

x = pd.read_csv('x.csv', header=None)
y = pd.read_csv('y.csv', header=None)
z = pd.read_csv('z.csv', header=None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

ax1.scatter(x, z, color='blue', alpha=0.5)
ax1.set_xlabel('x'); ax1.set_ylabel('z')
ax1.set_title('x vs z')

ax2.scatter(y, z, color='green', alpha=0.5)
ax2.set_xlabel('y'); ax2.set_ylabel('z')
ax2.set_title('y vs z')

plt.tight_layout()
plt.show()  
