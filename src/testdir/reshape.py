import numpy as np

h = 64
w = 64
n = 42

v1 = np.zeros(h*w*n)
v2 = v1.reshape(n, h*w)

v3 = np.zeros(h*w*n).reshape(n, h*w)

print(v1.shape, v2.shape, v3.shape)
