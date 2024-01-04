import cupy as np

a = np.ones((10000,1000))
# b = np.ones((10000,1000))
for i in range(100):
    a = 2*a+1