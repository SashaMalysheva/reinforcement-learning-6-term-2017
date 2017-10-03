import numpy as np

a = np.array([1,0,0,0,1])
m = a==1
m = np.sum([m[i] * (2 ** i) for i in range(len(m))])
print(m)
