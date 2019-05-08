 from sklearn.datasets.samples_generator import make_blobs
x,y=make_blobs(n_samples=500,centers=2,random_state=0,cluster_std=0.40)

import matplotlib.pyplot as plt

plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='plasma')

plt.show()