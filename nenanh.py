import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sys
from numpy.ma.core import size

img = mpimg.imread('girl3.jpg')
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

#nén 
kmeans = KMeans(n_clusters=10).fit(X)
label = kmeans.predict(X)
print(label)
img4 = np.zeros_like(X)

    # replace each pixel by its center
for k in range(10):
    img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image

img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
plt.imshow(img5)

plt.axis('off')
plt.show()
