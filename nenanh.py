import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sys
from numpy.ma.core import size

#đọc ảnh vào bằng 1 hàm. Hàm này cho ra img là một ma trận số. Mỗi phần tử của ma trận là một pixel
img = mpimg.imread('girl3.jpg')
#tách ma trận số thành một mảng các vec tơ , mỗi vecto sẽ có 3 chiều với ảnh màu
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

#Bước này là xây dựng codeBook. Dùng 1 hàm phân cụm nào đó để tìm ra các codeVecto. ở đây mình dùng thuật toán Kmean
kmeans = KMeans(n_clusters=10).fit(X)
label = kmeans.predict(X)
img4 = np.zeros_like(X)

#Chỗ này là thay thế những mảng vec tơ ban đầu thành mảng các codebook tương ứng. Vec tơ thuộc cụm nào, gần vecto codebook nào nhất
#thì thay bằng code vecto đó

#Này là codebook có 10 vecto    
for k in range(10):
    img4[label == k] = kmeans.cluster_centers_[k]

#Đảo ngược lại mảng để trở về ma trận ảnh số
img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))

#ngược lại với hàm imread thì có hàm imsave để ghi ra ảnh từ ma trận
mpimg.imsave('ketqua.jpg',img5)

#có thể in ảnh bằng matlotlip
plt.imshow(img5)

plt.axis('off')
plt.show()
