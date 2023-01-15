#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import skimage
from skimage import io
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time
import seaborn as sns; sns.set()

n_colors = 5

plt.style.use("ggplot")
path = "C:/Users/Asus/Documents/DL_img/ccs.tif"

#img = Image.open(open(path,'rb'))
#img.show()

area = imread(path)
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(area)
print(area.shape)
print(area.size)
#plt.show()

area_qt = np.array(area, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(area_qt.shape)
assert d == 3
area_img_array = np.reshape(area_qt, (w * h, d))

kmeans = KMeans(n_clusters=n_colors, n_init=100, random_state=0).fit(area_img_array)
labels = kmeans.predict(area_img_array)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)

plt.figure(1)
plt.clf()
plt.axis("off")
plt.title("Original image")
plt.imshow(area)
plt.show()

plt.figure(2)
plt.clf()
plt.axis("off")
plt.title(f"Quantized image ({n_colors} colors, K-Means)")
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(area)
plt.axis("off")
plt.title("Original image")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.axis("off")
plt.title(f"Quantized image ({n_colors} colors, K-Means)")

plt.show()

fig, (axA,axB)= plt.subplots(1,2)
axA.imshow(area)
axA.set_title("Original image")
axA.axis("on")
axA.grid(False)
axB.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
axB.set_title(f"Quantized image ({n_colors} colors, K-Means)")
axB.axis("on")
axB.grid(False)

plt.show()

