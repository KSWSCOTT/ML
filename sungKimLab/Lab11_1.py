import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[1.], [2.], [3.]],
                  [[4.], [5.], [6.]],
                  [[7.], [8.], [9.]]], dtype=np.float32)
print (image.shape)
print (image)
image = image.reshape(1, 3, 3, 1)
print (image.shape)
print (image)

weight = tf.constant([[[1., 10., -1.], [1., 10., -1.]],
                       [[1., 10., -1.], [1., 10., -1.]]], dtype=tf.float32)
# weight = weight.reshape(1, 2, 2, 1)
weight = tf.reshape(weight, [2,2,1,3])  #   한 장으로부터 세 장의 이미지가 나온다.
print (weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding="SAME") #   or padding="VALID"
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)

pool = tf.nn.max_pool(conv2d_img, ksize=[1,2,2,1], strides=[1,1,1,1], padding="SAME")
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
    print (conv2d_img.shape)
    print (pool_img.shape)
    print (pool_img.reshape(2, 2))
    plt.subplot(1, 3, i + 1)
    plt.imshow(pool_img.reshape(2,2), cmap="gray")
    #   12분58초부터
plt.show()