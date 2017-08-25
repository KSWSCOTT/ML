import tensorflow as tf

#   Loading data from file!

import numpy as np
tf.set_random_seed(777)
#   Tensorflow uses random seed.
#   It means if we use same random seed number, then we can use same random number (sequences)
xy = np.loadtxt('Lab4_3.csv', delimiter=',', dtype=np.float32, skiprows=1)
#   '-1' means at the end of array
x_data = xy[:, 0:-1]    #   row, col
y_data = xy[:, [-1]]

print (x_data.shape, "\n", x_data, len(x_data))
print (y_data.shape, "\n", y_data)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.add(tf.matmul(X, W), b)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print ("Step: ", step, ", Cost: ", cost_val, "\nPrediction", hy_val)

print ("My score: ", sess.run(hypothesis, feed_dict={X: [[100, 90, 100]]}))