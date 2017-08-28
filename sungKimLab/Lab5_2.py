import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

#   Use sigmoid function
hypoothesis = tf.sigmoid(tf.add(tf.matmul(X, W), b))

cost = -tf.reduce_mean(Y * tf.log(hypoothesis) + (1 - Y) * tf.log(1 - hypoothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

#   Accuracy computation
predicted = tf.cast(hypoothesis > 0.5, dtype=tf.float32)    #   if bigger than 0.5, TRUE. else FALSE
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    #   if same --> TRUE

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print ("Step: ", step, ", Cost: ", cost_val)

    h, p, a = sess.run([hypoothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print ("\nHypothesis: ", h, "\nPredicted: ", p, "\nAccuracy: ", a)