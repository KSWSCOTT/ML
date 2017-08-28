#   Diabetes (Use kaggle data)
#   URL: https://www.kaggle.com/uciml/pima-indians-diabetes-database
import tensorflow as tf
import numpy as np
import math

xy = np.loadtxt("diabetes.csv", delimiter=",", dtype=np.float32, skiprows=1)
x = xy[:,0:-1]
y = xy[:,[-1]]
for i in range(len(x[:,0])):
    for j in range(len(x[0,:])):
        x[i,j] = (1 / (1 + math.exp(-x[i,j])))
print ("Convert Input Datas")

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")


#   Use sigmoid function
hypoothesis = tf.sigmoid(tf.add(tf.matmul(X, W), b))

cost = -tf.reduce_mean(Y * tf.log(hypoothesis) + (1 - Y) * tf.log(1 - hypoothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#   Accuracy computation
predicted = tf.cast(hypoothesis > 0.5, dtype=tf.float32)    #   if bigger than 0.5, TRUE. else FALSE
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    #   if same --> TRUE

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x, Y: y})
        if step % 200 == 0:
            print ("Step: ", step, ", Cost: ", cost_val)

    h, p, a = sess.run([hypoothesis, predicted, accuracy], feed_dict={X: x, Y: y})
    print ("\nHypothesis: ", h, "\nPredicted: ", p, "\nAccuracy: ", a)