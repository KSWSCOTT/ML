import tensorflow as tf

#   XOR and NN

x_data = [[1,1], [0,0], [0,1], [1,0]]
y_data = [[0], [0], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10001):
        cost_val, w_val, _ = sess.run([cost, W, train], feed_dict={X: x_data, Y: y_data})
        if i % 100 == 0:
            print ("Step: ", i, ", Cost: ", cost_val, "\nWeight: ", w_val)
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print ("Hypothesis: ", h, "\nCost: ", c, "\nAccuracy: ", a)

#   XOR can't solved by single perceptron