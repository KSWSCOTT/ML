import tensorflow as tf

x_data = [[1,1], [0,0], [0,1], [1,0]]
y_data = [[0], [0], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        cost_val, w_val, _ = sess.run([cost, W, train], feed_dict={X: x_data, Y: y_data})
        if i % 20 == 0:
            print ("Step: ", i, ", Cost: ", cost_val, "\nWeight: ", w_val)