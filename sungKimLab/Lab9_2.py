import tensorflow as tf

#   XOR and NN

x_data = [[1,1], [0,0], [0,1], [1,0]]
y_data = [[0], [0], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([2, 2]), name="weight1")
b1 = tf.Variable(tf.random_normal([2]), name="bias1")
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name="weight1")
b2 = tf.Variable(tf.random_normal([1]), name="bias1")
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)   #   include softmax T_T
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10001):
        cost_val, w1_val, w2_val, _ = sess.run([cost, W1, W2, train], feed_dict={X: x_data, Y: y_data})
        if i % 100 == 0:
            print ("Step: ", i, ", Cost: ", cost_val, "\nWeight1: ", w1_val, "\nWeight2: ", w2_val)
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print ("Hypothesis: ", h, "\nCost: ", c, "\nAccuracy: ", a)