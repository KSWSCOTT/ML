import tensorflow as tf
#   Multi-variable linear regression

#   5x3 input matrix
#   3x1 weight matrix
#   5x1 output matrix

x1_data = [73, 93, 89, 96, 73]
x2_data = [80, 88, 91, 98, 66]
x3_data = [75, 93, 90, 100, 70]
y_data = [152, 185, 180, 196, 142]

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
X3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([1]), name="weight1")
W2 = tf.Variable(tf.random_normal([1]), name="weight2")
W3 = tf.Variable(tf.random_normal([1]), name="weight3")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = X1 * W1 + X2 * W2 + X3 * W3 + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
#   if we use learning rate == 0.01 it can be divergence!!
#   Use proper learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X1: x1_data, X2: x2_data, X3: x3_data, Y: y_data})
    if step % 10 == 0:
        print ("Step: ", step, ", Cost: ", cost_val, '\nPrediction: ', hy_val)