import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# learning_rate = 0.1
gradient = tf.reduce_mean((X * W - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)  #   tensorflow cannot substitute value using '=', we should use .assign

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
    #   Run update
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print ("Step: ", step, ", Cost: ", sess.run(cost, feed_dict={X: x_data, Y: y_data}), ", W: ", sess.run(W))
sess.close()