import tensorflow as tf

#   Verity optimizer with our own gradient optimizer
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]))

hypothesis = X * W
#   Own made gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

#   TF's gradient
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print ("Step: ", step, ", Cost and Weight for own and built-in: ", sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)