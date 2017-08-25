import tensorflow as tf

#   Hypothesis and cost function
#   Hypothesis --> Predict data from input data (literally hypothesis)
#   cost function -->   Average of square difference between hypothesis and y value (in Lab2)
#                       In machine learning, we should reduce value of cost function

#   Nodes: x_train, y_train, hypothesis, cost, train

#   X, Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight', dtype=tf.float32)   #   shape --> [1]
b = tf.Variable(tf.random_normal([1]), name='bias', dtype=tf.float32)

X = tf.placeholder(tf.float32)  #   shape = [None] --> Every vector can be used
Y = tf.placeholder(tf.float32)

#   Hypothesis is WX + b (in matrix, it's XW + b)
hypothesis = X * W + b
# hypothesis = tf.add(tf.matmul(X, W), b)   --> matmul need higher(or same) than rank 2 shape

#   cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#   ex) t = [1., 2.] --> tf.square(t) == [1., 4.]
#   tf.reduce_mean --> mean of matrix or vector
#   ex) t = [1., 2., 3., 4.] --> tf.reduce_mean == 2.5

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#   minimize cost
train = optimizer.minimize(cost)

sess = tf.Session()
#   We have to initialize variables before use (if we declare tf.Variable, now it's W and b)
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: x_train, Y: y_train})
    #   Every loop we train
    if step % 20 == 0:
        #   See every 20 times
        print ("Step: ", step, ", Cost: ", cost_val, ", W: ", w_val, ", b: ", b_val)

print ("sess.run(hypothesis, feed_dict={X: [200]}): ", sess.run(hypothesis, feed_dict={X: [200]}))

#   PLACEHOLDER!!