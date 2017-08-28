#   Hypothesis에 의한 Predict 값이 변함에 따라 Cost가 어떻게 변화하는가에 대해 알아보기 위한 실험

import tensorflow as tf
#   Use matplotlib to draw a graph
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

#   Declare W as placeholder
W = tf.placeholder(tf.float32)

#   Hypothesis (WX)
hypothesis = X * W

#   Same cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#   Save W and cost value
W_val = []
cost_val = []

#   Use W as variable (-3 to 5)
for i in range(-30, 50):
    feed_W = i * 0.1
    #   cost and W
    #   We know (W == 1) is correct answer, so if (feed_W == 1) --> cost == 0, and cost is gonna higher when feed_W away from 1 (2nd order)
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

#   Plot W_val and cost_val (it's 2nd order graph)
plt.plot(W_val, cost_val)
plt.show()

#   So we should use gradient descent algorithm to minimize cost!!
