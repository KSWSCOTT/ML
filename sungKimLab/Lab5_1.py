#   Logistic (binary) classification
#   Binary classification is similar to linear regression, but it has output function (like softmax function)
#   Use sigmoid function!
#   Sigmoid: curved in two directions, like the letter "S", or the Greek sigma (ref. Lec5_1)
#   가설이 바뀜에 따라 cost function이 조금 바뀌게 된다. (울퉁불퉁한 2차함수 형태, SGD를 사용할 때 문제가 생긴다.)

#   Sigmoid의 Cost function을 그린 코드

import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np

X = [[-1], [0], [1]]
Y = [[(1/(1+math.exp(-1)))], [(1/(1+math.exp(0)))], [(1/(1+math.exp(1)))]]

W = tf.placeholder(tf.float32)

hypothesis = tf.sigmoid(X * W)

cost = tf.reduce_mean(tf.square(hypothesis - Y))
# cost = tf.reduce_mean(-tf.reduce_sum(- Y * tf.log(hypothesis) - (1 - Y) * tf.log(1 - hypothesis)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-100, 100):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()

#   Same code as Lab3_1