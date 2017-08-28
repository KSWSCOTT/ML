#   Softmax Classification
#   Softmax function --> 확률로 결과값을 출력
import tensorflow as tf

#   Use one-hot encoding
x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([4, 3]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([3]), dtype=tf.float32)

hypothesis = tf.nn.softmax(tf.add(tf.matmul(X,W), b))
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print ("Step: ", step, ", Cost: ", sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    a = sess.run([hypothesis], feed_dict={X: [[1, 2, 1, 0],
                                              [1,3,4,2],
                                              [1,2,3,1]]})
    print (a)
    # print ("Predict: ", a)
    print (a, sess.run(tf.arg_max(a[0], 1)))



# sess = tf.Session()
#
# for i in range(1001):
#     pass
#
# sess.close()