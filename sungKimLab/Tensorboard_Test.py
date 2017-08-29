import tensorflow as tf

# a = tf.constant(3.0)
# b = tf.constant(5.0)
# c = a * b
#
# c_summary = tf.summary.scalar("point", c)
# merged = tf.summary.merge_all()
#
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter("./board/sample_2", sess.graph)
#
#     result = sess.run([merged])
#     tf.global_variables_initializer()
#
#     writer.add_summary(result[0])

# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# add = tf.add(X, Y)
# mul = tf.multiply(X, Y)
#
# add_hist = tf.summary.scalar("add_scalar", add)
# mul_hist = tf.summary.scalar("mul_scalar", mul)
#
# merged = tf.summary.merge_all()
#
# with tf.Session() as sess:
#     tf.global_variables_initializer()
#
#     writer = tf.summary.FileWriter("./board/sample_2", sess.graph)
#
#     for step in range(100):
#         summary = sess.run(merged, feed_dict={X: step * 1.0, Y: 2.0})
#         writer.add_summary(summary, step)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
import tensorflow as tf

nb_classes = 10

with tf.name_scope("input") as scope:
    X = tf.placeholder(tf.float32, [None, 784])
with tf.name_scope("output") as scope:
    Y = tf.placeholder(tf.float32, [None, nb_classes])
W1 = tf.Variable(tf.random_normal([784, nb_classes]), name="weight")
b1 = tf.Variable(tf.random_normal([nb_classes]), name="bias")

# Layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
# Layer2 = tf.sigmoid(tf.matmul(Layer1, W2) + b2)
# Layer3 = tf.sigmoid(tf.matmul(Layer2, W3) + b3)
hypothesis = tf.nn.softmax(tf.matmul(X, W1) + b1)

w_hist = tf.summary.histogram("weight", W1)
b_hist = tf.summary.histogram("bias", b1)
h_hist = tf.summary.histogram("hypothesis", hypothesis)

with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    cost_sum = tf.summary.scalar("cost", cost)
with tf.name_scope("train") as scope:
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epoch = 15
batch_size = 100



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./board/MNIST", sess.graph)
    # for epoch in range(training_epoch):
    #     avg_cost = 0
        # total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(100001):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
        # avg_cost += c / total_batch
        if i % 10 == 0:
            summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys})
            writer.add_summary(summary, i)

        # print ("Epoch: ", "%04d" % (epoch + 1), ", Cost: ", "{:.9f}".format(avg_cost))


    print ("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

