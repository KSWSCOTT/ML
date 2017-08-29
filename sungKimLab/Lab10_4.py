#   Use Dropout!!
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

keep_prob = tf.placeholder(tf.float32)

# W1 = tf.Variable(tf.random_normal([784, 256]), name="weight1")
W1 = tf.get_variable("weight1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]), name="bias1")
Layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
Layer1 = tf.nn.dropout(Layer1, keep_prob=keep_prob)

# W2 = tf.Variable(tf.random_normal([256, 256]), name="weight2")
W2 = tf.get_variable("weight2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]), name="bias2")
Layer2 = tf.nn.relu(tf.matmul(Layer1, W2) + b2)
Layer2 = tf.nn.dropout(Layer2, keep_prob=keep_prob)

# W2 = tf.Variable(tf.random_normal([256, 256]), name="weight2")
W3 = tf.get_variable("weight3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]), name="bias3")
Layer3 = tf.nn.relu(tf.matmul(Layer2, W3) + b3)
Layer3 = tf.nn.dropout(Layer3, keep_prob=keep_prob)

# W2 = tf.Variable(tf.random_normal([256, 256]), name="weight2")
W4 = tf.get_variable("weight4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]), name="bias4")
Layer4 = tf.nn.relu(tf.matmul(Layer3, W4) + b4)
Layer4 = tf.nn.dropout(Layer4, keep_prob=keep_prob)

# W3 = tf.Variable(tf.random_normal([256, nb_classes]), name="weight3")
W5 = tf.get_variable("weight5", shape=[512, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]), name="bias5")
Layer5 = tf.matmul(Layer4, W5) + b5
Layer5 = tf.nn.dropout(Layer5, keep_prob=keep_prob)

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Layer3), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Layer5, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.arg_max(Layer5, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epoch = 15
batch_size = 100


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epoch):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch

        print ("Epoch: ", "%04d" % (epoch + 1), ", Cost: ", "{:.9f}".format(avg_cost))

    print ("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    # import matplotlib.pyplot as plt
    # import random
    # r = random.randint(0, mnist.test.num_examples - 1)
    # print (r)
    # print ("Label: ", sess.run(tf.argmax(mnist.test.labels[r: r+1], 1)))
    # print ("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r: r+1]}))
    #
    # plt.imshow(mnist.test.images[r: r+1].reshape(28, 28), cmap="Greys", interpolation="nearest")
    # plt.show()
