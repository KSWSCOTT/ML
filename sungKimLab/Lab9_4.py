from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
import tensorflow as tf


nb_classes = 10


X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W1 = tf.Variable(tf.random_normal([784, nb_classes * 5]), name="weight")
b1 = tf.Variable(tf.random_normal([nb_classes * 5]), name="bias")
W2 = tf.Variable(tf.random_normal([nb_classes * 5, nb_classes * 2]), name="weight")
b2 = tf.Variable(tf.random_normal([nb_classes * 2]), name="bias")
W3 = tf.Variable(tf.random_normal([nb_classes * 2, nb_classes * 3]), name="weight")
b3 = tf.Variable(tf.random_normal([nb_classes * 3]), name="bias")
W4 = tf.Variable(tf.random_normal([nb_classes * 3, nb_classes]), name="weight")
b4 = tf.Variable(tf.random_normal([nb_classes]), name="bias")

Layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
Layer2 = tf.sigmoid(tf.matmul(Layer1, W2) + b2)
Layer3 = tf.sigmoid(tf.matmul(Layer2, W3) + b3)
hypothesis = tf.nn.softmax(tf.matmul(Layer3, W4) + b4)



# W = tf.Variable(tf.random_normal([784, nb_classes]), name="weight")
# b = tf.Variable(tf.random_normal([nb_classes]), name="bias")
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
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
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print ("Epoch: ", "%04d" % (epoch + 1), ", Cost: ", "{:.9f}".format(avg_cost))

    print ("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # import matplotlib.pyplot as plt
    # import random
    # r = random.randint(0, mnist.test.num_examples - 1)
    # print (r)
    # print ("Label: ", sess.run(tf.argmax(mnist.test.labels[r: r+1], 1)))
    # print ("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r: r+1]}))
    #
    # plt.imshow(mnist.test.images[r: r+1].reshape(28, 28), cmap="Greys", interpolation="nearest")
    # plt.show()

#   Last layer --> Softmax
#   1 Layer --> 82%
#   2 Layer (Sigmoid) -->87% (middle, 30)
#   2 Layer (Sigmoid) -->88% (middle, 50)

#   Setting layer is too hard T_T