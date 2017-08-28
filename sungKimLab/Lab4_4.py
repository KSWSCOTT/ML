#   Queue Runners!
#   여러 개의 데이터를 불러와서 batch로 대입하는 방식
#   batch의 개념

import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(["./Lab4_4/Lab4_4_2_copy.csv"], shuffle=False, name="filename_queue")

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#   이 형태의 data type로 불러와진다. (각 field의 data type)
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)
#   잡담/ Image의 경우 decode_image를 사용하면 되고 불러온 이미지에 대한 label은 아래 batch설정에서 2번째 성분에 label을 넣으면 된다.
#   batch를 사용해 데이터를 읽어온다. 랜덤한 지점부터 Batch_size만큼 연속적으로 데이터를 불러온다.
batch_size = 10

# train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=batch_size)
#   Shuffle_batch를 사용하면 연속적이지 않은 랜덤한 batch를 얻을 수 있다.
min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
#   이렇게 사용하면 랜덤으로 batch를 얻을 수 있다는건 알겠는데 이 코드에 대한 명확한 이해가 부족하다.
#   capacity: queue에 들어갈 수 있는 최대 수의 양(이게 늘면 메모리가 증가하겠군)
#   min_after_dequeue: queue에 들어갈 수 있는 가장 작은 개수 (항상 들어가있는 개수라고 보는 게 더 좋을 것 같음), dequeue이후 성분을 섞어주는 역할
train_x_batch, train_y_batch = tf.train.shuffle_batch([xy[0:-1], xy[-1:]], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.add(tf.matmul(X, W), b)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#   일반적으로 큐 관리하는 부분 1
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)



for step in range(2001):
    #   세션에서 batch를 불러옴
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print ("Step: ", step, ", Cost: ", cost_val, "\nPrediction: \n", hy_val, "\n X_Val: \n", x_batch)

#   일반적으로 큐 관리하는 부분 2
coord.request_stop()
coord.join(threads)