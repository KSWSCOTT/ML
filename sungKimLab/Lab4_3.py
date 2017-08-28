#   파일에서 데이터 읽어오기
#   Loading data from file!
import tensorflow as tf
#   넘파이(numpy)라이브러리를 이용해 데이터를 불러오고 관리한다.
import numpy as np

tf.set_random_seed(777)
#   Tensorflow uses random seed.
#   It means if we use same random seed number, then we can use same random number (sequences)
#   다른 언어와 마찬가지로 텐서플로우에서도 "랜덤"에 시드가 존재한다. 동일한 시드의 경우 랜덤으로 등장하는 숫자의 패턴이 같기 때문에 같은 랜덤시드를 사용하면 같은 랜덤 값을 갖게 된다.

#   skiprows = 1을 하게 되면 한 개의 행을 건너띄고 불러온다. (대게 첫 행은 아래의 값들과 다른 형태이기 때문에 유용하게 사용할 수 있다.
xy = np.loadtxt('Lab4_3.csv', delimiter=',', dtype=np.float32, skiprows=1)
#   '-1' means at the end of array
#   배열에서 -1은 맨 마지막으로 인식한다. -2, -3도 사용할 수 있다.
x_data = xy[:, 0:-1]    #   row, col
y_data = xy[:, [-1]]

print (x_data.shape, "\n", x_data, len(x_data))
print (y_data.shape, "\n", y_data)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.add(tf.matmul(X, W), b)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print ("Step: ", step, ", Cost: ", cost_val, "\nPrediction", hy_val)

#   내 점수가 얼마인지 대입하면 지금까지 학습한 weight로 내 최종 점수를 예측할 수 있다.
print ("My score: ", sess.run(hypothesis, feed_dict={X: [[100, 90, 100]]}))