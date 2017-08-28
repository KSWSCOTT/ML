import tensorflow as tf

#   Hypothesis and cost function
#   Hypothesis --> Predict data from input data (literally hypothesis)
#   cost function -->   Average of square difference between hypothesis and y value (in Lab2)
#                       In machine learning, we should reduce value of cost function

#   가설과 코스트 함수
#   가설 --> 입력 데이터로부터 결과값을 예측하기 위한 것
#   코스트 함수 --> 가설과 결과값의 차의 제곱의 평균 (Lab2에서)
#                   기계학습에서는 학습을 통해 줄이고자 하는 값, 학습이 잘 되었는지 알아볼 수 있는 값

#   Nodes: x_train, y_train, hypothesis, cost, train

#   X, Y data
#   Placeholder을 사용할 것이기 때문에 x, y 데이터를 미리 배열로 정의해 둔다.
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
#   변수(Variable) 형태의 변수를 쓰는 경우 세션을 시작하고 항상 변수를 초기화해 주는 작업을 수행하여야 한다.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: x_train, Y: y_train})
    #   Every loop we train
    if step % 20 == 0:
        #   See every 20 times
        print ("Step: ", step, ", Cost: ", cost_val, ", W: ", w_val, ", b: ", b_val)

print ("sess.run(hypothesis, feed_dict={X: [200]}): ", sess.run(hypothesis, feed_dict={X: [200]}))

#   PLACEHOLDER!!