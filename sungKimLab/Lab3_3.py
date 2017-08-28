#   내장된 Gradient descent optimizer의 성능을 알아보기 위해 직접 gradient descent optimizer를 제작해보는 실험
import tensorflow as tf

#   Verity optimizer with our own gradient optimizer
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]))

hypothesis = X * W
#   Own made gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

#   TF's gradient
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#   100번의 epoch동안 내장된 gradient descent optimizer과 직접 구현한 것의 값이 얼마나 차이나는가 살펴본다.
for step in range(100):
    print ("Step: ", step, ", Cost and Weight for own and built-in: ", sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)