#   한글 코맨트가 작성이 되는군요.
#   텐서플로우 기초에 관한 내용

#   텐서플로우 라이브러리를 tf란 이름으로 불러옴
import tensorflow as tf

#   텐서플로우의 버전을 확인
print (tf.__version__)
#   tensorflow version 1.1.0


#   hello에 Hello TF를 대입 (상수 자료형)
hello = tf.constant("Hello, Tensorflow!")

#   그래프를 실행시키기 위해 세션을 연다.
sess = tf.Session()
#   세션을 연 후 실행(run)을 이용해 그래프 내에서 작업을 수행한다.
print (sess.run(hello))
#   세션을 닫는다.
sess.close()

#   상수 자료형 두 개를 이용해서 그래프 내부가 어떻게 작용하나 알아본다.
#   node1, node2에는 상수 자료형을 입력한다. (입력된 자료형은 float32)
#   node3은 둘을 더하는 작업을 한다. (node1과 node2가 edge를 따라 node3으로 이동해 node3에서 덧셈 연산을 수행한다.)
#   실제 수행은 세션을 연 후 수행된다.
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)    #   if don't write data type, it's tf.float32
node3 = tf.add(node1, node2)

#   세션을 열고 위에서 언급한 작업을 수행한다.
sess = tf.Session()
print ("sess.run(node1, node2): ", sess.run([node1, node2]))
print ("sess.run(node3): ", sess.run(node3))
sess.close()


#   텐서플로우에서 자료형으로 사용할 수 있는 placeholder의 사용법
#   a,b를 placeholder로 선언하고 세션에서 a와 b에 어떤 값이 들어갈지 선택한다.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

sess = tf.Session()

#   아래와 같이 feed_dict를 이용하여 placeholder에 원하는 값을 입력한다. 자료의 형태는 float32만 만족하면 배열로 써도 무방하다.
print ("sess.run(adder_node, feed_dict={a: 3, b: 4.5}): ", sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print ("sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}): ", sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

sess.close()

#   tensor Ranks, Shapes, Types
#   Ranks
#   order --> Scalar, Vector, Matrix, 3-Tensor, ...
#   ex) [1., 2., 3.] --> rank 1 tensor
#       [[1., 2., 3.], [4., 5., 6.]] --> rank 2 tensor
#   Shapes
#   ex) [[1., 2., 3.], [4., 5., 6.]] --> shape [2, 3] (2: row, 3: col)
#       [[[1., 1., 1.], [2., 2., 2.]], [[3., 3., 3.], [4., 4., 4.]]]    --> shape [2, 2, 3]
#   Types
#   ex) tf.float32 --> 32 bit floating variable