import tensorflow as tf
print (tf.__version__)
#   tensorflow version 1.1.0

hello = tf.constant("Hello, Tensorflow!")

sess = tf.Session()
print (sess.run(hello))
sess.close()

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)    #   if don't write data type, it's tf.float32
node3 = tf.add(node1, node2)

sess = tf.Session()
print ("sess.run(node1, node2): ", sess.run([node1, node2]))
print ("sess.run(node3): ", sess.run(node3))
sess.close()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

sess = tf.Session()

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
