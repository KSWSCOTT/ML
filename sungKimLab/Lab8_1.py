import tensorflow as tf

#   BroadCasting
#   다른 형태의 배열에서 연산을 가능하게 하는 것
#   ex) [[1, 2]] + 3 --> [[4, 5]]의 결과를 내보임

#   mean  값을 구할 때 tf.float32로 선언하면 좋음.

#   argmax 에서 axis 잘 설정하기.

#   ones_like, zeros_like

#   expand_dims, squeeze

# for x,y in zip([1,2,3], [4,5,6]):
#     print (x,y)
#     #   1 4
#     #   2 5
#     #   3 6