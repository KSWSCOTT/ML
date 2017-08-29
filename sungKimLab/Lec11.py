#   Padding --> 이미지가 급격하게 작아지는 것을 방지
#   CNN의 기원에 대한 정리가 필요함
#   한 번의 CNN Layer 에서 여러 개의 weight가 있을 수 있음? 5x5x3 x6 weight layer을 아직 이해하지 못함
#
#   Pooling ~= sampling
#   Max, Min Pooling이 존재.
#
#   마지막 FC 로 Softmax classifier 을 이용해 결과를 판단.

#   cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html

#   Fast Forward 를 사용한 ResNet 에서는 음, 뭔가 여러 개의 레이어를 사용하지만 한 개에서 사용한 것 같은 느낌이 들긴 한다. 왜 잘 되는지는 아직 잘 모르나 무튼 잘된다.