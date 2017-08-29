# Sigmoid를 이용하여 학습하면 깊은 신경 망에서는 잘 작동하지 않는다는 문제가 생긴다.
# Vanishing gradient problem <-- Sigmoid의 문제
# ReLU로 해결
# 그래도 마지막 단은 Sigmoid를 사용한다.
# Sigmoid의 단점은 tanh가 조금 해결하고
# ReLU의 새로운 버전인 Leaky ReLU도 유용하다. max(0.1X, X)


#   Lec10-2
# 초기값을 어떻게 설정할 것인가.
# 1. 전부 다 0으로 맞추기.
# 2. RBM(Restricted Boatman Machine) R: 앞뒤로 한 층씩만 연결되어 있다.
#     인코더와 디코더의 값을 같게 하는 것. (오 신기...)
#     두 레이어씩 같게 만드는 과정을 거침
#     Pre-Trained model
#
#     Fine tuning -- RBM 이용
#     RBM 안써도 된다.
#     2010 논문에 "Understanding the difficulty of training deep feedforward neural networks", Xavier 이 제안
#     입/출력에 따라 weight 를 임의로 주는 것
#     weight = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
#                                                         여기에 /2만 해주면 개꿀...
#     실습때 xavier initializer 써봄
#
#     완벽한 초기값이 무엇일까에 대한 연구가 계속 진행된다.


#   Lec10-3
# Overfitting??? --> More training data, Regularization (L2 regularization)
#
# Dropout!?   --> 2014 Srivastava //  학습할 때 몇 개의 노드를 죽여버리자. 랜덤하게 몇 개씩 훈련을 시킴. (Feature),
# 한 개씩 학습하다가 마지막에 한 번에 다 구현.
# tf에서는 drop라는 layer에 한 번 넣으면 됨. rate는 보통 0.5, 학습할 때만 dropout시킨다. (학습할 때는 0.5, 평가 시 1을 준다.)
#
# 앙상블 (Ensemble)
# 초기값이 조금씩 다르게 학습(weight) 이후 예측값을 다 합치고 훈련한다.
#
# 이제 원하는 대로 네트워크롤 조립한다.
#
# 레즈넷 에서는 fast forward(Layer의 출력 값이 다음 입력에 참여한다.)
#
# 한 번에 많은 곳에서 학습 (CNN)
#
# 옆으로도 학습 (Recurrent Network) (RNN)
#
# 음? 내가 생각한건데, Class별로 다르게 Input으로 주고 한 번에 학습? (근데 이건 나중에 test하기가 애매하군.)
