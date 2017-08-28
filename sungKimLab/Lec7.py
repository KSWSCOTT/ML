#   Learning rate, Regularization, Overfitting
import tensorflow as tf

#   Gradient descent --> learning rate --> overshooting (by big step)
#   데이터에 따라 적절한 학습률이란 다르다. 그렇기 떄문에 data에 따라 선정하거나
#   학습이 될수록 학습률을 줄여가는 방법도 있다.

#   데이터 전처리에 관하여.
#   입력 데이터의 편차가 매우 크다면, 이상한 형태의 등고선이 나올 것이다.
#   Normalize 해준다. (+ zero-centered data) Normalize해줄 떄는 out-lier을 제거하자.
#   Standardization을 하자.  python의 경우
#   X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std() 를 해준다.

#   OverFitting
#   너무 데이터에 의존하게 학습하는 경우.
#   머신러닝은 이후 테스트할 데이터를 잘 맞춰야하기 때문에 (일반화)
#   Overfitting 이 쉽게 발생할 수 있다.
#   데이터를 많이 가지고 있거나.
#
#   일반화를 시켜야 한다. Boundary를 좀 '편다'는 것이다. Cost함수 뒤에 한 개의 텀을 추가해 준다.
#   각 element의 값을 제곱하고 람다를 곱하고 최소화 시킨다. 오..?
#   람다: 0, 안한다 / 0.01좀 한다 / 1 많이 한다.

#   Training, Test Set
#   Performance evaluation

#   Validation set을 모의고사라고 말하네 재미있다 ㅎ

#   Online learning?
#   Training set 이 100만개가 있다고 할 때 한 번에 다 학습하니 ㅄ같잖아? 너무 많고
#   그래서 10만개씩 나눠서 학습 시키며 weight를 계속 저장해 두는 것 (있는 데이터에 추가로 학습)

#   최근 이미지 인식은 대략 95%의 성능