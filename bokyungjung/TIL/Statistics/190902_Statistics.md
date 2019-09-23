
# 지난 시간 복습
* 그래프 모형 : 두 변수 간의 관계(영향)을 그래프로 표현. 품질이 만족도에 영향을 준다는 것을 품질 -> 만족도로 표현.
* 선형 모형 : 계산이 쉽고 간단하기 때문에 상대적으로 틀릴 확률도 낮다. y = ax + b 등의 식으로 표현.

## 이항분포
* A를 할 때 B일 확률이 p
* A를 N번 했을 때 B가 X번 발생할 확률 분포
    * ex1) 고객이 방문할 확률이 30%(p)일 때, 100명의 고객 중 방문한 고객의 수는?
    * ex2) 강아지가 짜증낼 확률이 10%(p)일 때, 10마리의 강아지 중 짜증내는 강아지가 각각 0, 1, 2마리일 때는?
    * ex3) 


```python
import numpy
from numpy.random import binomial
binomial(100, .3)
data = binomial(100, .3, 100)
```

## 정규분포


```python
from numpy.random import normal
data2 = normal(50, 15, 100)
```

# 최적화를 이용한 파라미터 추정
* 최적화 : 제약조건 아래서 목표함수를 최대화(최소화)하는 것.
    * ex)L = J(p) (J : 오차, p : 파라미터), 오차를 줄이는 파라미터를 찾아 오차를 최소화하는 것. 제약 조건은 0<=p<=1

## 평균 오차 제곱 (MSE)
* y가 실제값이고 y^(y-hat)이 예측값일 때
* MSE : (y-y^)^2
* cf.) MAE : |y-y^|
* MSE에서 예측값을 평균으로 두면 결국 분산을 구하는 식과 같아진다.
* 예측값에 대한 오차를 줄이는 과정.
* 파라미터가 바뀌면 예측이 바뀌고, 예측이 바뀌면 오차가 바뀌고, 오차가 바뀌면 MSE가 바뀐다.
* 무한대 범위의 연속인 y를 선형 모형(y = ax + b)을 통해 최저 MSE를 추정하는 것.


```python
예측 = 100
numpy.mean((data2 - 예측) ** 2) #MSE
```

```python
예측2 = 20
numpy.mean((data2 - 예측2) ** 2)
평균 = numpy.mean(data2)
numpy.mean((data2 - 평균) ** 2) #평균으로 예측하면 오차가 가장 적다
```

정규분포에서 나온 데이터가 있을 때 모분포의 평균을 어떻게 추정하나?  
MSE가 가장 작아지는 예측값을 모분포의 평균으로 추정  
(수학적인 이유로 데이터의 평균이 MSE가 가장 작음)  
(그래서 데이터의 평균이 모분포의 평균에 대한 가장 좋은 추정치가 된다)


```python
from scipy.optimize import minimize
def mse(m) :
    return (numpy.mean(data2-m) ** 2)

mse(30)
minimize(mse, 100) #100은 초기값, 이를 기준으로 오차를 줄여보게 하는 것
```

```python
def mae(m) :
    return numpy.mean(numpy.abs(data2-m))

minimize(mae, 100)
```

## 영화 예시 - 파라미터가 2개인 경우


```python
연기력 = numpy.array([4, 4, 3, 5, 2])
흥행 = numpy.array([5, 4, 3, 5, 1])
def mse(param) :
    a, b = param
    예측치 = 연기력 * a + b
    return numpy.mean((흥행 - 예측치) ** 2)

mse([1, 2])

# minimize는 parameter를 하나만 받을 수 있어서 함수를 만들 때 param에 여러 개의 파라미터를 넣어야한다.
minimize(mse, (1, 2))

mse([1.38461534, -1.38461524])
```


## 파라미터가 3개인 경우
* ex) 독립변수 : 연기력, CG수준 / 종속변수 : 흥행
    * a * 연기력 + b * CG수준 + c = 흥행
* 독립변수 : 맛, 배달 속도 / 종속변수 : 별점


```python
# 배달 어플 별점 예시

배달속도 = numpy.array([5, 3, 1, 4, 4])
맛 = numpy.array([2, 4, 1, 5, 3])
별점 = numpy.array([3, 4, 1, 5, 2])

def mse(param) :
    a, b, c = param
    예측치 = a * 배달속도 + b * 맛 + c
    return numpy.mean((별점 - 예측치) ** 2)

minimize(mse, (1, 2, 3))
```

# 우도 (Likelihood)
* 어떠한 가정에서 특정 데이터가 관찰될 확률
* 파라미터를 가정했을 때 데이터가 관찰될 조건부 확률
* 최대우도법(Maximum Likelihood, ML)
* y가 0~1 사이에 있는 로지스틱 선형 모형(y = expit(ax + b))을 통해 최대 우도를 추정하는 것.


```python
from scipy.stats import binom

# binom.pmf(데이터, n, p), 동전 2개를 던졌을 때 앞면이 나올 확률 20%일 때 앞면이 하나도 안 나올(0) 확률
binom.pmf(0, 2, .2) # 0의 우도

# binom.pmf(데이터, n, p), 동전 2개를 던졌을 때 앞면이 나올 확률 20%일 때 앞면이 0, 1, 2개 나올 각각의 확률
binom.pmf([0, 1, 2], 2, .2)
```

## 로그 우도
* numpy.log(3) + numpy.log(4) 는 numpy.log(12)의 값과 같다.
* 곱셈보다 덧셈이 계산이 더 간편하기 때문에 로그 우도를 활용하는 것.


```python
data = binomial(2, .2, 100)
p = .2
like = binom.pmf(data, 2, p)

# maximize 함수는 없가 때문에 -를 붙여 minimize 함수로 비교할 수 있게 변환
-numpy.sum(numpy.log(like)) 

def lk(p) :
    like = binom.pmf(data, 2, p)
    return -numpy.sum(numpy.log(like))

minimize(lk, .5, bounds=[(0.01, 0.99)])
```

## 로지스틱 회귀분석 예시 1


```python
from scipy.special import expit # 로지스틱

김치찌개 = numpy.array([5, 1, 3, 4, 5, 3]) # 김치 5 4 3 2 1 된장
아메리카노 = [0, 1, 0, 1, 1, 1] # 아메리카노 선호 1, 라떼 선호 0

a = .3
b = .2

# 예측(아메리카노를 좋아할 확률)이 0~1 사이의 결과로 나올 수 있게
예측 = expit(a * 김치찌개 + b) 
```


위의 결과에서, 첫번째 사람은 아메리카노를 싫어하고 예측 결과 아메리카노를 좋아할 확률이 0.84라고 나왔으므로 첫번째 사람의 우도는 0.16이다.  
두번째 사람은 아메리카노를 좋아하고 예측 결과 아메리카노를 좋아할 확률이 0.62라고 나왔으므로 우도는 0.62이다.

## 로지스틱 회귀분석 예시 2


```python
# 잘한다 5 4 3 2 1 못한다
실력 = numpy.array([1, 5, 3, 4, 5, 3]) 
# 합격 1, 불합격 0
합격 = numpy.array([0, 1, 0, 0, 1, 1]) 

a = .3
b = .2

합격확률 = expit(a * 학생의실력 + b) 

합격확률 * 합격 
(1 - 합격확률) * (1 - 합격) 
```
```python
합격자우도 = numpy.log(합격확률) * 합격
탈락자우도 = numpy.log(1 - 합격확률) * (1 - 합격)
우도 = 합격자우도 + 탈락자우도
-numpy.sum(우도)
```

```python
def lk(param) :
    a, b = param
    합격확률 = expit(a * 실력 + b)
    합격자우도 = numpy.log(합격확률) * 합격
    탈락자우도 = numpy.log(1-합격확률) * (1-합격)
    우도 = 합격자우도 + 탈락자우도
    return -numpy.sum(우도)

minimize(lk, [0, 0]) # minimize(함수, [초기값])
```

```python
# 점수가 5점인 특정 인물의 합격 확률 구하기
expit(1.30380145 * 5 -4.70457161)
```

```python
# 추정은 가능하나 이 예시에 도입하려면 논리에는 약간 억지가 있는 MSE
def mse(param) :
    a, b = param
    합격확률 = expit(a * 실력 + b) # 로지스틱 선형 모형
    return numpy.mean((합격확률 - 합격) ** 2)

minimize(mse, [0, 0])
```

## 예시


```python
연령 = 17
소득 = 100
expit(0.01 * 연령 + 0.02 * 소득 - 2)
```


실습용 데이터셋이 필요하다면 : https://archive.ics.uci.edu/ml/index.php

## Automobile 데이터 실습


```python
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./imports-85.data', header=None)

안전등급 = df[0]
연비 = df[24]

def mse(param):
    a, b = param
    예측 = a * 연비 + b
    return (numpy.mean((안전등급 - 예측) ** 2))

minimize(mse, [0, 0])
```