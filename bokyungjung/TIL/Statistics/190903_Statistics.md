
# 신뢰구간


```python
import numpy
from numpy.random import uniform

# 0~100 사이에서 30개의 수를 고르게 뽑기
x = uniform(0, 100, 30)
numpy.mean(x) # x의 평균
```

```python
#for문에 i를 써도 되지만 그 변수는 for문 안에서 쓸 일이 없기 때문에 _로 써도 된다

for _ in range(50):
    x = uniform(0, 100, 30) # 데이터 x 생성(샘플링)
    m = numpy.mean(x)       # x의 평균
    print(m)
```

```python
import seaborn

ms = []                        # 빈 리스트 생성
for _ in range(50) :
    x = uniform(0, 100, 30)
    m = numpy.mean(x)
    ms.append(m)               # 리스트에 평균 추가

```


```python
seaborn.distplot(x) # 하나의 샘플에서 데이터는 어떻게 퍼져있는가?
```

```python
seaborn.distplot(ms) # 샘플링할 때마다 평균이 어떻게 달라지는가?
```

* 모분포, Sample 분포, Sampling 분포는 각각 다른 개념이니 헷갈리지 말자
* Sampling 분포 (Sampling Distribution) : Sample을 모아 통계치를 추정한 결과
* 모분포와 Sampling 분포는 형태가 다를 수도 있다.

## 95% 신뢰구간


```python
import sklearn
numpy.quantile(ms, .025) # 하위 2.5%
```

```python
numpy.quantile(ms, .975) # 하위 97.5% (상위 2.5%)
```

## 99% 신뢰구간


```python
numpy.quantile(ms, .005)
```

```python
numpy.quantile(ms, .995)
```


# Cars.csv 데이터 실습


```python
import pandas as pd

cars = pd.read_csv('cars.csv')
cars.head()
```

```python
numpy.mean(cars['speed'])
```

```python
cars['speed'].mean()
```
# Bootstrapping


```python
from sklearn.utils import resample

resample([3, 4, 5])
```
```python
ms = []
for _ in range(10000):
    x = resample(cars['speed']) # 우리가 가진 샘플에서 다시 샘플링
    m = numpy.mean(x)
    ms.append(m)

seaborn.distplot(ms)
```

```python
numpy.quantile(ms, .025)
```

```python
numpy.quantile(ms, .975)
```

## 실습
cars의 dist 컬럼의 중간값(median)을 구하고 95% 신뢰구간 구하기.


```python
ms = []
for _ in range(10000) :
    x = resample(cars['dist'])
    m = numpy.median(x)
    ms.append(m)

seaborn.distplot(ms)
```

```python
numpy.quantile(ms, .975)
```

```python
numpy.quantile(ms, .025)
```
```python
numpy.median(ms)
```

```python
seaborn.regplot('speed', 'dist', cars)
```
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()  # 선형 모형
x = cars[['speed']]         # 독립변수
y = cars[['dist']]          # 종속변수
model.fit(x,y)              # 파라미터 추정
```

```python
model.coef_                 # 독립변수의 가중치(y = ax + b 중에서 a), 즉 coefficient(계수)
                            # 독립변수가 n개면 가중치도 n개가 나온다
```
```python
model.intercept_            # 절편(y = ax + b 중에서 b)
```

```python
resample(cars)              # resample할 때 전체를 같이 돌려야함. speed 따로, dist 따로 돌리면 짝이 바뀌어 의미가 없기 때문.
```
#### 참고


```python
x['speed'].shape
```

```python
x['speed'] # 시퀀스(데이터가 한 줄로 늘어서있는 1차원 자료형)
```

```python
x[['speed']].shape
```

```python
x[['speed']] # 표 형태
```

### 실습 : 부트스트래핑으로 선형 모형의 파라미터의 95% 신뢰구간 구하기


```python
coefs = []
intercepts = []

for _ in range(10000):
    df = resample(cars)                     # 샘플링
    x = df[['speed']]                       # 독립변수 데이터
    y = df['dist']                          # 종속변수 데이터
    
    model = LinearRegression()              # 모형 만들기
    model.fit(x,y)                          # 파라미터 추정
    
    coefs.append(model.coef_)               # 추정한 계수(파라미터)를 리스트에 추가
    intercepts.append(model.intercept_)     # 추정한 절편(파라미터)을 리스트에 추가
```


```python
numpy.quantile(coefs, [0.025, 0.975])
```

```python
numpy.quantile(intercepts, [0.025, 0.975])
```

신뢰구간이 3.1부터 4.9 사이에 위치한다고 나온다면 속도가 높을 수록 제동 거리가 길어진다는 양의 관계를 파악할 수 있지만, 신뢰구간이 -1.6부터 +3.4 사이에 위치한다고 나오면 두 변수가 양의 관계인지 음의 관계인지 해석하기 나름이 된다. 이런 상황에서는 한 방향으로 결론이 나올 수 있도록 데이터를 더 많이 모아야 한다. 데이터를 많이 모을수록 신뢰구간이 줄어들기 때문.

## 데이터를 많이 모으면 신뢰구간이 줄어든다


```python
sample_size = 50
ms = []

for _ in range(10000) :
    x = uniform(0, 100, sample_size)
    m = numpy.mean(x)
    ms.append(m)

numpy.quantile(ms, [.025, .975])     # sample_size의 값을 키울수록 범위가 좁아짐
```

### CASE 비교 : 작은 Sample로 resample을 많이 하는 경우 vs. 큰 Sample을 다루는 경우
* 작은 sample은 넓은 신뢰구간을 가지고 있음. 이 sample을 토대로 resample을 하는 것은 신뢰구간을 정확하게 알게 하는 것이지 신뢰구간을 좁힐 수는 없다. 즉, 부트스트래핑은 샘플이 작아서 발생하는 자체 문제를 해결할 수 없고 단지 추정치를 보다 정확하게 알려주는 역할을 한다.  
* 정확한 신뢰구간은 모분포에서 무한히 샘플링을 했을 때 존재한다. 우리가 분석할 때는 모분포를 알 수 없는 상황이기때문에 부트스트래핑을 통해 정확한 신뢰구간 추정치를 알아가야 함.
* Resampling을 많이 하면 신뢰구간 범위가 넓어지면서 정확해지는 경향이 있음.

#### 정리
1. 샘플의 크기가 커지면 -> 실제 신뢰구간이 좁아진다
2. Resample을 많이 하면 -> 추정된 신뢰구간이 정확해진다
3. Sample이 얼마면 충분한가? -> 충분히 좁은 신뢰구간
4. 필요한 최소한의 Sample -> -와 +가 신뢰구간 안에서 공존하지 않을 정도

# 과적합
* 현재 가진 표본에 지나치게 치우친 추정을 하는 것.
* 우리가 가진 Sample에 비해서 너무 많은 파라미터를 가지고 있을 때.
* 파라미터는 적으면 적을수록 좋다. 파라미터가 많을수록 더 많은 데이터가 필요하기 때문.
* 과적합이 있는지 판단하기 위해서 교차 검증(Cross validation) 활용. 

## 교차 검증
* 현재 데이터를 training set와 test set으로 무작위 분할
* training set로 추정하고 test set으로 테스트를 해본다.
* 이 때, training set보다 test set의 예측이 떨어진다면 과적합이 있다는 것.

## 과적합을 방지하기 위한 방법
* 선형 모형을 쓴다
* 파라미터의 갯수를 줄인다
* 과도한 추정치에 페널티를 부과한다
* 추정을 조기에 중단한다

# Sleep.csv 파일을 이용한 실습
sleep 데이터에서 group으로 extra를 예측하는 선형 모형을 만들고 각 파라미터의 신뢰구간을 구해보자.


```python
sleep = pd.read_csv('sleep.csv')
sleep.head()
```
```python
coefs = []
intercepts = []

for _ in range(100):
    df = resample(sleep)
    x = df[['group']]
    y = df['extra']
    model = LinearRegression()
    model.fit(x,y)
    coefs.append(model.coef_)
    intercepts.append(model.intercept_)

numpy.quantile(coefs, [0.025, 0.975])
```
```python
numpy.quantile(intercepts, [0.025, 0.975])
```