
# Correlation (상관)
* 산점도 : 두 연속 변수의 관계를 점(한 건의 데이터)으로 시각화
* 공분산
    * 두 연속 변수의 관계를 시각화 (얼마나 퍼져있는지)
    * 두 변수가 같은 방향이면 +, 반대 방향이면 -
    * 함께 변하는 경향이 강할수록 절대값이 커진다(+, - 관계 없이)
* 상관 계수
    * 공분산을 두 변수의 표준편차로 나누어 공분산을 일정 범위(-1부터 +1까지)로 만든 것
    * +1 : 두 변수가 완벽하게 같은 방향으로 움직임
    * 0 : 두 변수 사이에 아무 관계가 없음 (산점도가 직선으로 나타나지 않을 때 ex. W, U, O 등의 형태)
    * -1 : 두 변수가 완벽하게 다른 방향으로 움직임
* 허위 상관관계
    * 두 변수 사이에 실제로는 관계가 없어도 상관관계가 나타나는 경우
    * 데이터가 적을 수록 나타나기 쉽다
    * 해결 : 상관계수의 신뢰구간 확인
    
* p-value는 작은 게 좋다

## 실습 (Cars.csv)


```python
import pandas as pd
from scipy.stats import pearsonr   # 피어슨 상관계수 (pearsonr, p값)

cars = pd.read_csv('cars.csv')

pearsonr(cars['speed'], cars['dist'])
```

```python
# 부트스트래핑

from sklearn.utils import resample

df = resample(cars)
pearsonr(df['speed'], df['dist'])
```

```python
# 부트스트래핑 활용

import numpy

cors = []
for _ in range(10000) :
    df = resample(cars)
    res = pearsonr(df['speed'], df['dist'])    # 상관계수 구하기
    cors.append(res[0])                        # 상관계수를 리스트에 추가, [0]은 상관계수, [1]은 p값
    
numpy.quantile(cors, [.025, .975])             # 상관계수의 95% 신뢰구간

# numpy.quantile(cors, [.005, .995])           # 상관계수의 99% 신뢰구간 (신뢰구간을 넓히면 구간이 넓어지고 유의수준(p값)은 낮아짐)
```

(상관계수, p값)일 때  
p값 < .05 : 95% 신뢰구간에서 반대 부호가 포함되지 않음  
p값 < .01 : 99% 신뢰구간에서 반대 부호가 포함되지 않음  
p값 < .001 : 99.9% 신뢰구간에서 반대 부호가 포함되지 않음  

## 상관계수의 종류

### 피어슨 상관계수
* 일반적으로 '상관계수'라는 것은 피어슨 상관계수를 지칭한다

### 스피어만 상관계수
* 실제 변수값 대신 서열을 사용하여 피어슨 상관계수를 계산


## 실습 (liar.csv)
창의성과 거짓말 등수는 역상관 관계 -> 창의성이 높읖수록 거짓말을 잘한다


```python
from scipy.stats import spearmanr, kendalltau

liar = pd.read_csv('liar.csv')
liar.head()

spearmanr(liar['Creativity'], liar['Position'])
```

```python
kendalltau(liar['Creativity'], liar['Position'])
```

```python
pearsonr(liar['Creativity'], liar['Position'])
```

# 필기 빈 부분 나중에 채워넣기

## 실습 (child.csv)


```python
from statsmodels.formula.api import ols

child = pd.read_csv('child.csv')
child.head()
```

```python
res = ols('Aggression ~ Computer_Games', child).fit()
res.summary()
```

### 결과 분석

R-Squared : 3.5%만 설명 가능  
Prob(F-statistic) : 0.05보다 작기 때문에 pass  
Computer_Games의 95% 신뢰구간 : +부호만 있기 때문에 pass  
Computer_Games의 p-value : 0.05보다 작기 때문에 pass


```python
res = ols('Aggression ~ Television', child).fit()
res.summary()
```

### 결과 분석

R-Squared : 2.5%만 설명 가능  
Prob(F-statistic) : 0.05보다 작기 때문에 pass  
Television의 95% 신뢰구간 : +부호만 있기 때문에 pass  
Television의 p-value : 0.05보다 작기 때문에 pass

-----

Computer_Games의 AIC, BIC는 Television의 AIC, BIC보다 낮고  
Adj-Squared는 높기 때문에  
Computer_Games와 Aggression 간의 상관관계가 더 높다고 볼 수 있음.


```python
res = ols('Aggression ~ Diet', child).fit()
res.summary()
```
```python
# 독립변수가 2개인 경우

res = ols('Aggression ~ Television + Computer_Games', child).fit()
res.summary()
```


```python
res = ols('Aggression ~ Computer_Games + Diet + Television', child).fit()
res.summary()
```

```python
res = ols('Aggression ~ Computer_Games + Parenting_Style + Diet', child).fit()
res.summary()
```
```python
res = ols('Aggression ~ Computer_Games + Television + Sibling_Aggression', child).fit()
res.summary()
```


* TV와 게임을 "통계적으로 통제"했을 때 형제의 공격성은 아동의 공격성을 설명하지 못한다? <- 복습하면서 다시 공부하기

## 변수 선택

### 단계적 회귀 분석
* 변수를 단계적으로 추가/삭제하는 방법
* 데이터가 적고 변수의 수도 적을 때 쓸 수 있음
* 과거에는 많이 사용했으나 한계가 있음

### 전방 선택
* 절편만 있는 모형으로 시작
* 추가했을 때 모형을 가장 많이 개선할 수 있는 변수 추가
* 더이상 모형이 개선되지 않으면 중단
* 한계 : 지나치게 적은 변수를 포함시킬 위험 존재

### 후방 선택
* 모든 변수를 투입한 모형으로 시작
* 제외했을 때 모형을 가장 많이 개선하는 변수 제외
* 더이상 모형이 개선되지 않으면 중단
* 한계 : 지나치게 많은 변수를 포함시킬 위험 존재