

```python
import numpy
```


```python
x = [100, 100, 200, 100, 300, 200, 300, 500, 700, 800, 900, 500]
```


```python
numpy.mean(x) # 평균
numpy.median(x) # 중간값, 크기별로 자동 정렬한 후 중간에 위치한 값
```
```python
import scipy
import scipy.stats
from scipy.stats import mode
```


```python
mode(x) 
# 결과 : ModeResult(mode=array([제일 흔한 값]), count=array([나온 횟수]))
numpy.min(x) # 최소값
numpy.max(x) # 최대값
numpy.var(x) # 분산
numpy.std(x) # 표준편차
numpy.quantile(x,.5)
```

```python
from numpy.random import binomial
```


```python
x = binomial(100, .3, 30)
min(x)
max(x)
```
```python
for i in x:
    print(i)
```

# 레스토랑 예시


```python
수익 = 0
준비 = 20
for 고객수 in x:
    if 고객수 <= 준비:
        수익 += (고객수 - 준비) * 10000
        수익 += 고객수 * 1000
    else:
        수익 = 수익 + 준비 * 1000
수익
```

# 보험료 예시


```python
보험료 = 300
지급액 = 1000
고객 = 지급액 / 10
장사 = 100
사망자 = binomial(고객, .1, 장사)
매출 = 보험료 * 고객
지급 = 지급액 * 사망자
```


```python
numpy.mean(매출-지급)
binomial(1, .5)
```

```python
이탈성공 = 0
기간 = 0
while 이탈성공 == 0:
    이탈성공 = binomial(1, .05)
    기간 = 기간 + 1
기간

```python
from numpy.random import geometric
```


```python
기간 = geometric(.05, 100) #이탈률 5%일 때 100명의 고객이 이탈하기까지 걸리는 기간
numpy.sum(기간 * 0.5 - 10)
```
# 리트리버 케이스

```python
짜증 = 0
기간 = 0
while 짜증 < 3:                    # 세번째는 못 참는다
    사건 = binomial(1, .05)        # 5%의 확률로 사건 발생
    if 사건 == 1:                  # 사건 발생
        짜증 += 1                  # 짜증 증가
    기간 += 1
기간
```

```python
from numpy.random import negative_binomial
```


```python
negative_binomial(3, .05) # 5% 확률로 짜증이 나고 짜증이 3번 나면 폭발하는데, 그 때까지의 기간
```

```python
폭발 = []
for i in range(100):
    짜증 = 0
    기간 = 0
    while 짜증 < 3:
        사건 = binomial(1, .05)
        if 사건 == 1:
            짜증 += 1
        기간 += 1
    폭발.append(기간)
```


# 포아송


```python
from numpy.random import poisson
```


```python
binomial(100, .3, 50)
```

# 균등 분포


```python
from numpy.random import uniform
```


```python
int(uniform(10, 20)) # 최소값 10에서 최대값 20 사이에서 균등한 확률로 랜덤하게 숫자 출력
```

```python
uniform(0, 1)
```

```python
x = uniform(-1, 1, 1000)
y = uniform(-1, 1, 1000)
numpy.abs(x + y) < 1
numpy.mean(numpy.abs(x) + numpy.abs(y) < 1)
```


# 정규 분포

### CASE 1


```python
from numpy.random import normal
```


```python
normal(10, 5)
```

```python
#고객 100명이 평균 50만원(표준편차 20만원)을 쓸 때 각 고객이 쓰는 돈
normal(50, 20, 100) 
```


```python
하루매출 = normal(50, 20, 100)
기간 = geometric(.05, 100)

하루매출 * 기간
```

### CASE 2


```python
유저집단 = binomial(1, .5, 100)

총매출 = []
for 유저 in 유저집단:
    if 유저 == 0: # 라이트 유저
        매출 = normal(30, 10)
    else: # 헤비 유저
        매출 = normal(100, 10)
    총매출.append(매출)
```
# 다항 분포, multinomial


```python
from numpy.random import multinomial
```


```python
# 여러가지 경우의 수가 각각 50%, 20%, 20%, 10%의 확률을 가지고 있을 때 한 번 실행 시 어떤 결과가 나오는가
multinomial(1, [.5, .2, .2, .1])
```

```python
# 여러가지 경우의 수가 각각 50%, 20%, 20%, 10%의 확률을 가지고 있을 때 10번 실행 시 어떤 결과가 나오는가
multinomial(10, [.5, .2, .2, .1])
```

```python
multinomial(10, [.3, .7])
# binomial(10, .3)과 같다
```

### CASE 1
한국인 50%, 중국인 30%, 일본인 20%일 때, 100명의 손님이 가게를 방문했다면 각 나라별 손님의 수는?


```python
multinomial(100, [.5, .3, .2])
```

```python
# 30회 반복
multinomial(100, [.5, .3, .2], 30)
```
# 지수 분포, exponential


```python
from numpy.random import exponential
```


```python
exponential(30)
```

```python
알바 = 10
리드타임 = 10 / 알바
numpy.mean(exponential(30, 1000) < 리드타임) # 리드타임 이내 도착
```
