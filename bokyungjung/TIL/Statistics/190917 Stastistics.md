
# 복습
### 실습 (child.csv)


```python
import pandas as pd
from statsmodels.formula.api import ols

child = pd.read_csv('child.csv')
ols('Aggression ~ Television', child).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Aggression</td>    <th>  R-squared:         </th> <td>   0.025</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.024</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   17.11</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>3.98e-05</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:51:34</td>     <th>  Log-Likelihood:    </th> <td> -175.93</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   666</td>      <th>  AIC:               </th> <td>   355.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   664</td>      <th>  BIC:               </th> <td>   364.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>   -0.0005</td> <td>    0.012</td> <td>   -0.041</td> <td> 0.967</td> <td>   -0.025</td> <td>    0.024</td>
</tr>
<tr>
  <th>Television</th> <td>    0.1634</td> <td>    0.040</td> <td>    4.137</td> <td> 0.000</td> <td>    0.086</td> <td>    0.241</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>24.471</td> <th>  Durbin-Watson:     </th> <td>   1.931</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  58.038</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.108</td> <th>  Prob(JB):          </th> <td>2.50e-13</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.430</td> <th>  Cond. No.          </th> <td>    3.23</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



# 회귀분석
## 잔차
* 회귀 분석의 예측과 실제값의 차이

### 왜도 (Skewness)
* 분포의 비대칭성
* Negative Skew : 오른쪽으로 축이 치우친 경우, Skew가 음의 값을 가짐
* Positive Skew : 왼쪽으로 축이 치우친 경우, Skew가 양의 값을 가짐 ex. 소득
* 잔차가 Skewed 되었다는 것
    * 예측이 한 방향으로 치우쳤다는 의미
    * 데이터 자체가 치우쳤을 가능성 있음
    
### 첨도 (Kurtosis)
* 분포가 한 점에 몰린 정도
* 정규분포의 첨도 = 3
* 첨도가 높다 : 데이터가 중심에 몰려 있음
* 첨도가 낮다 : 데이터가 퍼져 있음

### 잔차의 정규성
* Omnibus, Jarque-Bera
* 잔차가 정규분포에 가까운 성질을 가지고 있는가? 를 의미
* 둘 다 Prob가 1에 가까울수록 정규분포에 가까움

### 등분산성 (Homoscedasticity)
* 모든 범위에서 잔차의 분산이 같음, 즉 어떤 x에서든 비슷한 정도로 y를 맞출 수 있음
* Dubin-Watson 통계량이 1~2 정도
* 나름 중요한 수치

### 조건수 (Condition Number)
* 입력의 변화에 따른 출력의 변화를 나타내는 숫자
* 조건수가 크면 데이터가 조금만 달라져도 결과에 큰 차이가 있음
* 보통 30 이하
#### 다중공선성 (Multicollinearity)
    * 독립변수들이 서로 예측가능할 경우
    * 계수를 뭘 넣어도 성립하게 됨 -> 계수 추정이 불안정하다
    * <b>조건수</b>가 커진다
    * 따라서 데이터나 변수의 변화에 따라 추정된 계수가 크게 달라짐
    * ex. TV를 몇 시간 보는가 + TV를 몇 분 보는가 <- 독립변수라면 사실상 같은 의미를 가지고 있는 변수


```python
# 다중공선성 예시

child['TV2'] = child['Television']
ols('Aggression ~ Television+TV2', child).fit().summary()

# Cond. No.가 2.35e+16인 것 확인하기
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Aggression</td>    <th>  R-squared:         </th> <td>   0.025</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.024</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   17.11</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>3.98e-05</td>
</tr>
<tr>
  <th>Time:</th>                 <td>10:46:41</td>     <th>  Log-Likelihood:    </th> <td> -175.93</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   666</td>      <th>  AIC:               </th> <td>   355.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   664</td>      <th>  BIC:               </th> <td>   364.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>   -0.0005</td> <td>    0.012</td> <td>   -0.041</td> <td> 0.967</td> <td>   -0.025</td> <td>    0.024</td>
</tr>
<tr>
  <th>Television</th> <td>    0.0817</td> <td>    0.020</td> <td>    4.137</td> <td> 0.000</td> <td>    0.043</td> <td>    0.120</td>
</tr>
<tr>
  <th>TV2</th>        <td>    0.0817</td> <td>    0.020</td> <td>    4.137</td> <td> 0.000</td> <td>    0.043</td> <td>    0.120</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>24.471</td> <th>  Durbin-Watson:     </th> <td>   1.931</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  58.038</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.108</td> <th>  Prob(JB):          </th> <td>2.50e-13</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.430</td> <th>  Cond. No.          </th> <td>2.35e+16</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 1.21e-30. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



## 변수 선택

### 교차 검증 (Cross Validation, CV)
* 데이터를 무작위로 두 세트로 나눈 다음 한 세트에서는 추정, 다른 세트에서는 검증

### k-fold Cross Validation
* 데이터를 k개로 나누어 CV를 k번 하는 방법
* ex. k = 3
    1. 1, 2번 데이터로 추정 -> 3번 데이터로 검증
    2. 2, 3번 데이터로 추정 -> 1번 데이터로 검증
    3. 1, 3번 데이터로 추정 -> 2번 데이터로 검증

## 정규화 (Regularization)
* 가능한 계수를 작게 추정하여 <b>과적합</b>을 피하는 것
* 계수가 0이라면 변수를 추가하지 않은 것과 같기 때문에 사실상 변수를 제외하는 효과
* 회귀분석에서의 정규화
    * ㅇ

### 과적합 (overfitting)
    * 모형의 계수가 주어진 데이터(Sample)에 지나치게 의존하여 추정되는 경우.
    * 다른 Sample에서는 잘 적용되지 않을 수 있어 좋은 예측이라고 할 수 없다.
    * 변수가 많을 수록 / 계수가 클수록 과적합 위험성이 높다
    
## 라쏘 회귀분석
* 계수의 절대값을 최소화하는 방법
* 람다 : 클수록 계수를 최소화하는 데 더 큰 비중을 차지한다
* MSE가 더 크더라도 람다가 반영된 값과 더한 합계가 작다면 그것을 선택하게 됨

## 릿지 회귀분석
* 계수의 제곱을 최소화하는 방법
* 라쏘보다 예측성능이 더 좋음
* 회귀계수가 0이 되지 않는다

## 엘라스틱 넷
* 라쏘 + 릿지
* a = 1 : 라쏘
* a = 0 : 릿지

## 하이퍼파라미터
* 모형의 특성을 결정하지만 데이터로부터 학습되지 않는 값
* 엘라스틱 넷에서 람다, a
* CV를 통해 결정


### 실습 (child.csv)


```python
# 정규화

from sklearn.linear_model import LinearRegression, ElasticNetCV

x = child.iloc[:, 1:6]   # 1~6번째 열까지가 x
y = child['Aggression']  # Aggression열이 y

# 선형 모형

m1 = LinearRegression()
m1.fit(x, y)
m1.coef_
```




    array([ 0.03291628,  0.14216117,  0.08168417, -0.10905412,  0.0566481 ])




```python
# 엘라스틱 넷 + 교차검증
# l1_ratio : 수식에서 a, 라쏘(L1), 릿지(L2)

m2 = ElasticNetCV(l1_ratio=[.1, .5, .1], cv = 3)  # 3-fold 교차검증
m2.fit(x, y)
m2.coef_
```




    array([ 0.02572348,  0.12471677,  0.0687051 , -0.08714782,  0.05652955])




```python
m2.l1_ratio_    # 계산이 끝난 후의 값 뒤에는 _를 붙여야 한다
```




    0.1




```python
m2.alpha_      # 람다에 해당하는 값
```




    0.008882602491363027



## 더미 코딩
* 독립변수에 이산형(범주형) 변수가 있을 경우(예: 짜장, 짬뽕, 볶음밥)
* 짜장을 기준으로 삼는다면 나머지 값을 변수로 추가해서 해당하면 1, 아니면 0으로 값을 설정한다.
* 이 때 점심_짜장을 변수로 추가하지 않는 이유는 다중공선성 때문!

### 실습 (hsb2.csv)
- SES (Socio-Economic Status) : 사회경제적 수준


```python
hsb = pd.read_csv('hsb2.csv')

# 성별은 이미 female을 변수로 더미 코딩이 되어 있지만 race는 아직 더미 코딩이 되어 있지 않음
hsb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>female</th>
      <th>race</th>
      <th>ses</th>
      <th>schtyp</th>
      <th>prog</th>
      <th>read</th>
      <th>write</th>
      <th>math</th>
      <th>science</th>
      <th>socst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>57</td>
      <td>52</td>
      <td>41</td>
      <td>47</td>
      <td>57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>121</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>68</td>
      <td>59</td>
      <td>53</td>
      <td>63</td>
      <td>61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>86</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>44</td>
      <td>33</td>
      <td>54</td>
      <td>58</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>141</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>63</td>
      <td>44</td>
      <td>47</td>
      <td>53</td>
      <td>56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>172</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>47</td>
      <td>52</td>
      <td>57</td>
      <td>53</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>




```python
ols('read ~ C(race)', hsb).fit().summary()    # C() : 범주형 변수를 표시하여 더미 코딩
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>read</td>       <th>  R-squared:         </th> <td>   0.084</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.070</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   5.964</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>0.000654</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:00:12</td>     <th>  Log-Likelihood:    </th> <td> -740.06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   1488.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   196</td>      <th>  BIC:               </th> <td>   1501.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>   46.6667</td> <td>    2.019</td> <td>   23.117</td> <td> 0.000</td> <td>   42.685</td> <td>   50.648</td>
</tr>
<tr>
  <th>C(race)[T.2]</th> <td>    5.2424</td> <td>    3.601</td> <td>    1.456</td> <td> 0.147</td> <td>   -1.859</td> <td>   12.344</td>
</tr>
<tr>
  <th>C(race)[T.3]</th> <td>    0.1333</td> <td>    2.994</td> <td>    0.045</td> <td> 0.965</td> <td>   -5.772</td> <td>    6.038</td>
</tr>
<tr>
  <th>C(race)[T.4]</th> <td>    7.2575</td> <td>    2.179</td> <td>    3.330</td> <td> 0.001</td> <td>    2.959</td> <td>   11.556</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.544</td> <th>  Durbin-Watson:     </th> <td>   1.966</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.170</td> <th>  Jarque-Bera (JB):  </th> <td>   2.702</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.147</td> <th>  Prob(JB):          </th> <td>   0.259</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.512</td> <th>  Cond. No.          </th> <td>    8.25</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### 결과 분석
* 인종이 읽기 성적의 8.4퍼센트를 설명한다. (R-Squared)
* 46.6 + 5.24 * 인종2 + 0.13 * 인종3 + 7.25 * 인종4 (coef)
* 따라서 인종1은 46.6, 인종2는 46.6 + 5.24, 인종3은 46.6 + 0.13, 인종4는 46.6+7.25
* 인종2과 인종3은 신뢰구간 내에서 부호가 바뀌므로 인종1과 차이가 난다고 단정할 수 없다.
* 인종4는 신뢰구간 내에서 같은 부호가 유지되므로 결과적으로 인종1보다 점수가 높다고 볼 수 있다.
* 이 방법은 인종1과 나머지 인종2~4만 비교, 인종2와 인종4 비교 등은 할 수 없다는 게 한계.(이를 해결하려면 데이터 내에서 순서를 바꿔주거나 더미 코딩을 수동으로 해야한다.->pandas.get_dummies)

### pandas.get_dummies


```python
race = pd.get_dummies(hsb['race'], prefix='race') # prefix : 변수명 이름 붙이기 (a_b에서 a에 해당, b는 기존 컬럼의 값 ex. 짜장, 백인 등)
new = pd.concat([hsb, race], axis=1)              # hsb와 race(dummy)를 이어붙임(concat)

ols('read ~ race_2 + race_3 + race_4', new).fit().summary()    # 제외하는 열이 기준이 됨(여기서는 race_1)
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>read</td>       <th>  R-squared:         </th> <td>   0.084</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.070</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   5.964</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>0.000654</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:03:41</td>     <th>  Log-Likelihood:    </th> <td> -740.06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   1488.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   196</td>      <th>  BIC:               </th> <td>   1501.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   46.6667</td> <td>    2.019</td> <td>   23.117</td> <td> 0.000</td> <td>   42.685</td> <td>   50.648</td>
</tr>
<tr>
  <th>race_2</th>    <td>    5.2424</td> <td>    3.601</td> <td>    1.456</td> <td> 0.147</td> <td>   -1.859</td> <td>   12.344</td>
</tr>
<tr>
  <th>race_3</th>    <td>    0.1333</td> <td>    2.994</td> <td>    0.045</td> <td> 0.965</td> <td>   -5.772</td> <td>    6.038</td>
</tr>
<tr>
  <th>race_4</th>    <td>    7.2575</td> <td>    2.179</td> <td>    3.330</td> <td> 0.001</td> <td>    2.959</td> <td>   11.556</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.544</td> <th>  Durbin-Watson:     </th> <td>   1.966</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.170</td> <th>  Jarque-Bera (JB):  </th> <td>   2.702</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.147</td> <th>  Prob(JB):          </th> <td>   0.259</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.512</td> <th>  Cond. No.          </th> <td>    8.25</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### .replace로 컬럼값 바꾸기


```python
hsb['race'].replace({1: '흑', 2: '백', 3: '황', 4: '기타'})   # 기준은 가나다 순이므로 이 때는'기타'가 기준이 된다.
```




    0      기타
    1      기타
    2      기타
    3      기타
    4      기타
    5      기타
    6       황
    7       흑
    8      기타
    9       황
    10     기타
    11     기타
    12     기타
    13     기타
    14      황
    15     기타
    16     기타
    17     기타
    18     기타
    19     기타
    20     기타
    21     기타
    22      황
    23      흑
    24      흑
    25      황
    26     기타
    27     기타
    28     기타
    29      백
           ..
    170    기타
    171     흑
    172     황
    173     백
    174     황
    175    기타
    176    기타
    177    기타
    178    기타
    179    기타
    180    기타
    181    기타
    182    기타
    183    기타
    184     백
    185     백
    186    기타
    187     백
    188    기타
    189     황
    190    기타
    191    기타
    192    기타
    193     백
    194    기타
    195     백
    196    기타
    197    기타
    198    기타
    199    기타
    Name: race, Length: 200, dtype: object




```python
ols('read ~ C(race)', hsb).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>read</td>       <th>  R-squared:         </th> <td>   0.084</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.070</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   5.964</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>0.000654</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:08:48</td>     <th>  Log-Likelihood:    </th> <td> -740.06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   1488.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   196</td>      <th>  BIC:               </th> <td>   1501.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>   46.6667</td> <td>    2.019</td> <td>   23.117</td> <td> 0.000</td> <td>   42.685</td> <td>   50.648</td>
</tr>
<tr>
  <th>C(race)[T.2]</th> <td>    5.2424</td> <td>    3.601</td> <td>    1.456</td> <td> 0.147</td> <td>   -1.859</td> <td>   12.344</td>
</tr>
<tr>
  <th>C(race)[T.3]</th> <td>    0.1333</td> <td>    2.994</td> <td>    0.045</td> <td> 0.965</td> <td>   -5.772</td> <td>    6.038</td>
</tr>
<tr>
  <th>C(race)[T.4]</th> <td>    7.2575</td> <td>    2.179</td> <td>    3.330</td> <td> 0.001</td> <td>    2.959</td> <td>   11.556</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 3.544</td> <th>  Durbin-Watson:     </th> <td>   1.966</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.170</td> <th>  Jarque-Bera (JB):  </th> <td>   2.702</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.147</td> <th>  Prob(JB):          </th> <td>   0.259</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.512</td> <th>  Cond. No.          </th> <td>    8.25</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## 상호작용
* 두 독립변수의 곱으로 이루어진 항
* 두 독립변수의 상호작용에 대한 내용 
* y = x + m + xm
* 간단히 하기 위해 m은 0 또는 1로 이루어진 범주형 변수라고 가정(범주형 변수가 아니어도 됨)
* 상호작용이 있는 경우
    * y = x + xm : m에 따라 x의 기울기가 바뀌는 것으로 해석
    * y = x + m + xm : m에 따라 x의 절편과 기울기가 바뀌는 것으로 해석


### 실습 (hsb2.csv)


```python
ols('read ~ ses', hsb).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>read</td>       <th>  R-squared:         </th> <td>   0.086</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.081</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   18.64</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>2.49e-05</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:47:00</td>     <th>  Log-Likelihood:    </th> <td> -739.80</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   1484.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   198</td>      <th>  BIC:               </th> <td>   1490.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   43.6972</td> <td>    2.095</td> <td>   20.858</td> <td> 0.000</td> <td>   39.566</td> <td>   47.829</td>
</tr>
<tr>
  <th>ses</th>       <td>    4.1522</td> <td>    0.962</td> <td>    4.317</td> <td> 0.000</td> <td>    2.256</td> <td>    6.049</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 6.283</td> <th>  Durbin-Watson:     </th> <td>   1.816</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.043</td> <th>  Jarque-Bera (JB):  </th> <td>   3.593</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.104</td> <th>  Prob(JB):          </th> <td>   0.166</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.377</td> <th>  Cond. No.          </th> <td>    7.82</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
hsb['schtyp2'] = hsb['schtyp'] - 1

# 일반적인 회귀분석
ols('read ~ ses + schtyp2', hsb).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>read</td>       <th>  R-squared:         </th> <td>   0.088</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.079</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   9.529</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>0.000112</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:48:41</td>     <th>  Log-Likelihood:    </th> <td> -739.57</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   1485.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   197</td>      <th>  BIC:               </th> <td>   1495.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   43.6743</td> <td>    2.098</td> <td>   20.816</td> <td> 0.000</td> <td>   39.537</td> <td>   47.812</td>
</tr>
<tr>
  <th>ses</th>       <td>    4.0613</td> <td>    0.972</td> <td>    4.178</td> <td> 0.000</td> <td>    2.144</td> <td>    5.979</td>
</tr>
<tr>
  <th>schtyp2</th>   <td>    1.3109</td> <td>    1.916</td> <td>    0.684</td> <td> 0.495</td> <td>   -2.467</td> <td>    5.089</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 6.257</td> <th>  Durbin-Watson:     </th> <td>   1.804</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.044</td> <th>  Jarque-Bera (JB):  </th> <td>   3.633</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.115</td> <th>  Prob(JB):          </th> <td>   0.163</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.381</td> <th>  Cond. No.          </th> <td>    7.85</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# 상호작용
ols('read ~ ses + ses:schtyp2', hsb).fit().summary() # ses:schtyp2 : 상호작용되는 부분
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>read</td>       <th>  R-squared:         </th> <td>   0.087</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.078</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   9.376</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>0.000129</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:50:20</td>     <th>  Log-Likelihood:    </th> <td> -739.71</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   1485.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   197</td>      <th>  BIC:               </th> <td>   1495.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>   43.7565</td> <td>    2.104</td> <td>   20.799</td> <td> 0.000</td> <td>   39.608</td> <td>   47.905</td>
</tr>
<tr>
  <th>ses</th>         <td>    4.0600</td> <td>    0.987</td> <td>    4.114</td> <td> 0.000</td> <td>    2.114</td> <td>    6.006</td>
</tr>
<tr>
  <th>ses:schtyp2</th> <td>    0.3568</td> <td>    0.822</td> <td>    0.434</td> <td> 0.665</td> <td>   -1.265</td> <td>    1.979</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 6.259</td> <th>  Durbin-Watson:     </th> <td>   1.810</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.044</td> <th>  Jarque-Bera (JB):  </th> <td>   3.603</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.108</td> <th>  Prob(JB):          </th> <td>   0.165</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.379</td> <th>  Cond. No.          </th> <td>    7.99</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### 결과 해석
* schtyp = 0이 공립, 1이 사립이라고 가정
* read = (4.06 + 0.35*학교) * ses
* ses:schtyp2의 신뢰구간 내에서 부호가 바뀌므로 공/사립 학교가 ses에서 성적에 미치는 영향을 조절한다고 볼 수 없다.
    * 둘 다 양의 부호였다면 사립일수록 ses가 성적에 큰 영향을 미친다는 해석이 가능



```python
ols('write ~ ses + ses:female', hsb).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>write</td>      <th>  R-squared:         </th> <td>   0.116</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.107</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   12.90</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>5.42e-06</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:10:18</td>     <th>  Log-Likelihood:    </th> <td> -720.78</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   1448.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   197</td>      <th>  BIC:               </th> <td>   1457.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>   46.6663</td> <td>    1.914</td> <td>   24.378</td> <td> 0.000</td> <td>   42.891</td> <td>   50.441</td>
</tr>
<tr>
  <th>ses</th>        <td>    1.7444</td> <td>    0.909</td> <td>    1.918</td> <td> 0.057</td> <td>   -0.049</td> <td>    3.538</td>
</tr>
<tr>
  <th>ses:female</th> <td>    2.3479</td> <td>    0.583</td> <td>    4.027</td> <td> 0.000</td> <td>    1.198</td> <td>    3.498</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11.857</td> <th>  Durbin-Watson:     </th> <td>   1.792</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.003</td> <th>  Jarque-Bera (JB):  </th> <td>   7.477</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.324</td> <th>  Prob(JB):          </th> <td>  0.0238</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.309</td> <th>  Cond. No.          </th> <td>    8.80</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### 결과 해석
* write = (1.7 + 2.3 * female) * ses
* ses에 곱해지는 가중치가 남자/여자일 때 각각 다른 것
* 기울기가 달라짐


```python
ols('write ~ ses + ses:female + female', hsb).fit().summary()
# ols('write ~ ses * female', hsb).fit().summary()도 같은 의미
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>write</td>      <th>  R-squared:         </th> <td>   0.124</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.111</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   9.256</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>9.38e-06</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:34:49</td>     <th>  Log-Likelihood:    </th> <td> -719.84</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   1448.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   196</td>      <th>  BIC:               </th> <td>   1461.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>   43.3109</td> <td>    3.120</td> <td>   13.881</td> <td> 0.000</td> <td>   37.157</td> <td>   49.464</td>
</tr>
<tr>
  <th>ses</th>        <td>    3.1618</td> <td>    1.382</td> <td>    2.288</td> <td> 0.023</td> <td>    0.437</td> <td>    5.887</td>
</tr>
<tr>
  <th>ses:female</th> <td>    0.0388</td> <td>    1.795</td> <td>    0.022</td> <td> 0.983</td> <td>   -3.501</td> <td>    3.578</td>
</tr>
<tr>
  <th>female</th>     <td>    5.3668</td> <td>    3.946</td> <td>    1.360</td> <td> 0.175</td> <td>   -2.415</td> <td>   13.149</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11.529</td> <th>  Durbin-Watson:     </th> <td>   1.789</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.003</td> <th>  Jarque-Bera (JB):  </th> <td>   7.780</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.349</td> <th>  Prob(JB):          </th> <td>  0.0205</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.332</td> <th>  Cond. No.          </th> <td>    22.5</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
ols('write ~ read + female:read', hsb).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>write</td>      <th>  R-squared:         </th> <td>   0.429</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.423</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   73.90</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 17 Sep 2019</td> <th>  Prob (F-statistic):</th> <td>1.13e-24</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:24:49</td>     <th>  Log-Likelihood:    </th> <td> -677.12</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   200</td>      <th>  AIC:               </th> <td>   1360.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   197</td>      <th>  BIC:               </th> <td>   1370.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>   23.2004</td> <td>    2.654</td> <td>    8.741</td> <td> 0.000</td> <td>   17.966</td> <td>   28.434</td>
</tr>
<tr>
  <th>read</th>        <td>    0.5144</td> <td>    0.050</td> <td>   10.218</td> <td> 0.000</td> <td>    0.415</td> <td>    0.614</td>
</tr>
<tr>
  <th>female:read</th> <td>    0.0961</td> <td>    0.019</td> <td>    5.000</td> <td> 0.000</td> <td>    0.058</td> <td>    0.134</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 7.827</td> <th>  Durbin-Watson:     </th> <td>   1.974</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.020</td> <th>  Jarque-Bera (JB):  </th> <td>   4.749</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.201</td> <th>  Prob(JB):          </th> <td>  0.0931</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.361</td> <th>  Cond. No.          </th> <td>    322.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### 결과분석
* 읽기를 잘 하면 글쓰기 성적이 올라간다.
* 남자의 글쓰기 성적이 0.5144점씩 올라갈 때 여자는 0.5144+0.0961점씩 올라간다.

#### 다시 체크해야할 부분들
1. 통계적 통제
2. 상호작용과 여러개의 독립변수의 차이점


# 복습 테스트
1. 선형 모형 : y가 연속인 것
2. 로지스틱 선형 모형 : y가 ~여부 등으로 설명되는 것
3. 두 집단의 평균 차이를 검정하기 위해 부트스트래핑 할 때는 두 집단이 원래 같다고 가정한다. 집단에서 어느 정도의 차이까지 날 수 있을까 구해보는 것
4. 신뢰구간이 헷갈리면 오차 범위라고 생각하기. -> 신뢰구간을 벗어난다 = 오차 범위를 벗어난다 = 두 집단이 다르다
