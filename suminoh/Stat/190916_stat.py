#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[2]:


cars = pandas.read_csv('cars.csv')


# In[3]:


cars.head()


# In[7]:


from scipy.stats import pearsonr


# (피어슨 상관계수, p값)  
# p값 < .05 : 95% 신뢰구간 반대 부호가 포함 X  
# p값 > .01 : 99% 신뢰구간 반대 부호가 포함 X  
# p값 < .001 : 99.9% 신뢰구간 반대 부호가 포함 X

# In[8]:


pearsonr(cars['speed'], cars['dist'])


# In[12]:


from sklearn.utils import resample


# In[24]:


cors = [] # 빈 리스트를 만든다.
for _ in range(10000): # 1만번 반복
    df = resample(cars) # 리샘플링
    res = pearsonr(df['speed'], df['dist']) # 상관계수를 구한다.
    cors.append(res[0]) # 상관계수를 리스트에 추가, [0]은 상관계수, [1]은 p값


# In[25]:


import numpy


# In[26]:


numpy.quantile(cors, [.025, .975]) # 상관계수의 95% 신뢰구간


# In[27]:


numpy.quantile(cors, [.005, .995]) # 상관계수의 99% 신뢰구간


# In[ ]:





# In[32]:


liar = pandas.read_csv('liar.csv')


# In[33]:


from scipy.stats import spearmanr, kendalltau


# In[34]:


spearmanr(liar['Creativity'], liar['Position'])


# In[35]:


kendalltau(liar['Creativity'], liar['Position'])


# In[36]:


pearsonr(liar['Creativity'], liar['Position']) 
# 다른 상관계수보다 P-value가 크다. 오차범위가 높음. 서열데이터에는 다른 상관계수가 더 적절.


# In[ ]:





# ## statsmodels를 이용한 선형 모형

# ### 설치

# In[44]:


#conda install statsmodels


# In[45]:


from statsmodels.formula.api import ols


# ### 자동차 데이터로 회귀분석

# In[48]:


res = ols('dist ~ speed', data = cars).fit() 
# R에서 쓰던 표기법, 종속변수 ~ 독립변수 / dist(y, 종속) ~ speed(x, 독립)


# In[49]:


res.summary()


# #### dist = -17.5791 + 3.9324speed
# #### <해석>
# * coef : 계수  
# * [0.025, 0.975] : 95% 신뢰구간 -> 부호가 바뀌지 않으니 믿어도 됨  
# * P>|t| : P-value  
# * std err : 신뢰구간을 계산하는 이론적 수치 / 직접 해석할 일 없음
# * t : P>|t|를 계산하는 이론적 수치 / 직접 해석할 일 없음  
# 
# 
# * R-squared : 에타제곱 (분산 %) / 에타제곱은 두 집단 간의 차이를 말할 때 사용, 회귀에서는 R제곱을 사용 / 모형적합도 지수  
# * Adj.R-squared : R제곱을 보정
# * R-squared는 독립변수와 종속변수의 상관계수를 제곱한 값과 같음
# * F-statistic : 절편을 제외한 모든 회귀계수가 0일 때를 가정하고 계산한 수치 / 독립변수를 다 0이여도 이런 결과가 나올 수 있는가?
# * Prob(F-statistic) : P-value랑 비슷하게 해석 / 0.05보다 크면 데이터를 더 보아야함
# * Log-Likelihood : 로그우도 / 모델을 만들었을 때, 현재 데이터가 관찰될 확률 / 0에 가까울 수록 좋음 / 독립변수가 많아지면 좋아지는 경향이 있음
# * AIC / BIC : 로그우도를 보정 / 낮을 수록 좋음  
# 
# 
# * 높을수록, 낮을수록 좋은 것의 기준은 서로 다른 두 모델의 적합도를 비교하는 것  
# 
# 
# 1. 독립변수의 계수 -> 신뢰구간(부호 확인)
# 2. Prob(F) < 0.05
# 3. 모형 비교 Adj.R-squared는 높을수록 AIC, BIC는 낮을수록 좋음

# In[50]:


child = pandas.read_csv('child.csv')


# In[51]:


child.head()


# In[54]:


res = ols('Aggression ~ Computer_Games', child).fit()


# In[55]:


res.summary()


# Aggression = -0.0068+0.1742*Computer_Games
# 
# 
# 모형 자체의 설명력은 Adj가 아닌 그냥 R-squared를 이용해야함  
# R-squared = 3.5% / 컴퓨터 게임으로 공격을 설명하는 설명력은 3.5% 밖에 되지 않음  
# Prob(F-statistic) = 1.27e-06 < 0.05
# 
# 절편의 신뢰구간의 부호가 바뀌는 것은 중요하지 않음 (예외도 있음)

# In[56]:


res = ols('Aggression ~ Television', child).fit()


# In[57]:


res.summary()


# In[59]:


res = ols('Aggression ~ Television + Computer_Games', child).fit()
# Television, Computer_Games 두 개의 독립변수로 종속변수를 설명


# In[60]:


res.summary()


# Aggression = -0.0029 +0.1353 * Television + 0.1539 * Coputer_Games  
# 독립변수 두개를 합쳐서 회귀식을 구해보니 회귀식의 예측력이 더 향상됨  
# -> 두 개를 합쳐서 설명하는 것이 더 설명력있음.

# In[64]:


child.columns


# In[66]:


res = ols('Aggression ~ Television + Computer_Games + Sibling_Aggression', child).fit()
res.summary()


# In[67]:


res = ols('Aggression ~ Television + Computer_Games + Sibling_Aggression + Diet', child).fit()
res.summary()


# In[76]:


res = ols('Aggression ~ Television + Computer_Games + Sibling_Aggression + Diet + Parenting_Style', child).fit()
res.summary()


# In[75]:


res = ols('Aggression ~ Television + Computer_Games + Sibling_Aggression + Parenting_Style', child).fit()
res.summary()


# In[72]:


res = ols('Aggression ~ Television + Computer_Games + Parenting_Style', child).fit()
res.summary()


# In[73]:


res = ols('Aggression ~ Television + Sibling_Aggression + Parenting_Style', child).fit()
res.summary()


# In[74]:


res = ols('Aggression ~ Computer_Games + Sibling_Aggression + Parenting_Style', child).fit()
res.summary()


# In[77]:


res = ols('Aggression ~ Television + Sibling_Aggression + Computer_Games', child).fit()
res.summary()


# In[ ]:




