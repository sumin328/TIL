#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[2]:


from statsmodels.formula.api import ols


# In[3]:


child = pandas.read_csv('child.csv')


# In[4]:


ols('Aggression ~ Television', child).fit().summary()


# ## 더미코딩

# In[8]:


import numpy.random


# In[9]:


child.shape


# In[15]:


child['TV2'] = child['']


# In[16]:


ols('Aggression ~ Television + TV2', child).fit().summary()


# 

# In[27]:


from sklearn.linear_model import LinearRegression, ElasticNetCV


# In[28]:


x = child.iloc[:, 1:6]        # 1번열부터 6번열까지를 x
y = child['Aggression']   # Aggression 열을 y


# In[33]:


x.head()


# ## 선형 모형

# In[29]:


m1 = LinearRegression()


# In[30]:


m1.fit(x,y)


# In[31]:


m1.coef_


# ## 엘라스틱넷 + 교차 검증
# 'l1_ratio' : 교안에서 a, 라쏘(L1), 릿지(L2)

# In[37]:


m2 = ElasticNetCV(l1_ratio=[.1, .5, 1], cv = 3)    # 3-fold 교차검증 l1_ration는 라쏘를 얼마나 할 것인지  # L1을 .1로, .5로, 1로 각 각 세번씩 교차검증
m2.fit(x,y)
m2.coef_


# In[38]:


m2.l1_ratio_                    # 계산을 돌린 후의 결과는 뒤에 _를 붙임 # 그 결과 나온 최적의 ratio


# In[39]:


m2.alpha_                       # 람다에 해당하는 부분 -> 계수에 0.008 곱할때 최적


# ## 더미코딩

# In[41]:


hsb = pandas.read_csv('hsb2.csv')


# In[42]:


hsb.head()   # SES:Socio-Economix Status 사회경제적 수준


# In[43]:


ols('read ~ C(race)', hsb).fit().summary()  # C 이용해서 더미코딩


# read = 46.6 + 5.24*T2 + 0.13*T3 + 7.25*T4 

# ## pandas.get_dummies

# In[62]:


race = pandas.get_dummies(hsb['race'], prefix = 'race') # 변수명 앞에 race라고 붙여라


# In[63]:


new = pandas.concat([hsb, race], axis = 1) # hsb와 dummy를 이어붙임(concat) 


# In[65]:


ols('read~race_2+race_3+race_4', new).fit().summary() # dummy 코딩을 직접 함, race1이 기준 / C(race)를 하면 dummy코딩을 자동으로 해줌 


# ## .replace로 컬럼값 바꾸기

# In[70]:


hsb['race2'] = hsb['race'].replace({1:'_흑', 2:'백', 3:'황', 4:'기타'})


# In[71]:


ols('read ~ C(race2)', hsb).fit().summary()


# In[73]:


hsb['schtyp2'] = hsb['schtyp']-1


# In[75]:


ols('read ~ ses + schtyp2', hsb).fit().summary()


# In[76]:


ols('read ~ ses + ses:schtyp2', hsb).fit().summary() # ses:schtyp2 -> 상호작용


# ses:schtyp2 의 신뢰구간이 음수, 양수를 모두 포함하고 있으므로 공립, 사립에 따라 ses가 읽기능력에 영향을 끼치는 정도에 차이를 끼친다고 할 수 없다.

# In[77]:


ols('science ~ ses + female:ses', hsb).fit().summary() 


# In[78]:


ols('write ~ read + female:read', hsb).fit().summary() 


# In[79]:


ols('write ~ read + female:read + female', hsb).fit().summary() 


# In[81]:


ols('write ~ female * read', hsb).fit().summary() # write ~ read + female + read:female


# In[ ]:




