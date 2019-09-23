#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[2]:


sleep = pandas.read_csv('sleep.csv')


# In[3]:


sleep.head()


# In[4]:


전체평균 = sleep['extra'].mean()


# In[5]:


X = sleep['extra']


# In[7]:


import numpy


# In[18]:


전체SS = numpy.sum((X - 전체평균) ** 2)


# In[19]:


전체SS


# In[20]:


sleep.groupby('group').agg({'extra':'mean'})
# sleep을 group 변수에 따라(by) 그룹을 지어 (group)
# extra 컬럼을 평균(mean)내라
# agg(aggregate)의 약자 (모아서 통계내다)


# In[21]:


처치SS = (
        10* (0.75 - 전체평균) ** 2 +  # 1번 그룹의 10명의 사람
        10 *(2.33 - 전체평균) ** 2) # 2번 그룹의 10명의 사람


# In[22]:


처치SS


# In[25]:


에타제곱 = 처치SS / 전체SS


# In[26]:


에타제곱 # 집단간 차이가 수면시간의 (분산의) 16.1%를 설명한다. 


# In[28]:


코헨의d = (2.33 - 0.75) / sleep['extra'].std()


# In[31]:


코헨의d     # 두 집단이 0.78 표준편차 차이가 난다. 


# In[32]:


코헨의d * 15   # IQ로 예를 들면 11점 정도 차이가 난다. 아이큐의 1표준편차 = 15점


# In[72]:


from numpy.random import normal


# In[73]:


A = normal(10, 1, 1000) # 평균 10, 표준편차 1, 데이터 1000개
B = normal(9, 1, 1000) # 평균 9


# In[74]:


epsilon = 0.1


# In[79]:


from numpy.random import uniform


# In[80]:


import random


# In[96]:


count = {'A':0, 'B':0}
reward = {'A':0, 'B':0}
value = {'A':0, 'B':0}

for _ in range(100):
    if uniform(0,1) < epsilon:
        if random.choice('AB') == 'A':
            reward['A'] += normal(10,1)
            count['A'] += 1
            value['A'] = reward['A'] / count['A']
            print('A')
    else : 
        if value['A'] > value['B']:
            reward['A'] += normal(10, 1)
            count['A'] += 1
            value['A'] = reward['A'] / count['A']
            print('A')
        else :
            reward['B'] += normal(9.1)
            count['B'] += 1
            value['B'] = reward['B']/count['B']
            print('B')


# In[86]:


from numpy.random import normal
random.choice('AB')

