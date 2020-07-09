#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import numpy as np
import os


# In[2]:


df=pd.read_csv('heart.csv')
df.head()


# In[3]:


df=df.drop(columns=['oldpeak', 'exang','slope','ca','thal','cp'], axis=1)
df


# In[4]:


df.shape


# In[43]:


df.info()


# In[44]:


df.describe()


# In[36]:


df.nunique()


# In[45]:


df['target'].unique()
target_age=pd.crosstab(df['target'],df['age'])
target_age


# In[100]:


df['target'].unique()
target_max_heart_rate=pd.crosstab(df['target'],df['max heart rate'])
target_max_heart_rate


# In[50]:


# filter with chol,target is not 1 and trestbps more than 170
df[(df['chol']>200) &(df['target']!=1) & (df['trestbps']>170)]


# In[51]:


# from 30 people we have 2 people with thalach 180
def clip_max_heart_rate(max_heart_rate):
    if max_heart_rate>180:
        max_heart_rate=180
        return max_heart_rate
df['restecg'].apply(lambda x:clip_max_heart_rate(x))[:30] 


# In[52]:


# find the average of each features based on the chol and sort the thalach from down to up
df.groupby('chol').mean().sort_values('restecg',ascending=True)


# In[53]:


# as you see ,we have 4 chest pain and we have categorized with the target(Typical Angina,Atypical Angina,Non-Anginal,Asymptomatic)
pd.pivot_table(df,index=['sex','age'],values='target')


# In[54]:


sns.countplot(data=df,y='target',palette='hls')
plt.title('amount of the target')
plt.figure(figsize=(20,10))
plt.show


# In[141]:


sns.countplot(data=df,y='chest pain',palette='hls')
plt.title('categorized of the chain pain')
plt.figure(figsize=(20,10))
plt.show


# In[55]:


# abundance of the with age
sns.swarmplot(df['age'])


# In[56]:


# ca(major vessel and cp is the chain pain)
sns.countplot(data=df,x='fbs',hue='target')
plt.show


# In[57]:


# consider the oldpeak with age that the highest one more than 6 and for age 60(almost)
# in this figure you can see the age more than 50,they have more oldpeak
sns.relplot('chol','age',data=df,kind='line',ci=None)


# In[58]:


# comapre with men and women that who have more target zero and who have not
fig,ax=plt.subplots(figsize=(10,5))
sns.countplot(df['target'],hue=df['sex'],ax=ax)
plt.xlabel('target')
plt.ylabel('sex')
plt.xticks(rotation=50)
plt.show


# In[104]:


nums=['age','sex','trestbps','chol','blood sugar','target']
for i in nums:
    plt.figure(figsize=(20,10))
    sns.jointplot(x=df[i],y=df['target'],kind='reg')
    plt.xlabel(i)
    plt.ylabel('resposne')
    plt.grid()
    plt.show()


# In[60]:


plt.bar(df['target'],df['age'],alpha=.5,width=0.8,label='chart')
plt.show()


# In[62]:


sns.catplot('sex','target',data=df,kind='box',hue='fbs')


# In[63]:


# abundance for each of the columns
import itertools
columns=df.columns[:14]
plt.subplots(figsize=(30,28))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),5,j+1)
    plt.subplots_adjust(wspace=0.3,hspace=0.8)
    df[i].hist(bins=30,edgecolor='black')
    plt.title(i)
plt.show()


# In[64]:


df.isnull().sum()


# In[5]:


import warnings
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# In[100]:


X = df.drop(columns=['target','age','fbs','age','chol','trestbps','restecg'],axis=1)
y = df['target']


# In[101]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=145)


# # Logistic Regression

# In[102]:


model = LogisticRegression()


# In[103]:


model.fit(X_train,y_train)


# In[104]:


predictions = model.predict(X_test)


# In[105]:


predictions


# In[106]:


score=accuracy_score(y_test,predictions)


# In[107]:


score


# In[14]:


metrics.confusion_matrix(y_test,predictions)


# In[15]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# # Decision Tree

# In[108]:


# decision tree
model=DecisionTreeClassifier()


# In[109]:


model.fit(X_train,y_train)


# In[110]:


predictions=model.predict(X_test)


# In[111]:


predictions


# In[112]:


score=accuracy_score(y_test,predictions)


# In[113]:


score


# In[128]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# # Random Forest

# In[114]:


model=RandomForestClassifier()


# In[115]:


model.fit(X_train,y_train)


# In[116]:


predictions=model.predict(X_test)


# In[117]:


predictions


# In[118]:


score=accuracy_score(y_test,predictions)


# In[119]:


score


# In[129]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# # Neural Networks

# In[122]:


model=MLPClassifier()


# In[123]:


model.fit(X_train,y_train)


# In[124]:


predictions=model.predict(X_test)


# In[125]:


predictions


# In[126]:


score=accuracy_score(y_test,predictions)


# In[127]:


score


# In[130]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# In[134]:


model = SVC()


# In[135]:


model.fit(X_train,y_train)


# In[136]:


predictions=model.predict(X_test)


# In[137]:


predictions


# In[138]:


score=accuracy_score(y_test,predictions)


# In[139]:


score


# In[ ]:




