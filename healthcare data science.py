#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import numpy as np
import os


# In[4]:


df=pd.read_csv('heart.csv')
df.head()


# In[5]:


df=df.drop(columns=['oldpeak', 'exang','slope','ca','thal','cp'], axis=1)
df


# In[4]:


df.shape


# In[138]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.nunique()


# In[139]:


df['target'].value_counts()


# In[10]:


df['target'].unique()
target_age=pd.crosstab(df['target'],df['sex'])
target_age


# In[11]:


#max heart rate
df['target'].unique()
target_thalach=pd.crosstab(df['target'],df['thalach'])
target_thalach


# In[12]:


# filter with chol,target is not 1 and trestbps(blood presure) more than 170
df[(df['chol']>200) &(df['target']!=0) & (df['trestbps']>170)]


# In[13]:


# from 30 people we have 2 people with thalach 180
def clip_max_heart_rate(max_heart_rate):
    if max_heart_rate>180:
        max_heart_rate=180
        return max_heart_rate
df['restecg'].apply(lambda x:clip_max_heart_rate(x))[:30] 


# In[14]:


# find the average of each features based on the chol and sort the thalach from down to up
df.groupby('chol').mean().sort_values('restecg',ascending=True)


# In[15]:


#categoroized with the age and sex with target(output)
pd.pivot_table(df,index=['sex','age'],values='target')


# In[16]:


sns.countplot(data=df,y='target',palette='hls')
plt.title('amount of the target')
plt.figure(figsize=(20,10))
plt.show


# In[17]:


# abundance of the with age
sns.swarmplot(df['age'])


# In[18]:


sns.pairplot(df[['target','fbs','age','sex','thalach']])


# In[19]:


# ca(major vessel and cp is the chain pain)
sns.countplot(data=df,x='fbs',hue='target')
plt.show


# In[20]:


plt.hist(df['target'])
plt.show()


# In[24]:


sns.relplot('chol','age',data=df,kind='line',ci=None)


# In[25]:


# comapre with men and women that who have more target zero and who have not
fig,ax=plt.subplots(figsize=(10,5))
sns.countplot(df['target'],hue=df['sex'],ax=ax)
plt.xlabel('target')
plt.ylabel('sex')
plt.xticks(rotation=50)
plt.show


# In[26]:


nums=['age','sex','trestbps','chol','trestbps','target']
for i in nums:
    plt.figure(figsize=(20,10))
    sns.jointplot(x=df[i],y=df['target'],kind='reg')
    plt.xlabel(i)
    plt.ylabel('resposne')
    plt.grid()
    plt.show()


# In[8]:


plt.bar(df['target'],df['age'],alpha=.5,width=0.8,label='chart')
plt.show()


# In[62]:


sns.catplot('sex','target',data=df,kind='box',hue='fbs')


# In[53]:


# abundance for each of the columns
import itertools
columns=df.columns[:8]
plt.subplots(figsize=(30,28))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),5,j+1)
    plt.subplots_adjust(wspace=0.3,hspace=0.8)
    df[i].hist(bins=30,edgecolor='black')
    plt.title(i)
plt.show()


# # Finding the outliers

# # With box tucky

# In[4]:


df=pd.read_csv('heart.csv')
df.head()


# In[5]:


df=df.drop(columns=['oldpeak', 'exang','slope','ca','thal','cp'], axis=1)


# In[6]:


x=df.iloc[:,0:4].values
y=df.iloc[:4]
df[:5]


# In[7]:


df.boxplot(return_type='dict')
plt.plot()


# In[215]:


thalach=x[:,1]
iris_outliers=(thalach<40)
df[iris_outliers]


# # Applying tucky outlier labeling

# In[8]:


pd.options.display.float_format='{:.1f}'.format
x_df=pd.DataFrame(df)
x_df.describe()


# In[ ]:


# we want to calculate this:
# iqr(for age) = 61.0 - 48.0 = 13.0
#iqr(1.5)= 19.5
#48.0 - 19.5 = 28.5
# 61.0 + 19.5 = 80.5 


# # Make a model

# In[6]:


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
from sklearn.cluster import KMeans


# In[41]:


df.isnull().sum()


# In[8]:


X = df.drop(columns=['target','age','fbs','age','chol','trestbps','restecg'],axis=1)
y = df['target']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=145)


# # Logistic Regression

# In[10]:


model = LogisticRegression()


# In[11]:


model.fit(X_train,y_train)


# In[12]:


predictions = model.predict(X_test)


# In[13]:


predictions


# In[14]:


score=accuracy_score(y_test,predictions)


# In[15]:


score


# In[50]:


metrics.confusion_matrix(y_test,predictions)


# In[51]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# In[52]:


classification_report(y_test,predictions)


# # Decision Tree

# In[54]:


model=DecisionTreeClassifier()


# In[55]:


model.fit(X_train,y_train)


# In[56]:


predictions=model.predict(X_test)


# In[57]:


predictions


# In[58]:


score=accuracy_score(y_test,predictions)


# In[59]:


score


# In[60]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# # Random Forest

# In[61]:


model=RandomForestClassifier()


# In[62]:


model.fit(X_train,y_train)


# In[63]:


predictions=model.predict(X_test)


# In[64]:


predictions


# In[65]:


score=accuracy_score(y_test,predictions)


# In[66]:


score


# In[129]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# # Neural Networks

# In[67]:


model=MLPClassifier()


# In[68]:


model.fit(X_train,y_train)


# In[70]:


predictions=model.predict(X_test)


# In[71]:


predictions


# In[72]:


score=accuracy_score(y_test,predictions)


# In[73]:


score


# In[74]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# # Suport Vector Machine

# In[75]:


model = SVC()


# In[76]:


model.fit(X_train,y_train)


# In[77]:


predictions=model.predict(X_test)


# In[78]:


predictions


# In[79]:


score=accuracy_score(y_test,predictions)


# In[80]:


score


# # KMeans

# In[81]:


model=KMeans(n_clusters=3)


# In[82]:


model.fit(X_train,y_train)


# In[83]:


df['clusters']=df['target']


# In[84]:


predictions=model.predict(X_test)


# In[85]:


predictions


# In[86]:


score=accuracy_score(y_test,predictions)


# In[87]:


score


# In[88]:


classification_report(y_test,predictions)


# In[17]:


#You can see that the value of root mean squared error is 0.5524, which is almost same  as 10% of the mean value which is 0.05131.
#This means that our algorithm was almsot accurate
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('10% of Mean Price:', df['target'].mean() * 0.1)


# # using anomaly detection with isolation forest

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# In[5]:


model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)


# In[6]:


model.fit(df)


# In[21]:


df['scores']=model.decision_function(df)
df['anomaly']=model.predict(df)
df.head(20)


# In[22]:


anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)


# # Time continuous for thalach

# In[ ]:


import time
import os

def doEvery_1_Seconds():
        print("I do it every 1 seconds")
        time.sleep(1)
TIMER_LIMIT = 1
setTimer = time.time()

while(1):
        def screen_clear():
            if os.name == 'posix':
              _ = os.system('clear')
            else:
              _ = os.system('cls')

        screen_clear()
        
        df = np.random.randint(71,202,size=10)
        df = pd.DataFrame(df, columns=['thalach'])
        print(df)
        if(time.time() - setTimer >= TIMER_LIMIT):  
            break
        else:
                doEvery_1_Seconds()
                setTimer = time.time()


# In[24]:


import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
from IPython.display import clear_output

 

def doEvery_1_Seconds():
    datetime_now = datetime.now()
    print("Heart rate at time {}:{}:{}".format((datetime_now.hour%12),datetime_now.minute,datetime_now.second))
    df = np.random.randint(71,202,size=10)
    df = pd.DataFrame(df, columns=['thalach'])
    print(df)
    time.sleep(1)
#set timer limit in seconds
TIMER_LIMIT = 10

 

start_time = datetime.now()
while(1):
    present_time = datetime.now()
    init = start_time.minute*60 +start_time.second
    ending = present_time.minute*60 + present_time.second
    
    if(ending - init >= TIMER_LIMIT):  
        break
    else:
        clear_output(wait=True)
        doEvery_1_Seconds()
        setTimer = time.time()


# In[ ]:




