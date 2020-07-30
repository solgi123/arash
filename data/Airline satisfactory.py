#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[140]:


df.columns


# In[3]:


# we have 2 data sets include test and tarin and test is 20%
df=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
df.sample(5)


# # data cleaning

# In[106]:


# we need to omit the unnamed,Arrival Delay in Minutes and ID
df=df.drop(columns=['Unnamed: 0','id','Arrival Delay in Minutes'])
df_test=df_test.drop(columns=["Unnamed: 0","id","Arrival Delay in Minutes"])


# In[107]:


df.head()


# In[7]:


# information for dataset
df.info()


# In[19]:


df.shape


# In[20]:


df.describe()


# In[22]:


df.nunique()


# In[25]:


# compare withg 2 attributes
df['satisfaction'].unique()
satisfaction_Class=pd.crosstab(df['satisfaction'],df['Class'])
satisfaction_Class


# In[3]:


df['Gender'].unique()
Gender_satisfaction=pd.crosstab(df['Gender'],df['satisfaction'])
Gender_satisfaction


# In[32]:


# lets have a filter with some Iot
df[(df['Age']>70) & (df['satisfaction']!='satisfied') & (df['Cleanliness']==5)]


# In[35]:


#find the average of each features based on the services and sort the flight distances from down to up
df.groupby('Checkin service').mean().sort_values('Flight Distance',ascending=True)


# In[40]:


#categoroized with the age and type of travel and class with flight distances and satisfaction(output is cleanliness)
pd.pivot_table(df,index=['Age','Type of Travel','Class','Flight Distance','satisfaction'],values='Cleanliness')


# In[34]:


# show the numbers from zero to 30 with cleanliness with larger than 4
def clip_Cleanliness(Cleanliness):
    if Cleanliness>4:
        Cleanliness=4
        return Cleanliness
df['Cleanliness'].apply(lambda x:clip_Cleanliness(x))[:30]


# # plot the model with seaborn and matplotlib

# In[8]:


sns.countplot(data=df,y='satisfaction',palette='hls')
plt.title('amount of the satisfaction')
plt.figure(figsize=(10,5))
plt.show()


# In[9]:


sns.countplot(data=df,y='Class',palette='hls')
plt.title('amount of the class')
plt.figure(figsize=(10,5))
plt.show()


# In[6]:


sns.relplot('Class','Food and drink',data=df,kind='line',ci=None,hue='satisfaction')


# In[10]:


fig,ax=plt.subplots(figsize=(10,6))
sns.countplot(df['satisfaction'],hue=df['Type of Travel'],ax=ax)
plt.xlabel('satisfaction')
plt.ylabel('type of travel')
plt.xticks(rotation=60)
plt.show()


# In[11]:


sns.pairplot(df[['satisfaction','Cleanliness','Age','Type of Travel','Class']])


# In[38]:


sns.boxenplot(df['Age'])


# In[12]:


plt.style.use('seaborn-whitegrid')
df.hist(bins=20,color='red',figsize=(20,10))
plt.show()


# In[13]:


# compare the numerical attributes
nums=['Cleanliness','Age','Inflight wifi service','Seat comfort']

for i in nums:
    plt.figure(figsize=(10,5))
    sns.jointplot(x=df[i],y=df['Ease of Online booking'],kind='reg')
    plt.xlabel(i)
    plt.ylabel('count')
    plt.grid()
    plt.show()


# # finding the outliers with tucky box

# In[39]:


# finding the ouliers
df.boxplot('Cleanliness',return_type='dict')
plt.plot()


# In[26]:


pd.options.display.float_format='{:.1f}'.format
x_df=pd.DataFrame(df)
x_df.describe()


# In[14]:


count_satisfied=len(df[df['satisfaction']=='satisfied'])
count_neutral_or_dissatisfied=len(df[df['satisfaction']=='neutral or dissatisfied'])
percentage_of_satisfied=count_satisfied/(count_satisfied+count_neutral_or_dissatisfied)
percentage_of_neutral_or_dissatisfied=count_neutral_or_dissatisfied/(count_neutral_or_dissatisfied+count_satisfied)
print('percentage of satisfied is:',percentage_of_satisfied*100)
print('percentage of neutral or dissatisfied is:',percentage_of_neutral_or_dissatisfied*100)

values= [percentage_of_satisfied, percentage_of_neutral_or_dissatisfied]
labels=['satisfied','neutral or dissatisfied']
plt.title("amount of the credit") 
plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()


# In[15]:


df['Class'].unique()


# In[16]:


count_Eco=len(df[df['Class']=='Eco'])
count_Eco_Plus=len(df[df['Class']=='Eco Plus'])
count_Business=len(df[df['Class']=='Business'])
percentage_of_Eco=count_Eco/(count_Eco+count_Eco_Plus+count_Business)
percentage_of_Eco_Plus=count_Eco_Plus/(count_Eco+count_Eco_Plus+count_Business)
percentage_of_Business=count_Business/(count_Eco+count_Eco_Plus+count_Business)
print('percentage of Eco is:',percentage_of_Eco*100)
print('percentage of Eco Plus is:',percentage_of_Eco_Plus*100)
print('percentage of Business is:',percentage_of_Business*100)


values= [percentage_of_Eco, percentage_of_Eco_Plus,percentage_of_Business]
labels=['Eco','Eco Plus','Business']
plt.title("amount of the credit") 
plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()


# In[17]:


plt.figure(figsize=(30,16))
sns.heatmap(df.corr())
plt.show()


# # use anomaly detection with isolation forest

# In[59]:


from sklearn.ensemble import IsolationForest


# In[60]:


model=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1),max_features=1.0)


# In[62]:


replacestruct={
    
    "Gender":  {"Male":0 , "Female":1},
    
    "Customer Type":{'Loyal Customer':0, 'disloyal Customer':1},
    
    "Class":{ 'Eco Plus':0, 'Business':1, 'Eco':2},
    
    "satisfaction":{'neutral or dissatisfied':0, 'satisfied':1},
    
    "Type of Travel": {'Personal Travel':0, 'Business travel':1}
    
}

df=df.replace(replacestruct)
df.head()


# In[71]:


df[df==np.inf]=np.nan
df.fillna(df.mean(), inplace=True)


# In[72]:


model=IsolationForest(n_estimators=10, max_samples='auto', contamination=float(.04),                         max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0,behaviour='new')
model.fit(df)


# In[73]:


df['scores']=model.decision_function(df)
df['anomaly']=model.predict(df)
df.head(20)


# In[74]:


anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)


# # make a model with some algorithms

# In[4]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans


# In[5]:


replacestruct={
    
    "Gender":  {"Male":0 , "Female":1},
    
    "Customer Type":{'Loyal Customer':0, 'disloyal Customer':1},
    
    "Class":{ 'Eco Plus':0, 'Business':1, 'Eco':2},
    
    "satisfaction":{'neutral or dissatisfied':0, 'satisfied':1},
    
    "Type of Travel": {'Personal Travel':0, 'Business travel':1},
    
    "satisfaction":{'satisfied':0, 'neutral or dissatisfied':1}
    
}

df_test=df_test.replace(replacestruct)
df_test.head()


# In[6]:


df_test=df_test.drop(columns=["Unnamed: 0","id","Arrival Delay in Minutes"])


# In[7]:


df_test.head()


# In[8]:


oneHotCols=["Gender","Customer Type","Type of Travel","Class"]
df_test=pd.get_dummies(df_test, columns=oneHotCols)


# In[9]:


df_test.head()


# In[10]:


x=df_test.drop(columns=['satisfaction','Age','Gate location','Departure/Arrival time convenient','Ease of Online booking','Food and drink','Online boarding',
'Seat comfort','Inflight entertainment','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes','Flight Distance','Inflight wifi service'],axis=1)
y = df_test[['satisfaction']]


# In[11]:


x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=7)


# # Making the model with Decision Tree

# In[12]:


model=DecisionTreeClassifier()


# In[192]:


model.fit(x_train,y_train)


# In[193]:


predictions=model.predict(x_test)


# In[194]:


predictions


# In[196]:


score=accuracy_score(y_test,predictions)


# In[197]:


score


# In[200]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
metrics.confusion_matrix(y_test,predictions)


# In[201]:


classification_report(y_test,predictions)


# # Making the model with Random Forest

# In[202]:


model=RandomForestClassifier()


# In[204]:


model.fit(x_train,y_train)


# In[205]:


predictions=model.predict(x_test)


# In[206]:


predictions


# In[208]:


score=accuracy_score(y_test,predictions)


# In[209]:


score


# In[210]:


metrics.confusion_matrix(y_test,predictions)


# In[211]:


classification_report(y_test,predictions)


# # Making the model with Logistic Regression

# In[212]:


model=LogisticRegression()


# In[213]:


model.fit(x_test,y_test)


# In[214]:


predictions=model.predict(x_test)


# In[215]:


predictions


# In[217]:


score=accuracy_score(y_test,predictions)


# In[218]:


score


# In[219]:


metrics.confusion_matrix(y_test,predictions)


# In[220]:


classification_report(y_test,predictions)


# # Make a model with svm

# In[13]:


model=SVC()


# In[14]:


model.fit(x_train,y_train)


# In[15]:


predictions=model.predict(x_test)


# In[16]:


predictions


# In[17]:


score=accuracy_score(y_test,predictions)


# In[18]:


score


# In[19]:


classification_report(y_test,predictions)


# In[ ]:




