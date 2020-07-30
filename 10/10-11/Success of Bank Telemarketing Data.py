#!/usr/bin/env python
# coding: utf-8

# # we want to implement the program with alpha bank(telemarketing data)

# In[32]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[33]:


df=pd.read_csv('Alpha_bank.csv')
df.sample(10)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe(include='object')


# In[11]:


df[df['Subscribed']!='no'] 


# In[12]:


df.nunique()


# In[9]:


df['Subscribed'].value_counts()


# In[10]:


def clip_Age(age):
    if age>60:
        age=60
        return age
df['Age'].apply(lambda x:clip_Age(x))[:40]    


# In[11]:


df['Subscribed'].unique()
Subscribed_Housing_Loan=pd.crosstab(df['Subscribed'],df['Housing_Loan'])
Subscribed_Housing_Loan


# In[12]:


df[(df['Age']>50)&(df['Subscribed']!='no')][:10]


# In[13]:


df.groupby('Job').mean().sort_values('Age',ascending=True)


# In[13]:


pd.pivot_table(df,index=['Subscribed','Job','Education'],values='Age')


# # using the outliers

# In[15]:


df.boxplot(return_type='dict')
plt.plot()


# In[16]:


pd.options.display.float_format='{:.1f}'.format
x_df=pd.DataFrame(df)
x_df.describe()


# In[ ]:


#10 and 66 the outliers for age is between 10 and 66


# # we want to plot with seaborn and matplotlib

# In[17]:


sns.countplot(data=df,y='Subscribed',palette='hls')
plt.title('amount of the Subscribed')
plt.show()


# In[18]:


sns.countplot(data=df,y='Education',palette='hls')
plt.title('categorize the level')
plt.show()


# In[6]:


count_Primary_Education_transaction=len(df[df['Education']=='Primary_Education'])
count_Secondary_Education_transaction=len(df[df['Education']=='Secondary_Education'])
count_Professional_Education_transaction=len(df[df['Education']=='Professional_Education'])
count_Tertiary_Education_transaction=len(df[df['Education']=='Tertiary_Education'])
percentage_of_Primary_Education=count_Primary_Education_transaction/(count_Primary_Education_transaction+count_Secondary_Education_transaction+count_Professional_Education_transaction+count_Tertiary_Education_transaction)
percentage_of_Secondary_Education=count_Secondary_Education_transaction/(count_Primary_Education_transaction+count_Secondary_Education_transaction+count_Professional_Education_transaction+count_Tertiary_Education_transaction)
percentage_of_Professional_Education=count_Professional_Education_transaction/(count_Primary_Education_transaction+count_Secondary_Education_transaction+count_Professional_Education_transaction+count_Tertiary_Education_transaction)
percentage_of_Tertiary_Education=count_Tertiary_Education_transaction/(count_Primary_Education_transaction+count_Secondary_Education_transaction+count_Professional_Education_transaction+count_Tertiary_Education_transaction)
print('percentage of Primary Education is:',percentage_of_Primary_Education*100)
print('percentage of secondary Education is:',percentage_of_Secondary_Education*100)
print('percentage of Primary Education is:',percentage_of_Professional_Education*100)
print('percentage of Primary Education is:',percentage_of_Tertiary_Education*100)

values= [percentage_of_Primary_Education, percentage_of_Secondary_Education,percentage_of_Professional_Education,percentage_of_Tertiary_Education]
labels=['Primary_Education', 'Secondary_Education',
       'Professional_Education', 'Tertiary_Education']
plt.title("amount of the credit") 
plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()


# In[16]:


df['Education'].unique()


# In[23]:


fig,ax=plt.subplots(figsize=(10,5))
sns.countplot(df['Subscribed'],hue=df['Marital_Status'],ax=ax)
plt.xlabel('amount of the Subscribed')
plt.ylabel('marital')
plt.xticks(rotation=50)
plt.show()


# In[7]:


fig,ax=plt.subplots(figsize=(10,5))
sns.countplot(df['Subscribed'],hue=df['Education'],ax=ax)
plt.xlabel('amount of the Subscribed')
plt.ylabel('education')
plt.xticks(rotation=50)
plt.show()


# In[32]:


df.hist(figsize=(10,10),color="blueviolet",grid=False)
plt.show()


# In[37]:


sns.boxenplot(df['Age'])


# In[43]:


plt.style.use('seaborn-whitegrid')

df.hist(bins=20,color='red',figsize=(10,5))
plt.show()


# # we want the implement the program with some algorithms

# In[34]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans


# In[35]:


replacestruct={
    "Job":  {'housemaid':0, 'services':1, 'admin.':2, 'technician':3, 'blue-collar':4,
       'unemployed':5, 'retired':6, 'entrepreneur':7, 'management':8, 'student':9,'self-employed':10 },
    
    "Marital_Status": {'married':0, 'single':1, 'divorced':2},
    
    "Education":  {'Primary_Education':0, 'Secondary_Education':1,'Professional_Education':2, 'Tertiary_Education':3},
    
    "Default_Credit":  {'no':0, 'yes':1},
    
    "Housing_Loan":{'no':0, 'yes':1},
    
    "Personal_Loan":{'no':0, 'yes':1},
    
    "Subscribed": {'no':0, 'yes':1}
}
    
df=df.replace(replacestruct)
df.head(5)


# In[36]:


df=df.drop('Default_Credit',axis=True)
df.head()


# In[37]:


train_features = df.iloc[:80,:-1]
test_features = df.iloc[80:,:-1]
train_targets = df.iloc[:80,-1]
test_targets = df.iloc[80:,-1]


# In[14]:


test_targets


# # DecisionTree model

# In[78]:


model = DecisionTreeClassifier()


# In[79]:


model.fit(train_features,train_targets)


# In[80]:


predictions = model.predict(test_features)


# In[81]:


predictions


# In[82]:


score=accuracy_score(test_targets,predictions)


# In[83]:


score


# In[84]:


classification_report(test_targets,predictions)


# # Random Forest model

# In[85]:


model=RandomForestClassifier()


# In[86]:


model.fit(train_features,train_targets)


# In[87]:


predictions=model.predict(test_features)


# In[88]:


predictions


# In[89]:


score=accuracy_score(test_targets,predictions)


# In[90]:


score


# In[91]:


sns.heatmap(confusion_matrix(test_targets,predictions), annot=True, cmap="mako")


# # Neural Network model

# In[92]:


model=MLPClassifier()


# In[93]:


model.fit(train_features,train_targets)


# In[94]:


predictions=model.predict(test_features)


# In[95]:


predictions


# In[96]:


score=accuracy_score(test_targets,predictions)


# In[97]:


score


# # use anamoly detection with isolation forest

# In[98]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# In[101]:


model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)


# In[102]:


model.fit(df)


# In[103]:


df['scores']=model.decision_function(df)
df['anomaly']=model.predict(df)
df.head(20)


# In[104]:


anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)


# # support vector machine model

# In[105]:


model=SVC()


# In[106]:


model.fit(train_features,train_targets)


# In[107]:


predictions=model.predict(test_features)


# In[108]:


predictions


# In[109]:


score=accuracy_score(test_targets,predictions)


# In[110]:


score


# In[ ]:





# In[ ]:




