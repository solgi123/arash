#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import model_selection#foe train and test the program
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
#import pylab as plb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# ## تعریف مجموعه داده
# این مجموعه داده درباره کمپین بازاریابی یک بانک در پرتغال است. ستون هدف ستون آخر شامل بله یا خیر است#. 
# 

# In[2]:


df = pd.read_csv('banking_updated.csv')
df.head()


# **Input variables:**
# 
# 1 - age (numeric)  
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')  
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)  
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')  
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')  
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')  
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')  
# 8 - contact: contact communication type (categorical: 'cellular','telephone')  
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')  
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')  
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no')  
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)  
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')  
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)  
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)  
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)  
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)  
# 20 - nr.employed: number of employees - quarterly indicator (numeric)  
# 
# **Output variable (desired target):**  
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

# In[3]:


df.describe()


# In[4]:


import seaborn as sns


# In[5]:


sns.countplot(x='y', data=df)


# In[6]:


sns.countplot(y='job', data=df)


# In[7]:


sns.countplot(x='marital', data=df)


# In[8]:


df.education.value_counts()


# In[9]:


sns.countplot(y='education', data=df)


# In[10]:


df.head()


# ### پیش پردازش مجموعه داده
# 

# In[11]:


le = preprocessing.LabelEncoder()


# In[12]:


df.job = le.fit_transform(df.job)


# In[13]:


df.marital = le.fit_transform(df.marital)


# In[14]:


df.education = le.fit_transform(df.education)
df.housing = le.fit_transform(df.housing)
df.loan = le.fit_transform(df.loan)
df.poutcome = le.fit_transform(df.poutcome)


# In[15]:


df.head()


# In[22]:


df.shape


# In[23]:


X = df.iloc[:,0:14]
X[0:10]


# In[24]:


y = df.iloc[:,14]
y[0:10]


# ### تعیین مجموعه داده تست و آموزش
# 

# In[25]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0) #80/20 split


# In[20]:


x_train.shape, y_train.shape


# In[21]:


x_test.shape, y_test.shape


# ### ایجاد مدل پیش بینی 
# 

# #### Logistic Regression

# In[23]:


model=LogisticRegression(penalty='l2', max_iter=1000)


# In[24]:


model.fit(x_train, y_train)


# In[93]:


prediction=model.predict(x_test)


# In[94]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)


# In[95]:


prediction


# In[96]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, prediction)
print(confusion_matrix)


# #### Support Vector Machine (SVM)

# In[97]:


from sklearn.svm import SVC
clf = SVC()


# In[98]:


clf.fit(x_train, y_train)


# In[99]:


pred = clf.predict(x_test)


# In[100]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)0


# #### Random Forest Classifier

# In[101]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


# In[102]:


rfc.fit(x_train, y_train)


# In[103]:


predict = rfc.predict(x_test)


# In[104]:


accuracy_score(y_test, predict)


# #### Neural Network

# In[4]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=20000, alpha=0.3, random_state=0)
mlp.fit(x_train, y_train)


# In[5]:


predict = mlp.predict(x_test)


# In[113]:


accuracy_score(y_test, predict)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




