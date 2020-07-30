#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import plotly.express as px


# In[29]:


df=pd.read_csv('USA_cars_datasets.csv')
df.head(5)


# In[8]:


df.info()


# In[9]:


df.shape


# In[13]:


df.describe(include=['object','float'])


# In[14]:


df.nunique()


# In[17]:


# compare two attributes with each other
df['brand'].unique()
price_brand=pd.crosstab(df['price'],df['brand'])
price_brand


# In[19]:


df['country'].value_counts()


# In[22]:


# do the filter with some attributes
df[(df['price']>40000) & (df['color']=='white') & (df['brand']=='ford')]


# In[23]:


df[(df['mileage']>500)& (df['year']==2018)]


# In[32]:


def clip_price(price):
    if price> 70000:
        price=70000
        return price
df['price'].apply(lambda x:clip_price(x))[200:300]


# In[36]:


# we sort the price from down to up and also we find the average of the model
df.groupby('model').mean().sort_values('price',ascending=True)


# In[38]:


# categorize with price and brand
pd.pivot_table(data=df,index=['price','brand'],values='year')


# In[41]:


# we have to use feature Engineering because in price we have zero values
median_price = df['price'].median()
df['price'] = df['price'].astype(int)
df['price'].replace(0,median_price ,inplace=True)
df.head()


# # Visulization with seaborn and matplotlib

# In[44]:


sns.countplot(data=df,y='brand',palette='hls')
plt.title('amount of the brand')
plt.figure(figsize=(10,5))
plt.show()


# In[46]:


sns.countplot(data=df,y='state',palette='hls')
plt.title('amount of the state')
plt.figure(figsize=(100,50))
plt.show()


# In[73]:


#I want to display relationship between clean car and salvage insurance status with price.
sns.swarmplot(x='price',y='title_status',data=df)


# In[48]:


plt.hist(df['price'])
plt.show()


# In[85]:


sns.pairplot(df[['price','year','mileage']],kind='reg')


# In[56]:


sns.relplot('country','price',data=df,kind='line',ci=None)


# In[70]:


# the cars with high prices
expensive_cars=df.sort_values('price',ascending=True)
fig=px.bar(expensive_cars,x='brand',y='price',color='price')
fig.show()


# In[72]:


# compare the price and the yaer
expensive_cars=df.sort_values('price',ascending=False)
fig=px.bar(expensive_cars,x='year',y='price',color='price')
fig.show()


# In[77]:


sns.catplot('title_status','price',data=df,kind='box')
plt.show()


# In[81]:


fig,ax=plt.subplots(figsize=(10,5))
sns.countplot(df['year'],hue=df['title_status'],ax=ax)
plt.xlabel('amount of the year')
plt.ylabel('category of title status')
plt.xticks(rotation=50)
plt.show()


# In[92]:


# how many cars ocme from which state
df['state'].value_counts().plot(kind='barh',figsize=(6,10))
plt.show()


# In[4]:


nums=['price','year','mileage']
for i in nums:
    plt.figure(figsize=(6,10))
    sns.jointplot(x=df[i],y=df['price'],kind='reg')
    plt.xlabel(i)
    plt.ylabel('count')
    plt.grid()
    plt.show()


# In[27]:


df['title_status'].unique()


# In[45]:


count_clean_vehicle=len(df[df['title_status']=='clean vehicle'])
count_salvage_insurance=len(df[df['title_status']=='salvage insurance'])
percentage_of_clean_vehicle=count_clean_vehicle/(count_clean_vehicle+count_salvage_insurance)
percentage_of_salvage_insurance=count_salvage_insurance/(count_clean_vehicle+count_salvage_insurance)
print('percentage of clean vehicle is ',percentage_of_clean_vehicle*100)
print('percentage of salvage insurace is ',percentage_of_salvage_insurance*100)


values= [percentage_of_clean_vehicle,percentage_of_salvage_insurance]
labels=['clean vehicle', 'salvage insurance']
plt.title("amount of the safety") 
plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()


# In[ ]:


df


# In[4]:


df.groupby(['brand','model']).price.mean().sort_values().tail(10)


# In[5]:


plt.style.use('seaborn-whitegrid')

df.hist(bins=20,color='red',figsize=(10,5))
plt.show()


# In[6]:


# price estimator
price_estimator_df = df.copy()
features_to_drop = ['vin','lot','country','condition']
price_estimator_df.drop(features_to_drop,axis=1,inplace=True)
price_estimator_df


# In[57]:


df['brand'].unique()


# In[3]:


# percentage of brand i the USA
count_toyota=len(df[df['brand']=='toyota'])
count_ford=len(df[df['brand']=='ford'])
count_dodge=len(df[df['brand']=='dodge'])
count_chevrolet=len(df[df['brand']=='chevrolet'])
count_gmc=len(df[df['brand']=='gmc'])
count_chrysler=len(df[df['brand']=='chrysler'])
count_kia=len(df[df['brand']=='kia'])
count_buick=len(df[df['brand']=='buick'])
count_infiniti=len(df[df['brand']=='infiniti'])
count_mercedes_benz=len(df[df['brand']=='mercedes-benz'])
count_jeep=len(df[df['brand']=='jeep'])
count_bmw=len(df[df['brand']=='bmw'])
count_cadillac=len(df[df['brand']=='cadillac'])
count_hyundai=len(df[df['brand']=='hyundai'])
count_mazda=len(df[df['brand']=='mazda'])
count_honda=len(df[df['brand']=='honda'])
count_heartland=len(df[df['brand']=='heartland'])
count_jaguar=len(df[df['brand']=='jaguar'])
count_acura=len(df[df['brand']=='acura'])
count_harley_davidson=len(df[df['brand']=='harley-davidson'])
count_audi=len(df[df['brand']=='audi'])
count_lincoln=len(df[df['brand']=='lincoln'])
count_lexus=len(df[df['brand']=='lexus'])
count_nissan=len(df[df['brand']=='nissan'])
count_land=len(df[df['brand']=='land'])
count_maserati=len(df[df['brand']=='maserati'])
count_peterbilt=len(df[df['brand']=='peterbilt'])
count_ram=len(df[df['brand']=='ram'])


percentage_of_toyota=count_toyota/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_ford=count_ford/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_dodge=count_dodge/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_chevrolet=count_chevrolet/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_gmc=count_gmc/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_chrysler=count_chrysler/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)


percentage_of_kia=count_kia/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_buick=count_buick/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_infiniti=count_infiniti/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_mercedes_benz=count_mercedes_benz/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_jeep=count_jeep/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_bmw=count_bmw/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_cadillac=count_cadillac/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_hyundai=count_hyundai/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_mazda=count_mazda/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_honda=count_honda/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_heartland=count_heartland/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_jaguar=count_jaguar/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_acura=count_acura/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_harley_davidson=count_harley_davidson/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_audi=count_audi/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_lincoln=count_lincoln/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_lexus=count_lexus/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_nissan=count_nissan/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_land=count_land/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_maserati=count_maserati/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_peterbilt=count_peterbilt/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)

percentage_of_ram=count_ram/(count_toyota+count_ford+count_dodge+count_chevrolet+count_gmc+count_chrysler+count_kia+count_buick+count_infiniti
+count_mercedes_benz+count_jeep+count_bmw+count_cadillac+count_hyundai+count_mazda+count_honda
+count_heartland+count_jaguar+count_acura+count_harley_davidson+count_audi+count_lincoln+count_lexus
+count_nissan+count_land+count_maserati+count_peterbilt+count_ram)




print('percentage of toyota is',percentage_of_toyota*100)
print('percentage of ford is',percentage_of_ford*100)
print('percentage of dodge is',percentage_of_dodge*100)
print('percentage of chevrolet is',percentage_of_chevrolet*100)
print('percentage of gmc is',percentage_of_gmc*100)
print('percentage of kia is',percentage_of_kia*100)
print('percentage of chrysler is',percentage_of_kia*100)
print('percentage of buick is',percentage_of_buick*100)
print('percentage of infiniti is',percentage_of_infiniti*100)
print('percentage of mercedes-benz is',percentage_of_mercedes_benz*100)
print('percentage of jeep is',percentage_of_jeep*100)
print('percentage of bmw is',percentage_of_bmw*100)
print('percentage of cadillac is',percentage_of_cadillac*100)
print('percentage of hyundai is',percentage_of_hyundai*100)
print('percentage of mazda is',percentage_of_mazda*100)
print('percentage of honda is',percentage_of_honda*100)
print('percentage of heartland is',percentage_of_heartland*100)
print('percentage of jaguar is',percentage_of_jaguar*100)
print('percentage of acura is',percentage_of_acura*100)
print('percentage of harley davidson  is',percentage_of_harley_davidson*100)
print('percentage of audi is',percentage_of_audi*100)
print('percentage of lincoln is',percentage_of_lincoln*100)
print('percentage of lexus is',percentage_of_lexus*100)
print('percentage of nissan is',percentage_of_nissan*100)
print('percentage of land is',percentage_of_land*100)
print('percentage of maserati is',percentage_of_maserati*100)
print('percentage of peterbilt is',percentage_of_peterbilt*100)
print('percentage of ram is',percentage_of_ram*100)


values= [percentage_of_toyota,percentage_of_dodge,percentage_of_ford,percentage_of_chevrolet,percentage_of_gmc,percentage_of_kia,percentage_of_chrysler,
        percentage_of_buick,percentage_of_infiniti,percentage_of_mercedes_benz,percentage_of_jeep,percentage_of_bmw,percentage_of_cadillac,percentage_of_hyundai,
        percentage_of_mazda,percentage_of_honda,percentage_of_heartland,percentage_of_jaguar,percentage_of_acura,percentage_of_harley_davidson,
        percentage_of_audi,percentage_of_lincoln,percentage_of_lexus,percentage_of_nissan,percentage_of_land,percentage_of_maserati,percentage_of_peterbilt,percentage_of_ram]
labels=['toyota', 'ford', 'dodge', 'chevrolet', 'gmc', 'chrysler', 'kia',
       'buick', 'infiniti', 'mercedes-benz', 'jeep', 'bmw', 'cadillac',
       'hyundai', 'mazda', 'honda', 'heartland', 'jaguar', 'acura',
       'harley-davidson', 'audi', 'lincoln', 'lexus', 'nissan', 'land',
       'maserati', 'peterbilt', 'ram']
plt.title("amount of the credit") 
plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()


# In[13]:


df['color'].unique()
state_color=pd.crosstab(df['state'],df['color'])
state_color


# In[5]:


# find the best colors among all of them
top_5_color=list(df.color.value_counts()[:6].index)
top_5_color


# In[28]:


df['vin'].nunique()


# # Make the model

# In[53]:


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
from sklearn.linear_model import LinearRegression


# In[31]:


replacestruct={
    "brand":   {'toyota':0, 'ford':1, 'dodge':2, 'chevrolet':3, 'gmc':4, 'chrysler':5, 'kia':6,
       'buick':7, 'infiniti':8, 'mercedes-benz':9, 'jeep':10, 'bmw':11, 'cadillac':12,
       'hyundai':13, 'mazda':14, 'honda':15, 'heartland':16, 'jaguar':17, 'acura':18,
       'harley-davidson':19, 'audi':20, 'lincoln':21, 'lexus':22, 'nissan':23, 'land':24,
       'maserati':25, 'peterbilt':26, 'ram':27},
    
    "model":{'cruiser':0, 'se':1, 'mpv':2, 'door':3, '1500':4, 'pk':5, 'malibu':6, 'coupe':7,
       'wagon':8, 'forte':9, 'encore':10, 'sorento':11, 'doors':12, 'chassis':13, 'q70':14,
       'camaro':15, 'convertible':16, 'vans':17, 'srw':18, 'compass':19, 'enclave':20,
       '300':21, 'cherokee':22, 'pacifica':23, 'x3':24, 'equinox':25, 'challenger':26, 'm':27,
       'colorado':28, 'focus':29, 'durango':30, 'escape':31, 'charger':32, 'explorer':33,
       'f-150':34, '3500':35, 'caravan':36, 'van':37, 'dart':38, '2500':39, 'esv':40,
       'cutaway':41, 'el':42, 'edge':43, 'series':44, 'flex':45, 'srx':46, 'cab':47, 'pickup':48,
       'vehicl':49, 'trax':50, 'tahoe':51, 'suburban':52, 'cargo':53, 'drw':54, 'fiesta':55,
       'impala':56, 'soul':57, 'elantra':58, 'pioneer':59, 'trail':60, 'traverse':61,
       'country':62, 'sundance':63, 'road/street':64, 'nautilus':65, 'gx':66, 'q5':67,
       'gle':68, 'sportage':69, '5':70, 'sport':71, 'discovery':72, 'acadia':73, 'ghibli':74,
       'glc':75, 'e-class':76, 'truck':77, 'utility':78, 'limited':79, 'sl-class':80,
       'cx-3':81, '2500hd':82, 'sonic':83, 'corvette':84, 'mdx':85, 'xt5':86, 'fusion':87,
       'mustang':88, 'passenger':89, 'volt':90, 'spark':91, 'cruze':92, 'ld':93, 'journey':94,
       'transit':94, 'ranger':96, 'taurus':97, 'max':98, 'energi':99, 'expedition':100,
       'bus':101, 'ecosport':102, 'f-750':103, 'd':104, 'dr':105, 'hybrid':106, 'suv':107, 'connect':108,
       'f-650':109, 'sentra':110, 'altima':111, 'frontier':112, 'rogue':113, 'maxima':114,
       'versa':115, 'note':116, 'armada':117, 'pathfinder':118, 'titan':119, 'sedan':120, 'juke':121,
       'murano':122, 'xterra':123, 'kicks':124, 'xd':125, 'nvp':126},
    
    "title_status": {'clean vehicle':0, 'salvage insurance':1},
    
    
    
    "color":{'black':0, 'silver':1, 'blue':2, 'red':3, 'white':4, 'gray':5, 'orange':6,
       'brown':7, 'no_color':8, 'gold':9, 'charcoal':10, 'turquoise':11, 'beige':12,
       'green':13, 'dark blue':14, 'maroon':15, 'phantom black':16, 'yellow':17,
       'color:':18, 'light blue':19, 'toreador red':20, 'bright white clearcoat':21,
       'billet silver metallic clearcoat':22, 'black clearcoat':23,
       'jazz blue pearlcoat':24, 'purple':25,
       'ruby red metallic tinted clearcoat':26, 'triple yellow tri-coat':27,
       'competition orange':28, 'off-white':29, 'shadow black':30,
       'magnetic metallic':31, 'ingot silver metallic':32, 'ruby red':33,
       'royal crimson metallic tinted clearcoat':34, 'kona blue metallic':35,
       'oxford white':36, 'lightning blue':37, 'ingot silver':38,
       'white platinum tri-coat metallic':39, 'guard':40,
       'tuxedo black metallic':41, 'tan':42, 'burgundy':43, 'super black':44,
       'cayenne red':45, 'morningsky blue':46,'pearl white':47 ,'glacier white':48},
    
    "state":{'new jersey':0, 'tennessee':1, 'georgia':2, 'virginia':3, 'florida':4,
       'texas':5, 'california':6, 'north carolina':7, 'ohio':8, 'new york':9,
       'pennsylvania':10, 'south carolina':11, 'michigan':12, 'washington':13,
       'arizona':14, 'utah':15, 'kentucky':16, 'massachusetts':17, 'nebraska':18,
       'ontario':19, 'missouri':20, 'minnesota':21, 'oklahoma':22, 'connecticut':23,
       'indiana':24, 'arkansas':25, 'kansas':26, 'wyoming':27, 'colorado':28, 'illinois':29,
       'wisconsin':30, 'mississippi':31, 'maryland':32, 'oregon':33, 'west virginia':34,
       'nevada':35, 'rhode island':36, 'louisiana':37, 'alabama':38, 'new mexico':39,
       'idaho':40, 'new hampshire':41, 'montana':42, 'vermont':43},
    
    "country" : {' usa':0, ' canada':1},
    
    "condition": {'10 days left':0, '6 days left':1, '2 days left':2, '22 hours left':3,
       '20 hours left':4, '19 hours left':5, '3 days left':6, '21 hours left':7,
       '17 hours left':8, '2 hours left':9, '3 hours left':10, '34 minutes':11,
       '16 hours left':12, '18 hours left':13, '1 days left':14, '32 minutes':15,
       '14 hours left':16 ,'5 hours left':17, '4 days left':18, '9 days left':19,
       '23 hours left':20,'8 days left':21, '7 days left':22, '5 days left':23,
       '9 minutes':24, '1 minutes':25, '7 hours left':26, '16 minutes':27,
       '6 hours left':28, '1 hours left':29, 'Listing Expired':30,'13 days left':31,
       '24 hours left':32, '15 hours left':33, '53 minutes':34 ,'27 minutes':35,
       '12 days left':36 ,'15 days left':37, '30 minutes':38 ,'29 minutes':39,
       '28 minutes':40,'48 minutes':41, '11 days left':42 ,'4 hours left':43,
       '47 minutes':44, '12 hours left':45, '36 minutes':46}
}

df=df.replace(replacestruct)
df.head(5)


# In[32]:


df=df.drop(columns=['vin','Unnamed: 0'],axis=1)
df.head(5)


# In[68]:


x = df['price'].values.reshape(-1,1)
y = df['price'].values.reshape(-1,1)


# In[69]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[70]:


model = DecisionTreeClassifier()  
model.fit(x_train, y_train)


# In[74]:


predictions=model.predict(y_test)


# In[75]:


predictions


# In[76]:


score=accuracy_score(y_test,predictions)


# In[77]:


score


# In[78]:


classification_report(y_test,predictions)


# # Random Forest 

# In[80]:


model=RandomForestClassifier()


# In[81]:


model.fit(x_train,y_train)


# In[82]:


predictions=model.predict(y_test)


# In[83]:


predictions


# In[84]:


score=accuracy_score(y_test,predictions)


# In[85]:


score


# In[87]:


#You can see that the value of root mean squared error is 42.839, which is much lower than 10% of the mean value which is 1876.
#This means that our algorithm was not very accurate
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('10% of Mean Price:', df['price'].mean() * 0.1)


# In[ ]:




