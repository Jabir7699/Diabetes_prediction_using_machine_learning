#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot
import seaborn as sns


# In[2]:


data=pd.read_csv("diabetes.csv")


# In[19]:


data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data.dtypes.value_counts()
np.unique(data['Outcome'])


# In[5]:


data.shape


# In[24]:


data.info()


# In[25]:


data_c=data.copy()


# In[7]:


data_c.describe()
pd.set_option('display.max_columns',10)
data.describe()


# In[8]:


data_c.columns


# In[9]:


sns.distplot(data_c['Pregnancies'],kde=False)
sns.boxplot(y=data_c['Pregnancies'])


# In[26]:


sns.distplot(data_c['Pregnancies'],kde=False)
sns.boxplot(y=data_c['Pregnancies'])


# In[20]:


#variable Glucose
sns.distplot(data_c['Glucose'],kde=False)
sns.boxplot(y=data_c['Glucose'])


# In[21]:


#variable BP
sns.distplot(data_c['BloodPressure'],kde=False)
sns.boxplot(y=data_c['BloodPressure'])
#Hence the data is clean


# In[22]:


sns.regplot(x=data_c['Pregnancies'],y=data_c['Outcome'])
sns.distplot(data_c['Glucose'])


# In[23]:


sns.pairplot(data_c,kind='scatter',hue='Outcome')


# In[11]:


corr_matrix=data_c.corr()


# In[12]:


sns.heatmap(corr_matrix,annot=True,cmap="RdYlGn")


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier




x=data_c.drop(['Outcome'],axis=1,inplace=False)
y=data_c['Outcome']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=3)


# In[14]:


base_pred=np.mean(y_test)
base_pred=round(base_pred,3)
print(base_pred)


# In[15]:


rf=RandomForestClassifier(random_state=10)
model_rf=rf.fit(X_train,y_train)

#rf2=RandomForestRegressor()

model_predict=rf.predict(X_test)

rf_test1=model_rf.score(X_test,y_test)
rf_train1=model_rf.score(X_train,y_train)
print(rf_test1,rf_train1)


# In[16]:


from sklearn.cluster import KMeans
model=KMeans(n_clusters=3,n_jobs=4,random_state=3)
model.fit(x)
centers=model.cluster_centers_
print(centers)


# In[28]:


from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression()
model_lgr=lgr.fit(X_train,y_train)
model_predict_lgr=lgr.predict(X_test)


# In[29]:


lgr_test=model_lgr.score(X_test,y_test)
lgr_train=model_lgr.score(X_train,y_train)
print(lgr_test,lgr_train)


# In[27]:


model_lgr.predict_proba(X_test)



