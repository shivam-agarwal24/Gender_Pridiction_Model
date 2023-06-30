#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sys, os
from matplotlib import pyplot as plt
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder
from itertools import product
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


voice_df = pd.read_csv('voice.csv')
voice_df.head()


# In[3]:


print("Size of Gender Recognition dataset       :",voice_df.shape)


# In[4]:


voice_df.info()


# In[5]:


voice_df.describe().T


# In[6]:


voice_df.isna().sum()    


# In[7]:


voice_df.isnull().sum()   


# In[8]:


plt.figure(figsize=(9,6))
sns.countplot(x='label', data=voice_df, order=["male", "female"] )


# In[9]:


voice_df['label'].value_counts()           # Prints the count of different classes in 'label'


# In[10]:


# creating instance of labelencoder
label_encode = LabelEncoder()


# In[11]:


# Perform Encoding by coverting 'label' feature into numerical form
voice_df['label'] = label_encode.fit_transform(voice_df['label'])


# In[12]:


voice_df.head()


# In[13]:


voice_df.columns


# In[14]:


print(voice_df["label"].value_counts())


# In[15]:


ax = voice_df['label'].value_counts().plot(kind='pie', figsize=(14,10), autopct='%1.1f%%', labels=["Male","Female"])
ax.axes.get_yaxis().set_visible(False)


# # Model Building

# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[17]:


x = voice_df.iloc[:,:-1]
y = voice_df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[18]:


def gen_metrics(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    print('Training score',model.score(x_train,y_train))
    print('Testing score',model.score(x_test,y_test))
    ypred = model.predict(x_test)
    print('predicted Results\n',ypred)
    cm = confusion_matrix(y_test,ypred)
    print('Confusion Matrix\n',cm)
    print('Classficaition Report\n',classification_report(y_test,ypred))


# # 1) Decision Tree Classifier

# In[19]:


m1 = DecisionTreeClassifier(criterion='gini',max_depth=8,min_samples_split=14)
gen_metrics(m1,x_train,x_test,y_train,y_test)


# # 2) Random Forest Classifer

# In[20]:


m2 = RandomForestClassifier(n_estimators=150,criterion='entropy',max_depth=12)
gen_metrics(m2,x_train,x_test,y_train,y_test)


# # 3) KNN Classifier

# In[21]:


m3 = KNeighborsClassifier(n_neighbors=27)
gen_metrics(m3,x_train,x_test,y_train,y_test)


# ## 4) Logistic Regression

# In[22]:


m4 = LogisticRegression(max_iter=1000)
gen_metrics(m4,x_train,x_test,y_train,y_test)


# # 5)SVM Classifer

# In[23]:


m5 = SVC(kernel='linear',C=1)
gen_metrics(m5,x_train,x_test,y_train,y_test)


# # RANDON FOREST CLASSIFIER GIVES THE BEST ACCURACY OF  0.98

# In[ ]:




