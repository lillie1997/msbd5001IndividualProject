#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data_train = pd.read_csv(r"C:\Users\Lillie\Desktop\msbd5001-fall2019\train.csv",parse_dates=['purchase_date','release_date'])


# In[2]:


data_train.info()


# In[3]:


data_train


# In[4]:


print(data_train.isnull())


# In[5]:


data_train.fillna(method="ffill",inplace=True)


# In[6]:


data_train.describe()


# In[7]:


features = data_train.loc[:, "price"] 
data_train.loc[:, "price"] = (features-features.mean())/features.std()
data_train


# In[8]:


features = data_train.loc[:, "total_positive_reviews"] 
data_train.loc[:, "total_positive_reviews"] = (features-features.mean())/features.std()
data_train


# In[9]:


features = data_train.loc[:, "total_negative_reviews"] 
data_train.loc[:, "total_negative_reviews"] = (features-features.mean())/features.std()
data_train


# In[10]:


dummies = data_train["genres"].str.get_dummies(",") 
data_train = pd.concat([data_train,dummies], axis = 1) 
data_train


# In[11]:


dummies2 = data_train["categories"].str.get_dummies(",") 
data_train = pd.concat([data_train,dummies2], axis = 1) 
data_train


# In[12]:


data_train = data_train.drop(["genres","categories"], axis=1) 


# In[13]:


data_train = data_train.drop(["Animation & Modeling","Audio Production","Design & Illustration","Racing","Sexual Content","Utilities","Valve Anti-Cheat enabled"], axis=1) 


# In[14]:


data_train.describe()


# In[15]:


from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
lbl.fit(data_train["tags"])
data_train["tags"] = lbl.transform(data_train["tags"])
data_train


# In[16]:


lb2 = preprocessing.LabelEncoder()
lb2.fit(data_train["purchase_date"])
data_train["purchase_date"] = lb2.transform(data_train["purchase_date"])
data_train


# In[17]:


lb3 = preprocessing.LabelEncoder()
lb3.fit(data_train["release_date"])
data_train["release_date"] = lb3.transform(data_train["release_date"])
data_train


# In[18]:


data_train.to_csv("preprocessed_data1.csv", index=False)


# In[27]:


from sklearn import datasets
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier 
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost 
all_data = pd.read_csv("preprocessed_data1.csv")
all_data = shuffle(all_data)
X = all_data.drop(["playtime_forever"], axis=1)
y = all_data["playtime_forever"]
xgb = xgboost.XGBRegressor(n_estimator=1660,max_depth=7,min_child_weight=19,gamma=0.1,colsample_bytree=0.88,subsample=0.12,reg_alpha=0.11,reg_lambda=0.21,learning_rate=0.01)
xgb.fit(X, y)


# In[24]:





# In[25]:





# In[28]:


accuracies = cross_val_score(estimator = xgb, X=X, y=y, cv=10, scoring='neg_mean_squared_error')
rmse_xgb = np.sqrt(-accuracies).mean()
print(rmse_xgb)


# In[29]:


submission=pd.read_csv(r"C:\Users\Lillie\Desktop\msbd5001-fall2019\test.csv",parse_dates=['purchase_date','release_date'])


# In[30]:


submission.fillna(method="ffill",inplace=True)
submission.info()


# In[31]:


tfeatures = submission.loc[:, "price"] 
submission.loc[:, "price"] = (tfeatures-tfeatures.mean())/tfeatures.std()
submission
tfeatures1 = submission.loc[:, "total_positive_reviews"] 
submission.loc[:, "total_positive_reviews"] = (tfeatures1-tfeatures1.mean())/tfeatures1.std()
submission
tfeatures2 = submission.loc[:, "total_negative_reviews"] 
submission.loc[:, "total_negative_reviews"] = (tfeatures2-tfeatures2.mean())/tfeatures2.std()
submission


# In[32]:


tdummies = submission["genres"].str.get_dummies(",") 
submission = pd.concat([submission,tdummies], axis = 1) 
submission


# In[33]:


tdummies1 = submission["categories"].str.get_dummies(",") 
tdummies1
submission = pd.concat([submission,tdummies1], axis = 1)
submission


# In[34]:


submission = submission.drop(["genres","categories"], axis=1) 
submission


# In[35]:


labell = preprocessing.LabelEncoder()
labell.fit(submission["tags"])
submission["tags"] = labell.transform(submission["tags"])

label2 = preprocessing.LabelEncoder()
label2.fit(submission["purchase_date"])
submission["purchase_date"] = label2.transform(submission["purchase_date"])

label3 = preprocessing.LabelEncoder()
label3.fit(submission["release_date"])
submission["release_date"] = label3.transform(submission["release_date"])
submission


# In[36]:



ypred = xgb.predict(submission)

print(ypred)


# In[37]:


df=pd.DataFrame()
df["id"]=submission["id"]
#df
df["playtime_forever"]= pd.DataFrame(ypred)
df


# In[38]:


df.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




