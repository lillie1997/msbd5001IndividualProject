#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np
data_train = pd.read_csv(r"C:\Users\Lillie\Desktop\msbd5001-fall2019\train.csv",parse_dates=['purchase_date','release_date'])


# In[88]:


data_train.info()


# In[89]:


data_train


# In[90]:


print(data_train.isnull())


# In[91]:


data_train.fillna(method="ffill",inplace=True)


# In[92]:


data_train.describe()


# In[93]:


features = data_train.loc[:, "price"] 
data_train.loc[:, "price"] = (features-features.mean())/features.std()
data_train


# In[94]:


features = data_train.loc[:, "total_positive_reviews"] 
data_train.loc[:, "total_positive_reviews"] = (features-features.mean())/features.std()
data_train


# In[95]:


features = data_train.loc[:, "total_negative_reviews"] 
data_train.loc[:, "total_negative_reviews"] = (features-features.mean())/features.std()
data_train


# In[96]:


dummies = data_train["genres"].str.get_dummies(",") 
data_train = pd.concat([data_train,dummies], axis = 1) 
data_train


# In[97]:


dummies2 = data_train["categories"].str.get_dummies(",") 
data_train = pd.concat([data_train,dummies2], axis = 1) 
data_train


# In[98]:


data_train = data_train.drop(["genres","categories"], axis=1) 


# In[99]:


data_train = data_train.drop(["Animation & Modeling","Audio Production","Design & Illustration","Racing","Sexual Content","Utilities","Valve Anti-Cheat enabled"], axis=1) 


# In[100]:


data_train.describe()


# In[101]:


from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
lbl.fit(data_train["tags"])
data_train["tags"] = lbl.transform(data_train["tags"])
data_train


# In[102]:


lb2 = preprocessing.LabelEncoder()
lb2.fit(data_train["purchase_date"])
data_train["purchase_date"] = lb2.transform(data_train["purchase_date"])
data_train


# In[103]:


lb3 = preprocessing.LabelEncoder()
lb3.fit(data_train["release_date"])
data_train["release_date"] = lb3.transform(data_train["release_date"])
data_train


# In[104]:


data_train.to_csv("preprocessed_data1.csv", index=False)


# In[105]:


from sklearn import datasets
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')
#import xgboost 
all_data = pd.read_csv("preprocessed_data1.csv")
all_data = shuffle(all_data)
X = all_data.drop(["playtime_forever"], axis=1)
y = all_data["playtime_forever"]
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=420,min_samples_split=19,max_depth=2,max_features='log2',min_samples_leaf=13,random_state=0)
rf.fit(X, y)


# In[106]:


submission=pd.read_csv(r"C:\Users\Lillie\Desktop\msbd5001-fall2019\test.csv",parse_dates=['purchase_date','release_date'])


# In[107]:


submission.fillna(method="ffill",inplace=True)
submission.info()


# In[108]:


tfeatures = submission.loc[:, "price"] 
submission.loc[:, "price"] = (tfeatures-tfeatures.mean())/tfeatures.std()
submission
tfeatures1 = submission.loc[:, "total_positive_reviews"] 
submission.loc[:, "total_positive_reviews"] = (tfeatures1-tfeatures1.mean())/tfeatures1.std()
submission
tfeatures2 = submission.loc[:, "total_negative_reviews"] 
submission.loc[:, "total_negative_reviews"] = (tfeatures2-tfeatures2.mean())/tfeatures2.std()
submission


# In[109]:


tdummies = submission["genres"].str.get_dummies(",") 
submission = pd.concat([submission,tdummies], axis = 1) 
submission


# In[110]:


tdummies1 = submission["categories"].str.get_dummies(",") 
tdummies1
submission = pd.concat([submission,tdummies1], axis = 1)
submission


# In[111]:


submission = submission.drop(["genres","categories"], axis=1) 
submission


# In[112]:


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


# In[116]:


ypred = rf.predict(submission)

print(ypred)


# In[117]:


df=pd.DataFrame()
df["id"]=submission["id"]
#df
df["playtime_forever"]= pd.DataFrame(ypred)
df


# In[118]:


df.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




