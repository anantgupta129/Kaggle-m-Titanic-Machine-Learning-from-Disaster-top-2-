#!/usr/bin/env python
# coding: utf-8

# 15/10/2020

# # 1. Importing Libraries

# In[1]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# # 2. Importing Dataset

# In[2]:


# training data
train_dts = pd.read_csv('../input/titanic/train.csv')
train_dts.head()


# In[3]:


# test data
test_dts = pd.read_csv('../input/titanic/test.csv')
test_dts.head()


# ### **2.1 Overview** 
# * `PassengerId` is the unique id of the row and it doesn't have any effect on target
# * `Survived` is the target variable we are trying to predict (**0** or **1**):
#     - **1 = Survived**
#     - **0 = Not Survived**
# * `Pclass` (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has **3** unique values (**1**,  **2 **or **3**):
#     - **1 = Upper Class**
#     - **2 = Middle Class**
#     - **3 = Lower Class**
# * `Name`, `Sex` and `Age` are self-explanatory
# * `SibSp` is the total number of the passengers' siblings and spouse
# * `Parch` is the total number of the passengers' parents and children
# * `Ticket` is the ticket number of the passenger
# * `Fare` is the passenger fare
# * `Cabin` is the cabin number of the passenger
# * `Embarked` is port of embarkation and it is a categorical feature which has **3** unique values (**C**, **Q** or **S**):
#     - **C = Cherbourg**
#     - **Q = Queenstown**
#     - **S = Southampton**

# # 3. Feature Engineering

# Calculating Survival rate of Male and Female on training set

# In[4]:


female = train_dts.loc[train_dts.Sex=='female']['Survived']
print('% of Female survived : {:.3f}'.format((sum(female)/len(female))*100))

male = train_dts.loc[train_dts.Sex=='male']['Survived']
print('% of Male survived : {:.3f}'.format((sum(male)/len(male))*100))


# can be seen clearly that female has much larger probablity of surviving then male

# In[5]:


print('Shape of Training Set : {}'.format(train_dts.shape))
print('Number of training data points : {}\n'.format(len(train_dts)))
print('Shape of Test Set : {}'.format(test_dts.shape))
print('Number of test data points : {}\n'.format(len(test_dts)))
print('Columns : {}'.format(train_dts.columns))
train_dts.info()


# In[6]:


test_dts.info()


# In[7]:


train_dts.describe()


# In[8]:


test_dts.describe()


# ### 3.1 Heatmap

# In[9]:


g = sns.heatmap(train_dts.corr(),annot=True, fmt = ".1f", cmap = "coolwarm")


# ### 3.2 Age Histogram Plot

# In[10]:


age_hist = train_dts.Age.hist()


# from the above histogram it can be observed that most of the passengers were from the age group of 20 - 40

# In[11]:


train_dts.groupby('Pclass').Survived.mean()


# the Passenger Class status also has an influence on the survival chances

# In[12]:


pd.crosstab(index=train_dts['Sex'], columns=train_dts['Pclass'], values=train_dts.Survived, aggfunc='mean')


# also the `gender` in the classes also has an influence as it can be seen that female has **96%** chances and males has **36%** in class 1 only and can seen decreasing with `lower class`

# In[13]:


embarked_hist = train_dts.Embarked.hist()


# ### **3.3 Missing Values**
# As seen from below, some columns have missing values. `display_missing` function shows the count of missing values in every column in both training and test set by using `info()` function in non null count
# 
# * Training set and test set both have missing values in `Age`, `Cabin` and `Embarked` columns. and in `fare` test set 
# 
# Missing values in `Age`, `Embarked` and `Fare` can be filled with descriptive statistical measures but that wouldn't work for `Cabin`.
# 
# above histogram shows that most of the passengers belong to the `S` (Southampton) in `Embarked`, let's fill tha `nan` values by **S** in training set and test set. And the `nan values` in `age` can be filled by takaing `mean` of all age in test and train sets

# In[14]:


#Fill nan values in Embarked with 'S' as it is most frequent value
train_dts['Embarked'] = train_dts['Embarked'].fillna('S')
train_dts['Age'] = train_dts['Age'].fillna(train_dts['Age'].mean())
train_dts['Age'].isnull().sum() 


# the `nan` values in `fare` column can be filled taking median in test set

# In[15]:


test_dts['Fare'] = test_dts['Fare'].fillna(test_dts['Fare'].median())
test_dts['Age'] = test_dts['Age'].fillna(test_dts['Age'].mean())


# creating titles from names of passengers

# In[16]:


title = [i.split(",")[1].split(".")[0].strip() for i in train_dts['Name']]
train_dts['Title'] = pd.Series(title)

title_ = [i.split(",")[1].split(".")[0].strip() for i in test_dts['Name']]
test_dts['Title'] = pd.Series(title_)

train_dts.Title.value_counts()


# `'Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don',  'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Ms', 'Mlle'` these titles appers less in the dataset as their count is 1 or 2 lets remove them by a single lable `rare`

# In[17]:


train_dts["Title"] = train_dts["Title"].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don',  'Dr',
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Ms', 'Mlle'],
                                                'Rare'
                                               )
test_dts["Title"] = test_dts["Title"].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme', 'Ms', 'Mlle'],
                                              'Rare'
                                             )


# In[18]:


plt.figure(figsize=(6,6))
plt.hist(train_dts.Title)
plt.xticks(rotation=45)
plt.show


# In[19]:


mr = train_dts.loc[train_dts['Title']=='Mr'].Survived
miss = train_dts.loc[train_dts['Title']=='Miss'].Survived
mrs = train_dts.loc[train_dts['Title']=='Mrs'].Survived
master = train_dts.loc[train_dts['Title']=='Master'].Survived
rare = train_dts.loc[train_dts['Title']=='Rare'].Survived

print("probablity of Surviving if Mr : {:.2f}".format(sum(mr)/len(mr)))
print("probablity of Surviving if Mrs : {:.2f}".format(sum(mrs)/len(mrs)))
print("probablity of Surviving if Miss : {:.2f}".format(sum(miss)/len(miss)))
print("probablity of Surviving if Master : {:.2f}".format(sum(master)/len(master)))
print("probablity of Surviving if Rare : {:.2f}".format(sum(rare)/len(rare)))


# the surviving probablities of titles `Miss`, `Mrs` and `Master` are higher

# In[20]:


g = sns.catplot(x="Title", y="Survived", data=train_dts, kind='bar').set_ylabels("Survival Probability")


# it can be assumend that the larger families size has difficulties to get on board as they to find all the mambers of families, lets create a new column of Family size

# In[21]:


train_dts['FamilySize'] = train_dts['SibSp'] + train_dts['Parch'] + 1
test_dts['FamilySize'] = test_dts['SibSp'] + test_dts['Parch'] + 1


# In[22]:


g = sns.catplot(data=train_dts, x='FamilySize', y='Survived', kind='point').set_ylabels("Survival Probability")


# above graph verifies that persons with family size of 1 to 4 had more surviving prob and larger families with 8 and 11 has nearly 0 probablity

# * Sigleton : a boolean variable that describes families of size = 1
# * SmallFamily : a boolean variable that describes families of 2 <= size <= 4
# * LargeFamily : a boolean variable that describes families of 5 < size

# In[23]:


# on training set
train_dts['Singleton'] = train_dts['FamilySize'].map(lambda s: 1 if s == 1 else 0)
train_dts['SmallFamily'] = train_dts['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
train_dts['LargeFamily'] = train_dts['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

# on test set
test_dts['Singleton'] = test_dts['FamilySize'].map(lambda s: 1 if s == 1 else 0)
test_dts['SmallFamily'] = test_dts['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
test_dts['LargeFamily'] = test_dts['FamilySize'].map(lambda s: 1 if 5 <= s else 0)


# it can be assumed that passengers with missing values in cabin not had a cabin aat all so we fill it by 0 and passengers with cabin can be filled by 1

# In[24]:


train_dts.loc[train_dts.Cabin.isnull(), 'Cabin'] = 0
train_dts.loc[train_dts.Cabin != 0, 'Cabin'] = 1

test_dts.loc[test_dts.Cabin.isnull(), 'Cabin'] = 0
test_dts.loc[test_dts.Cabin != 0, 'Cabin'] = 1

train_dts['Cabin'] = pd.to_numeric(train_dts['Cabin'])
test_dts['Cabin'] = pd.to_numeric(test_dts['Cabin'])


# In[25]:


train_dts.Ticket.describe()


# ticket has a lot of duplicate values with mixed number and Alphabets lets now filter the ticket feature. 
# We will clean ticket by getting prefix of the ticket number and for tickets with `digits only` will be replaced b `'X'`

# In[26]:


def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = ticket[0]
    if ticket.isdigit():
        return 'X'
    else:
        return ticket[0]
    
train_dts['Ticket'] = train_dts['Ticket'].map(cleanTicket)
test_dts['Ticket'] = test_dts['Ticket'].map(cleanTicket)


# In[27]:


train_dts.Ticket.unique()


# now we are remaning with just 8 unique values 

# In[28]:


train_dts.min()


# In[29]:


train_dts.max()


# # 4. Creating train and test set and label encoding

# In[30]:


X_train = pd.DataFrame.copy(train_dts)
X_test = pd.DataFrame.copy(test_dts)

# label encoding
X_train = pd.get_dummies(X_train, columns=['Sex', 'Embarked', 'Title', "Pclass", 'Ticket'])
X_test = pd.get_dummies(X_test, columns=['Sex', 'Embarked', 'Title', "Pclass", 'Ticket'])
X_train.shape


# now we having 33 features

# In[31]:


# droping columns
X_train.drop(labels=['Name', 'PassengerId', 'Survived'], axis=1, inplace=True)
X_test.drop(labels=['Name', 'PassengerId'], axis=1, inplace=True)


# In[32]:


plt.figure(figsize = (14,14))
g = sns.heatmap(X_train.corr(),annot=True, fmt = ".1f", cmap = "coolwarm")


# In[33]:


y_train = train_dts.Survived


# In[34]:


X_train.info()
X_train.head()


# In[35]:


X_test.info()
X_test.head()


# after the final processing of features we are remaning with 30 features

# # 5. XGBoost Model

# In[36]:


from xgboost import XGBClassifier
xgb_clf = XGBClassifier(n_estimators= 2000,
                        max_depth= 4,
                        min_child_weight= 2,
                        gamma=0.9,                    
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective= 'binary:logistic',
                        nthread= -1,
                        scale_pos_weight=1
                       )

xgb_clf.fit(X_train, y_train)

# testing on train set
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(xgb_clf.predict(X_train), y_train))
print('Accuracy of training')
print(accuracy_score(xgb_clf.predict(X_train), y_train))


# In[37]:


pred = pd.Series(xgb_clf.predict(X_test), name='Survived')
results = pd.concat([test_dts['PassengerId'], pred], axis=1)
results.to_csv("submission.csv", index=False)
results.head(10)


# if you like my work or it helped, an upvote will help me keep me motivated..
