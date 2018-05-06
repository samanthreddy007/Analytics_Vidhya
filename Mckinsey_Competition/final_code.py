# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 23:59:09 2018

@author: Karra's
"""

import pandas as pd

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
train=pd.read_csv("D:/Mckinsey/train_ajEneEa.csv")

test=pd.read_csv("D:/Mckinsey/test_v2akXPA.csv")

sample_submission=pd.read_csv("D:/Mckinsey/sample_submission_1.csv")


for col in train.columns:
    if train[col].dtype==object:
        train[col].value_counts()
        
train_len=train.shape[0];
test_len=test.shape[0];      
  
full_df=pd.concat([train,test]).reset_index(drop=True);

le = LabelEncoder()
for col in train.columns:
    if full_df[col].dtype=='O' or full_df[col].dtype=='object':
        print(col)
        try:
            full_df[col] = le.fit_transform(full_df[col])
        except TypeError:
            full_df[col] = le.fit_transform(full_df[col].fillna('NAN')) 
            
train=full_df[:train_len];
test=full_df[train_len:];            

i=train['stroke'].value_counts()
df_train = pd.concat([train.loc[train.stroke==0].sample(i[1]), train.loc[train.stroke==1]]).reset_index(drop=True)

predictors=train.columns.difference(['id','stroke'])

#y=train['stroke']

#train.drop(['id','stroke'],axis=1,inplace=True)
#test.drop(['id','stroke'],axis=1,inplace=True)
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(df_train[predictors], df_train['stroke']), verbose=3, random_state=1001 )

# Here we go
 # timing starts from this point for "start_time" variable
random_search.fit(df_train[predictors], df_train['stroke'])

#xgb = xgboost.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.015)





#xgb.fit(df_train[predictors], df_train['stroke'])


preds = random_search.predict_proba(test[predictors])[:,1]

preds[preds>.5]=1
preds[preds<=.5]=0
preds=preds.astype(int)
sample_submission['stroke'] = preds



sample_submission.to_csv('D:/Mckinsey/submission_22.csv', index=False)

