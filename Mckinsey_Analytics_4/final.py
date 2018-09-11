# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:40:12 2018

@author: Karra's
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 11:10:58 2018

@author: Karra's
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt

train=pd.read_csv('D:/Analytics_Vidhya/Mckinsey/dataset/train.csv')
test=pd.read_csv('D:/Analytics_Vidhya/Mckinsey/dataset/test.csv')
submission=pd.read_csv('D:/Analytics_Vidhya/Mckinsey/dataset/sample_submission.csv')


sum_wpos=train['renewal'].sum()
sum_wneg=train.shape[0]-train['renewal'].sum()


def getCountVar(compute_df, count_df, var_name):
        grouped_df = count_df.groupby(var_name)
        count_dict = {}
        for name, group in grouped_df:
                count_dict[name] = group.shape[0]

        count_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
                count_list.append(count_dict.get(name, 0))
        return count_list

def getPurchaseVar(compute_df, purchase_df, var_name):
        grouped_df = purchase_df.groupby(var_name)
      #  min_dict = {}
       # max_dict = {}
        mean_dict = {}
        for name, group in grouped_df:
         #       min_dict[name] = min(np.array(group["Purchase"]))
          #      max_dict[name] = max(np.array(group["Purchase"]))
                mean_dict[name] = np.mean(np.array(group["renewal"]))

        #min_list = []
        #max_list = []
        mean_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
         #       min_list.append(min_dict.get(name,0))
          #      max_list.append(max_dict.get(name,0))
                mean_list.append(mean_dict.get(name,0))

       # return min_list, max_list, mean_list
        return mean_list

train["source_Count"] = getCountVar(train, train, "sourcing_channel")
test["source_Count"] = getCountVar(test, train, "sourcing_channel")

train["source_mean"] = getPurchaseVar(train, train, "sourcing_channel")
test["source_mean"] = getPurchaseVar(test, train, "sourcing_channel")
    
train['severity']=(5*train['Count_3-6_months_late'])+(9*train['Count_6-12_months_late'])+(15*train['Count_more_than_12_months_late'])
test['severity']=(5*test['Count_3-6_months_late'])+(9*test['Count_6-12_months_late'])+(15*test['Count_more_than_12_months_late'])

train.loc[train['application_underwriting_score']>99,'insured']=1
train.loc[train['application_underwriting_score']<99,'insured']=0
test.loc[train['application_underwriting_score']>99,'insured']=1
test.loc[train['application_underwriting_score']<99,'insured']=0


train['null_count'] = train.apply(lambda x: train.shape[1]-x.count(), axis=1)
test['null_count'] = test.apply(lambda x: test.shape[1]-x.count(), axis=1)
#ulimit = np.percentile(train.Income.values, 99)
#train['Income'].ix[train['Income']>ulimit]=ulimit

#ulimit = np.percentile(test.Income.values, 99)
#test['Income'].ix[test['Income']>ulimit]=ulimit

train_no=train.shape[0]
y=train['renewal']
test['renewal']=-1
full=pd.concat([train,test])
full['age_in_years']=(full['age_in_days']/365).astype(np.int32)
full['Income_per_premium']=full['Income']/full['premium']
#full['Income']=np.log(full['Income']+1)

le=LabelEncoder()
full['sourcing_channel']=le.fit_transform(full['sourcing_channel'])
full['residence_area_type']=le.fit_transform(full['residence_area_type'])

#full=full.drop(['id','age_in_days','renewal'],axis=1)

train_new=full[:train_no]
test_new=full[train_no:]

age_dict = {'15-25':1, '25-35':2, '35-45':3, '45-55':4, '55-65':5, '65-120':6}
premium_dict = {'0-5':1, '10-15':3, '20-25':5, '30-35':7, '40-45':9, '50-55':11,'5-10':2, '15-20':4, '25-30':6, '35-40':8, '45-50':10, '55-60':12}

train_new['no_of_premiums_paid_bin'] = pd.cut(train_new['no_of_premiums_paid'],[0,5,10,15,20, 25,30,35,40,45,50,55,60], labels=['0-5', '5-10', '10-15','15-20','20-25', '25-30','30-35', '35-40','40-45', '45-50','50-55','55-60'])
test_new['no_of_premiums_paid_bin'] = pd.cut(test_new['no_of_premiums_paid'], [0,5,10,15,20, 25,30,35,40,45,50,55,60], labels=['0-5', '5-10', '10-15','15-20','20-25', '25-30','30-35', '35-40','40-45', '45-50','50-55','55-60'])




train_new['no_of_premiums_paid_bin'] = train_new['no_of_premiums_paid_bin'].apply(lambda x: premium_dict[x])
test_new['no_of_premiums_paid_bin'] = test_new['no_of_premiums_paid_bin'].apply(lambda x: premium_dict[x])


train_new["no_of_premiums_paid_Count"] = getCountVar(train_new, train_new, "no_of_premiums_paid_bin")
test_new["no_of_premiums_paid_Count"] = getCountVar(test_new, train_new, "no_of_premiums_paid_bin")

train_new["no_of_premiums_paid_mean"] = getPurchaseVar(train_new, train_new, "no_of_premiums_paid_bin")
test_new["no_of_premiums_paid_mean"] = getPurchaseVar(test_new, train_new, "no_of_premiums_paid_bin")


train_new['age_in_years_bin'] = pd.cut(train_new['age_in_years'], [15, 25,35,45,55,65,120], labels=['15-25', '25-35', '35-45','45-55', '55-65', '65-120'])
test_new['age_in_years_bin'] = pd.cut(test_new['age_in_years'], [15, 25,35,45,55,65,120], labels=['15-25', '25-35', '35-45','45-55', '55-65', '65-120'])


train_new['age_in_years_bin'] = train_new['age_in_years_bin'].apply(lambda x: age_dict[x])
test_new['age_in_years_bin'] = test_new['age_in_years_bin'].apply(lambda x: age_dict[x])


train_new["age_in_years_Count"] = getCountVar(train_new, train_new, "age_in_years_bin")
test_new["age_in_years_Count"] = getCountVar(test_new, train_new, "age_in_years_bin")

train_new["age_in_years_mean"] = getPurchaseVar(train_new, train_new, "age_in_years_bin")
test_new["age_in_years_mean"] = getPurchaseVar(test_new, train_new, "age_in_years_bin")

train_new=train_new.drop(['id','age_in_days','renewal'],axis=1)
test_new=test_new.drop(['id','age_in_days','renewal'],axis=1)


#train_new.fillna(-999, inplace=True)
#test_new.fillna(-999, inplace=True)
    
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.03
params['scale_pos_weight'] = sum_wneg/sum_wpos
params['silent'] = True
params['max_depth'] = 5
params['subsample'] = 0.9
params['min_child_weight'] = 10
params['colsample_bytree'] = 0.9
params['colsample_bylevel'] = 0.9
params['seed']=2

kfold = 4
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.2, random_state=9487)
X=train_new.values
y=y.values
submission['renewal']=0
score=0
def feval(preds,dm):
    labels=dm.get_label()
    auc=metrics.roc_auc_score(labels,preds)
    return [('my_auc',auc)]

for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test_new.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
    # and the custom metric (maximize=True tells xgb that higher metric is better)
    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=feval, maximize=True, verbose_eval=100)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    # Predict on our test data
    p_valid=mdl.predict(d_valid, ntree_limit=mdl.best_ntree_limit)
    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
    submission['renewal'] += p_test/kfold
    
    score +=mdl.best_score/kfold


print(score)
g=submission['renewal'].values
#f=-5*np.log(1-(g/20))
#h=-400*np.log(1-(f/10))
h=(-400/3)*np.log(10/test['premium']*g)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#f=-5*np.log(1-(g/20))
#h=-400*np.log(1-(f/10))
submission['incentives']=(h/3).astype(int)
#submission['incentives']=(h/100)*test['premium'].astype(int) 
submission.to_csv("D:/Analytics_Vidhya/Mckinsey/dataset/sol.csv",index=False) 

#xgb.plot_importance(mdl)

#confusion_matrix(y_actu, y_pred)