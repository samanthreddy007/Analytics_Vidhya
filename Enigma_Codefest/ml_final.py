# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 18:00:13 2018

@author: Karra's
"""

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb



train=pd.read_csv("D:/AV/ml/train_NIR5Yl1.csv")
test=pd.read_csv("D:/AV/ml/test_8i3B3FC.csv")
submission=pd.read_csv("D:/AV/ml/sample_submission_OR5kZa5.csv")


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.Upvotes.values
all_data = pd.concat((train, test)).reset_index(drop=True)

from sklearn.preprocessing import LabelEncoder
cols = ['Tag']
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
    
    

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
all_data["Username_Count"]=getCountVar(all_data, all_data, "Username")
 
train = all_data[:ntrain]
test = all_data[ntrain:]


np.random.seed(13)

def impact_coding(data, feature, target='Upvotes'):
    '''
    In this implementation we get the values and the dictionary as two different steps.
    This is just because initially we were ignoring the dictionary as a result variable.
    
    In this implementation the KFolds use shuffling. If you want reproducibility the cv 
    could be moved to a parameter.
    '''
    n_folds = 10
    n_inner_folds = 5
    impact_coded = pd.Series()
    
    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
    kf = KFold(n_splits=n_folds, shuffle=True)
    oof_mean_cv = pd.DataFrame()
    split = 0
    for infold, oof in kf.split(data[feature]):
            impact_coded_cv = pd.Series()
            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
            inner_split = 0
            inner_oof_mean_cv = pd.DataFrame()
            oof_default_inner_mean = data.iloc[infold][target].mean()
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
                            lambda x: oof_mean[x[feature]]
                                      if x[feature] in oof_mean.index
                                      else oof_default_inner_mean
                            , axis=1))

                # Also populate mapping (this has all group -> mean for all inner CV folds)
                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1
            
            impact_coded = impact_coded.append(data.iloc[oof].apply(
                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                                      if x[feature] in inner_oof_mean_cv.index
                                      else oof_default_mean
                            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean


impact_coding_map = {}
categorical_features=['Tag','Answers']
for f in categorical_features:
    print("Impact coding for {}".format(f))
    train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(train, f)
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    test["impact_encoded_{}".format(f)] = test.apply(lambda x: mapping[x[f]]
                                                                         if x[f] in mapping
                                                                         else default_mean
                                                               , axis=1)


def getDVEncode(compute_df, target_df, var_name, target_var, min_cutoff=1):
	if type(var_name) != type([]):
		var_name = [var_name]
	grouped_df = target_df.groupby(var_name)[target_var].agg(["mean"]).reset_index()
	grouped_df.columns = var_name + ["mean_value"]
	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(target_df[target_var].values), inplace=True)
	return list(merged_df["mean_value"])


def getDVEncodeVar_2(compute_df, target_df, var_name, target_var, min_cutoff=1):
	if type(var_name) != type([]):
		var_name = [var_name]
	grouped_df = target_df.groupby(var_name)[target_var].agg(["count"]).reset_index()
	grouped_df.columns = var_name + ["mean_value"]
	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(target_df[target_var].values), inplace=True)
	return list(merged_df["mean_value"])
'''
def getDVEncodeVar_3(compute_df, target_df, var_name, target_var, min_cutoff=1):
	if type(var_name) != type([]):
		var_name = [var_name]
	grouped_df = target_df.groupby(var_name)[target_var].agg(["nunique"]).reset_index()
	grouped_df.columns = var_name + ["mean_value"]
	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(target_df[target_var].values), inplace=True)
	return list(merged_df["mean_value"])

'''
    
#ulimit = np.percentile(train.Reputation.values, 99)
#train['Reputation'].ix[train['Reputation']>ulimit] = ulimit

train['tag_answer_mean']=getDVEncode(train, train, ['Tag'], 'Answers')
test['tag_answer_mean']=getDVEncode(test, train, ['Tag'], 'Answers')


train['tag_view_mean']=getDVEncode(train, train, ['Tag'], 'Views')
test['tag_view_mean']=getDVEncode(test, train, ['Tag'], 'Views')


train['tag_user_count']=getDVEncodeVar_2(train, train, ['Tag'], 'Username')
test['tag_user_count']=getDVEncodeVar_2(test, train, ['Tag'], 'Username')



train['user_answer_mean']=getDVEncode(train, train, ['Username'], 'Answers')
test['user_answer_mean']=getDVEncode(test, test, ['Username'], 'Answers')


train['user_view_mean']=getDVEncode(train, train, ['Username'], 'Views')
test['user_view_mean']=getDVEncode(test, test, ['Username'], 'Views')


train['user_tag_count']=getDVEncodeVar_2(train, train, ['Username'], 'Tag')
test['user_tag_count']=getDVEncodeVar_2(test, test, ['Username'], 'Tag')


#train['user_reputation']=getDVEncode(train, train, ['Username'], 'Reputation')
#test['user_reputation']=getDVEncode(test, test, ['Username'], 'Reputation')

train['views_answers']=train['Views']*train['Answers']
test['views_answers']=test['Views']*test['Answers']

train['views_per_answers']=train['Views']/(train['Answers']+.0001)
test['views_per_answers']=test['Views']/(test['Answers']+.0001)

#train['view_reputation']=train['Views']*train['Reputation']
#test['view_reputation']=test['Views']*test['Reputation']

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

from sklearn.model_selection import train_test_split

features_to_use=[]          
features_to_use.extend(train.columns)

these_features = [f for f in features_to_use if f not in ['Upvotes','Username','ID']]
''''
fig, ax = plt.subplots()
ax.scatter(x = train['Views'], y = train['Upvotes'])
plt.ylabel('Upvotes', fontsize=13)
plt.xlabel('Views', fontsize=13)
plt.show()
'''
train_new = train.drop(train[(train['Views']>4000000) & (train['Upvotes']>400000)].index)

X_train, X_val, y_tr, y_val = train_test_split(train_new, train_new['Upvotes'], test_size = 0,random_state=99)


model_xgb = xgb.XGBRegressor(random_state=99)
model_gbm = GradientBoostingRegressor(random_state=99)
#model_rf = ExtraTreesRegressor(random_state=99)
#model_et = RandomForestRegressor(random_state=99)

model_xgb.fit(X_train[these_features], y_tr)
model_gbm.fit(X_train[these_features], y_tr)
#model_rf.fit(X_train[these_features], y_tr)
#model_et.fit(X_train[these_features], y_tr)
xgb_test_pred = model_xgb.predict(test[these_features])
gbm_test_pred = model_gbm.predict(test[these_features])
#rf_test_pred = model_rf.predict(test[these_features])
#et_test_pred = model_et.predict(test[these_features])
#xgb_pred = (model_xgb.predict(X_val[these_features]))
#print(rmsle(y_val, xgb_pred))

'''
xgb_val_pred = model_xgb.predict(X_val[these_features])
print(rmsle(y_val, xgb_val_pred))
gbm_val_pred = model_gbm.predict(X_val[these_features])
print(rmsle(y_val, gbm_val_pred))
'''
'''
rf_train_pred = model_rf.predict(train[these_features])
et_train_pred = model_et.predict(train[these_features])

model_xgb = xgb.XGBRegressor()
model_xgb.fit(train[these_features], y_train)
xgb_train_pred = model_xgb.predict(train[these_features])
xgb_pred = (model_xgb.predict(test[these_features]))
print(rmsle(y_train, xgb_train_pred))
'''
#759.615605727
'''
s_val_xgb = np.floor(xgb_val_pred)
s_val_xgb[s_val_xgb<0]=0
print(rmsle(y_val, s_val_xgb))
s_val_gbm = np.floor(gbm_val_pred)
s_val_gbm[s_val_gbm<0]=0
print(rmsle(y_val, s_val_gbm))
'''
s_xgb = np.floor(xgb_test_pred)
s_xgb[s_xgb<0]=0
s_gbm = np.floor(gbm_test_pred)
s_gbm[s_gbm<0]=0
s=(.30*s_xgb)+(.70*s_gbm)

s = np.floor(s)
s[s<0]=0

test['Upvotes']=s 
submission=pd.read_csv("D:/AV/ml/sample_submission_OR5kZa5.csv")
submission.drop(['Upvotes'], axis=1, inplace=True)
sub=submission.merge(test[['ID','Upvotes']],how='left',on=['ID'])
#submission['Upvotes'] = s
#submission['Upvotes'] = ensemble.astype(int32)
sub.to_csv('D:/AV/ml/sol_1.csv',index=False)