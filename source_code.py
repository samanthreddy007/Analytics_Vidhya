# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:29:28 2018

@author: Karra's
"""

import pandas as pd
import numpy as np
import seaborn as sns

train=pd.read_csv("D:/Analytics_Vidhya_Hackathon/train.csv")
test=pd.read_csv("D:/Analytics_Vidhya_Hackathon/test.csv")
submission=pd.read_csv("D:/Analytics_Vidhya_Hackathon/sample_submission.csv")
campaign_data=pd.read_csv("D:/Analytics_Vidhya_Hackathon/campaign_data.csv")



train=train[train['campaign_id']!=43]
train_len=train.shape[0]
test_len=test.shape[0]
#full=pd.concat(train,test)

final=pd.merge(left=train,right=campaign_data,how='left', on='campaign_id')



#final['send_date'] =  pd.to_datetime(final['send_date'])

final['user_open_cnt']=final.groupby('user_id')['is_open'].transform('sum')
final['user_click_cnt']=final.groupby('user_id')['is_click'].transform('sum')


final['campaign_open_cnt']=final.groupby('campaign_id')['is_open'].transform('sum')
final['campaign_click_cnt']=final.groupby('campaign_id')['is_click'].transform('sum')

final['communication_type']=final['communication_type'].factorize()[0]

def f(t):
    if t<24:
        return 0
    elif 24<t<67:
        return 1
    elif 67<t<104:
        return 2
    else :
        return 3
def f_2(t):
    if t<19:
        return 0
    elif 19<t<61:
        return 1
    elif 61<t<100:
        return 2
    else :
        return 3
final['total_links']=final['total_links'].apply(lambda x:f(x) )
final['no_of_internal_links']=final['no_of_internal_links'].apply(lambda x:f_2(x) )
final['user_campaign']=final.groupby('user_id')['user_id'].transform('count')



final_test=pd.merge(left=test,right=campaign_data,how='left', on='campaign_id')

final_test=pd.merge(left=final_test,right=final[['campaign_id','campaign_open_cnt','campaign_open_cnt']],how='left',on='campaign_id')

#final['send_date'] =  pd.to_datetime(final['send_date'])

#final_test['user_open_cnt']=final_test.groupby('user_id')['is_open'].transform('sum')
#final_test['user_click_cnt']=final_test.groupby('user_id')['is_click'].transform('sum')


#final_test['campaign_open_cnt']=final_test.groupby('campaign_id')['is_open'].transform('sum')
#final_test['campaign_click_cnt']=final_test.groupby('campaign_id')['is_click'].transform('sum')

final_test['communication_type']=final_test['communication_type'].factorize()[0]

def f(t):
    if t<24:
        return 0
    elif 24<t<67:
        return 1
    elif 67<t<104:
        return 2
    else :
        return 3
def f_2(t):
    if t<19:
        return 0
    elif 19<t<61:
        return 1
    elif 61<t<100:
        return 2
    else :
        return 3
final_test['total_links']=final_test['total_links'].apply(lambda x:f(x) )
final_test['no_of_internal_links']=final_test['no_of_internal_links'].apply(lambda x:f_2(x) )
final_test['user_campaign']=final_test.groupby('user_id')['user_id'].transform('count')


x=final.drop_duplicates(subset='user_id')
final_test_new=pd.merge(left=final_test,right=x[['user_id','user_open_cnt', 'user_click_cnt']],how='left',on='user_id')
final_test_new['user_open_cnt'].isnull().sum()


final_test_new['user_open_cnt']=final_test_new['user_open_cnt'].fillna(0)
final_test_new['user_click_cnt']=final_test_new['user_click_cnt'].fillna(0)


train['send_Date']=pd.to_datetime(train['send_date'])


s=train.sort_values('send_date')
s=s[s['is_click']==1]
t=s.groupby('user_id')['campaign_id'].apply(list)

s['prev_campaign']=s['user_id'].apply(lambda x: t[x][0] if len(t[x])>=2  else 0)

s['prev_campaign']= np.where(s['prev_campaign']==s['campaign_id'], 0, s['prev_campaign'])

s=s[s['prev_campaign']>0]

final=pd.merge(left=final,right=s[['user_id','campaign_id','prev_campaign']],how='left',on=['user_id','campaign_id'])
final['prev_campaign']=final['prev_campaign'].fillna(0)
final_test_new=pd.merge(left=final_test_new,right=s[['user_id','campaign_id','prev_campaign']],how='left',on=['user_id','campaign_id'])
final_test_new['prev_campaign']=final_test_new['prev_campaign'].fillna(0)


final['send_Date']=pd.to_datetime(final['send_date'])
final_test_new['send_Date']=pd.to_datetime(final_test_new['send_date'])

final.loc[final['send_Date'].dt.weekday<5,'week_day']=1
final.loc[final['send_Date'].dt.weekday>=5,'week_day']=0

final_test_new.loc[final_test_new['send_Date'].dt.weekday<5,'week_day']=1
final_test_new.loc[final_test_new['send_Date'].dt.weekday>=5,'week_day']=0

def classifier(final):
        if (final['send_time_hour'] > 0 and final['send_time_hour'] < 6): 
            return 0;
        elif (final['send_time_hour'] > 6 and final['send_time_hour'] < 12): 
            return 1;
        elif (final['send_time_hour'] > 12 and final['send_time_hour'] < 18): 
            return 2;
        else:
            return 3;
            

final['send_time_hour']=final['send_Date'].dt.hour           
final["time_of_day"] = final.apply(classifier, axis=1) 

final_test_new['send_time_hour']=final_test_new['send_Date'].dt.hour           
final_test_new["time_of_day"] = final_test_new.apply(classifier, axis=1) 

final['user_open_rate']=final.groupby('user_id')['is_open'].transform('mean')
final['user_click_rate']=final.groupby('user_id')['is_click'].transform('mean')

final['click_by_open_rate'] = final['user_click_rate']/(final['user_open_rate'] +0.000000001)

r=final.drop_duplicates(subset='user_id', keep='first')
final_test_new=pd.merge(left=final_test_new,right=r[['user_id','user_open_rate','user_click_rate','click_by_open_rate']],how='left',on=['user_id'])
final_test_new['user_open_rate']=final_test_new['user_open_rate'].fillna(0)
final_test_new['user_click_rate']=final_test_new['user_click_rate'].fillna(0)
final_test_new['click_by_open_rate']=final_test_new['click_by_open_rate'].fillna(0)


final = pd.concat([final.loc[final.is_click==0].sample(12780), final.loc[final.is_click==1]]).reset_index(drop=True)

y=final['is_click']
final.drop(['id', 'user_id','send_Date','send_date', 'is_open','is_click','email_body', 'subject','send_time_hour', 'email_url', 'campaign_open_cnt',
       'campaign_click_cnt', 'user_campaign'],axis=1,inplace=True)
final_test_new.drop(['id', 'user_id','send_Date', 'send_date','email_body', 'subject', 'send_time_hour','email_url', 'campaign_open_cnt',
        'user_campaign'],axis=1,inplace=True)

from sklearn.linear_model import LogisticRegression

log_params={}

log_params['random_state']=99
log_params['class_weight']={0:1,1:4}
log_model = LogisticRegression(**log_params)    

log_model.fit(final,y)
predictions=log_model.predict(final_test_new)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(final, y)
Y_pred1 = logreg.predict(final_test_new)
logreg.score(final, y)


d_params={}
d_params['class_weight']={0:1,1:4}
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(**d_params)
decision_tree.fit(final, y)
Y_pred2 = decision_tree.predict(final_test_new)
decision_tree.score(final, y)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier() 
knn.fit(final,y)
Y_pred3=knn.predict(final_test_new)
knn.score(final, y)

predictions[predictions>=.5]=1
predictions[predictions<.5]=0
predictions=predictions.astype(int)
submission['is_click']=predictions

submission.to_csv("D:/hackerearth/sample_submission_40.csv",index=False)