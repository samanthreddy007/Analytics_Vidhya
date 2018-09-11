# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:47:55 2018

@author: Karra's
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

train=pd.read_csv('D:/innoplex/train.csv')
test=pd.read_csv('D:/innoplex/test.csv')
submission=pd.read_csv('D:/innoplex/sample_submission_eSUXEfp.csv')
import csv
information_test=pd.read_csv('D:/innoplex/information_test.csv',error_bad_lines=False,delimiter='\t')
information_train=pd.read_csv('D:/innoplex/information_train.csv',error_bad_lines=False,delimiter='\t')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer().fit_transform(new.abstract)
from sklearn.metrics.pairwise import linear_kernel



information_test['related']=0    
information_test['related']=information_test['related'].astype('object')    
s=[17,10,4,9,11,1,15,7,19]
u=0
h=0
e=0    
for i in s:
    print(i)
    new=information_test[information_test['set']==i]
    tfidf = TfidfVectorizer().fit_transform(new.abstract)
    t=new.shape[0]
    print(t)
    if(i==19):
        print(u)
    e=u    
    for y in range(t):
        cosine_similarities = linear_kernel(tfidf[y:y+1], tfidf).flatten()
        related_docs_indices = cosine_similarities.argsort()[-2:-5:-1]
        r=[]
        if(i==19):
            print(u)
        for j in related_docs_indices:
            
            r.append(information_test['pmid'].iloc[e+j])
        if(i==7):
            print(u)
        information_test['related'].iloc[u]=r
        if(i==7):
            print(r)
        u=u+1
        
submission_1=submission.merge(information_test[['pmid','related']],on='pmid',how='left')        
submission_1.drop(['ref_list'],inplace=True,axis=1)
submission_1.rename(columns={'related':'ref_list'},inplace=True)         
#submission['pmid'].merge(information_test['related'],on='pmid',how=left)        
#submission['ref_list']=information_test['related']        
submission_1.to_csv('D:/innoplex/submission_2.csv',index=False)