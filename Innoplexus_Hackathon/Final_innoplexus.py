# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 22:51:22 2018

@author: Karra's
"""
#73
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

from sklearn.model_selection import KFold

#train=pd.read_csv("D:/Analytics_Vidhya/innoplexus_august/train.csv")
#test=pd.read_csv("D:/Analytics_Vidhya/innoplexus_august/test_nvPHrOx.csv")
submission=pd.read_csv("D:/Analytics_Vidhya/innoplexus_august/sample_submission_poy1UIu.csv")
#html=pd.read_csv("D:/Analytics_Vidhya/innoplexus_august/html_data.csv",low_memory=False)

#html=pd.read_csv("D:/Analytics_Vidhya/innoplexus_august/html_data.csv",low_memory=False)

##train=train.merge(html,how='left',on='Webpage_id')
#test=test.merge(html,how='left',on='Webpage_id')

#train.to_pickle("D:/Analytics_Vidhya/innoplexus_august/train_new.csv")
#test.to_pickle("D:/Analytics_Vidhya/innoplexus_august/test_new.csv")



import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p


loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=' ')



traindata = list(np.array(pd.read_csv('D:/Analytics_Vidhya/innoplexus_august/train.csv'))[:,2])
testdata = list(np.array(pd.read_csv('D:/Analytics_Vidhya/innoplexus_august/test_nvPHrOx.csv'))[:,2])
y = np.array(pd.read_csv('D:/Analytics_Vidhya/innoplexus_august/train.csv'))[:,-1]


X_all = traindata + testdata
lentrain = len(traindata)
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(X_all)
train_word_features = word_vectorizer.transform(traindata)
test_word_features = word_vectorizer.transform(testdata)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(X_all)
train_char_features = char_vectorizer.transform(traindata)
test_char_features = char_vectorizer.transform(testdata)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

classifier = LogisticRegression(C=0.1, solver='sag')

classifier.fit(train_features,y)

LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='sag', tol=0.0001,
          verbose=0, warm_start=False)

pred = classifier.predict(test_features)

submission['Tag']=pred
submission.to_csv('D:/Analytics_Vidhya/innoplexus_august/sol.csv',index=False)