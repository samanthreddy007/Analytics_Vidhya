

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
color = sns.color_palette()

#%matplotlib inline

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None

train=pd.read_csv("D:/AV/nlp/train_2kmZucJ.csv")
test=pd.read_csv("D:/AV/nlp/test_oJQbWVk.csv")
submission=pd.read_csv("D:/AV/nlp/sample_submission_LnhVWA4.csv")

#f=test['tweet'].contains('suck')
#s=(test['tweet'].str.find('suck')!=-1).astype(int)
#train['there']=(train['tweet'].str.find('$&@*#')!=-1).astype(int)
#test['there']=(test['tweet'].str.find('$&@*#')!=-1).astype(int)
train["num_words_upper"] = train["tweet"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper"] = test["tweet"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
from nltk.corpus import stopwords
from bs4 import BeautifulSoup 

y=train.label.values
import string

for l in string.ascii_letters:
    train['lcnt_'+l] = train['tweet'].map(lambda s: str(s).count(l))
    test['lcnt_'+l] = test['tweet'].map(lambda s: str(s).count(l))
#unique digits count
for d in string.digits:
    train['dcnt_'+str(d)] = train['tweet'].map(lambda s: str(s).count(d))
    test['dcnt_'+str(d)] = test['tweet'].map(lambda s: str(s).count(d))


signs=['$','&','@','*','#']
for l in signs:
    train['lcnt_'+l] = train['tweet'].map(lambda s: str(s).count(l))
    test['lcnt_'+l] = test['tweet'].map(lambda s: str(s).count(l))
    
train['sign_all']=train['lcnt_$']+train['lcnt_&']+train['lcnt_@']+train['lcnt_*']+train['lcnt_#']
test['sign_all']=test['lcnt_$']+test['lcnt_&']+test['lcnt_@']+test['lcnt_*']+test['lcnt_#']

#train['lcnt_https'] = train['tweet'].map(lambda s: str(s).count('https'))
#test['lcnt_https'] = test['tweet'].map(lambda s: str(s).count('https'))    


#train['tweet'] = train['tweet'].apply(stem_sentences)
#test['tweet'] = test['tweet'].apply(stem_sentences)
## Number of words in the text ##
train["num_words"] = train["tweet"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["tweet"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train["num_unique_words"] = train["tweet"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["tweet"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train["num_chars"] = train["tweet"].apply(lambda x: len(str(x)))
test["num_chars"] = test["tweet"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train["num_stopwords"] = train["tweet"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["num_stopwords"] = test["tweet"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train["num_punctuations"] =train['tweet'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["num_punctuations"] =test['tweet'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##

## Number of title case words in the text ##
train["num_words_title"] = train["tweet"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["num_words_title"] = test["tweet"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
#train["mean_word_len"] = train["tweet"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#test["mean_word_len"] = test["tweet"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

#train["mean_word_std"] = train["tweet"].apply(lambda x: np.std([len(w) for w in str(x).split()]))
#test["mean_word_std"] = test["tweet"].apply(lambda x: np.std([len(w) for w in str(x).split()]))

#train['tweet']=train['tweet'].map(review_to_words)
#test['tweet']=test['tweet'].map(review_to_words)
 

train['tweet'] = train['tweet'].str.replace('[^a-zA-Z0-9]', ' ').replace('  ', ' ').str.lower()
test['tweet'] = test['tweet'].str.replace('[^a-zA-Z0-9]', ' ').replace('  ', ' ').str.lower()

train['lcnt'] = train['tweet'].map(len)
test['lcnt'] = test['tweet'].map(len)

train['tweet'] = train['tweet'].map(lambda x: str(x).strip())
test['tweet'] = test['tweet'].map(lambda x: str(x).strip())

import re
train['words'] = train['tweet'].map(lambda x: re.sub("[^\w]", " ",  str(x)).split())
test['words'] = test['tweet'].map(lambda x: re.sub("[^\w]", " ",  str(x)).split())
positive_count = 0
negative_count = 0
def get_positive_words():
    positive_words = []
    with open("D:/AV/nlp/positive-words.txt") as f:
        for line in f:
            str = line[:-1]
            if len(str) > 0 and ";" not in str:
                positive_words.append(str)
    return positive_words


def get_negative_words():
    negative_words = []
    with open("D:/AV/nlp/negative-words.txt") as f:
        for line in f:
            str = line[:-1]
            if len(str) > 0 and ";" not in str:
                negative_words.append(str)
    return negative_words
def read_words_from_file(filename):
    input_str = ""
    with open(filename) as f:
        input_str += "".join(f.readlines()).replace("\n", " ")
    words = input_str.lower().split()
    return words
positive_words = get_positive_words()
negative_words = get_negative_words()
def x(t):
    positive_count = 0
    for word in t:
        if word in positive_words:
            positive_count += 1 
    return positive_count        
 #       elif word in negative_words:
  #          negative_count += 1
def y(t):
    negative_count = 0
    for word in t:
        if word in negative_words:
            negative_count += 1
           # print(word) 
    return negative_count 

train['pos_words']=train['words'].map(x)
train['neg_words']=train['words'].map(y)
test['pos_words']=test['words'].map(x)
test['neg_words']=test['words'].map(y)

#train['neg_words']=train['neg_words']/(train['pos_words']+.0001)
#test['neg_words']=test['neg_words']/(test['pos_words']+.0001)
#wordList = re.sub("[^\w]", " ",  mystr).split()
train['lcnt2'] = train['tweet'].map(len)
test['lcnt2'] = test['tweet'].map(len)



cols_to_drop = ['id', 'tweet','words']
train_X = train.drop(cols_to_drop+['label'], axis=1)
test_X = test.drop(cols_to_drop, axis=1)
train_y=train.label
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'binary:logistic'
   #param['eta'] = 0.1
    #param['max_depth'] = 3
    #param['silent'] = 1
    #param['num_class'] = 2
    param['eval_metric'] = "auc"
    #param['min_child_weight'] = child
    #param['subsample'] = 0.8
    #param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model



import lightgbm as lgb


kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0], 1])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y.reshape(len(val_index),1)
    cv_scores.append(metrics.f1_score(val_y, np.round(pred_val_y).astype(int),average='weighted'))
    print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5


### Plot the important variables ###
fig, ax = plt.subplots(figsize=(12,12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit(train['tweet'].values.tolist() + test['tweet'].values.tolist())
train_tfidf = tfidf_vec.transform(train['tweet'].values.tolist())
test_tfidf = tfidf_vec.transform(test['tweet'].values.tolist())


def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2, model



cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0],1 ])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y.reshape(len(val_index),1)
    cv_scores.append(metrics.f1_score(val_y, pred_val_y,average='weighted'))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5

tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit(train['tweet'].values.tolist() + test['tweet'].values.tolist())
train_tfidf = tfidf_vec.transform(train['tweet'].values.tolist())
test_tfidf = tfidf_vec.transform(test['tweet'].values.tolist())


cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0], 1])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y.reshape(len(val_index),1)
    cv_scores.append(metrics.f1_score(val_y, pred_val_y,average='weighted'))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5

# add the predictions as new features #
train["predictions"] = pred_train[:,0]
test["predictions"] = pred_full_test


### Fit transform the tfidf vectorizer ###
tfidf_vec = CountVectorizer(ngram_range=(1,7), analyzer='char')
tfidf_vec.fit(train['tweet'].values.tolist() + test['tweet'].values.tolist())
train_tfidf = tfidf_vec.transform(train['tweet'].values.tolist())
test_tfidf = tfidf_vec.transform(test['tweet'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0], 1])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y.reshape(len(val_index),1)
    cv_scores.append(metrics.f1_score(val_y, pred_val_y,average='weighted'))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

# add the predictions as new features #
train["predictions_2"] = pred_train[:,0]

test["predictions_2"] = pred_full_test

from sklearn.pipeline import make_union
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=30000)
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=30000)
vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=2)
vectorizer.fit(train['tweet'].values.tolist() + test['tweet'].values.tolist())
train_tfidf = vectorizer.transform(train['tweet'].values.tolist())
test_tfidf = vectorizer.transform(test['tweet'].values.tolist())

cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0], 1])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y.reshape(len(val_index),1)
    cv_scores.append((metrics.f1_score(val_y, pred_val_y,average='weighted')))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.
     
# add the predictions as new features #
train["predictions_3"] = pred_train[:,0]
test["predictions_3"] = pred_full_test



cols_to_drop = ['id', 'tweet','words']
train_X = train.drop(cols_to_drop+['label'], axis=1)
test_X = test.drop(cols_to_drop, axis=1)

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train.shape[0],1])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7)
    #pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_X)
    
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y.reshape(len(val_index),1)
    cv_scores.append(metrics.f1_score(val_y, np.round(pred_val_y).astype(int),average='weighted'))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5


fig, ax = plt.subplots(figsize=(12,12))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

y=np.round(pred_full_test).astype(int)
#y[e]=1

submission['label']=y
submission.to_csv('D:/AV/nlp/sol_1.csv',index=False)