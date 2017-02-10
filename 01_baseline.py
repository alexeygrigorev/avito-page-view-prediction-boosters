# coding: utf-8

from collections import Counter

import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVR


# reading & preprocessing the data

df_train = pd.read_csv('data3/train.csv', sep=';')
df_train['target'] = np.log(df_train.item_views + 1)

df_test = pd.read_csv('data3/test.csv', sep=';')

df_train.fillna('NA', inplace=1)
df_test.fillna('NA', inplace=1)


for c in ['param1', 'param2', 'param3']:
    common = set(df_train[c]) & set(df_test[c])
    df_train.loc[~df_train[c].isin(common), c] = 'NA'
    df_test.loc[~df_test[c].isin(common), c] = 'NA'

    cnt = Counter(df_train[c])
    df_train.loc[df_train[c].apply(cnt.get) < 10, c] = 'NA'
    df_test.loc[df_test[c].apply(cnt.get) < 10, c] = 'NA'


# preparing the sparse matrices

tv = CountVectorizer(ngram_range=(1, 3), min_df=10, binary=True, dtype=np.uint8)
X_title = tv.fit_transform(df_train.title)
X_test_title = tv.transform(df_test.title)

ohe_cols = ['owner_type', 'category', 'subcategory', 'param1', 'param2', 'param3', 'region']

records = df_train[ohe_cols].to_dict(orient='records')
dv = DictVectorizer(dtype=np.uint8)
X_oh = dv.fit_transform(records)

records = df_test[ohe_cols].to_dict(orient='records')
X_test_oh = dv.transform(records)


X = sp.hstack([X_title, X_oh], format='csr')
y = df_train.target.values

X_test = sp.hstack([X_test_title, X_test_oh], format='csr')


# training the model

lr = LinearSVR(C=0.05, loss='squared_epsilon_insensitive', dual=False, random_state=1)
lr.fit(X, y)


# submission

y_pred = lr.predict(X_test)

df_sub = pd.DataFrame()
df_sub['id'] = df_test.iloc[:, 0].values
df_sub['item_views'] = np.exp(y_pred) - 1

df_sub.to_csv('submission.csv', sep=';', index=False)