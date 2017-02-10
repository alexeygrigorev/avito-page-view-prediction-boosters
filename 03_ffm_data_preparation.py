# coding: utf-8

from time import time

import pandas as pd
import numpy as np
import scipy.sparse as sp

import feather

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold

from tqdm import tqdm


# reading the data

df_train = feather.read_dataframe('df_train.feather')
df_test = feather.read_dataframe('df_test.feather')


# training the vectorizers 

tv = CountVectorizer(ngram_range=(1, 3), min_df=10, binary=True, dtype=np.uint8)
X_title = tv.fit_transform(df_train.title)
X_test_title = tv.transform(df_test.title)


ohe_cols = ['owner_type', 'category', 'subcategory', 'param1', 'param2', 'param3', 'region']

dv = DictVectorizer(dtype=np.uint8)
X_oh = dv.fit_transform(df_train[ohe_cols].to_dict(orient='records'))
X_test_oh = dv.transform(df_test[ohe_cols].to_dict(orient='records'))


# 2-fold split

cv = KFold(len(y), n_folds=2, shuffle=True, random_state=1)
fold0, fold1 = next(iter(cv))

df_fold0 = df_train.iloc[fold0].reset_index(drop=1)
X_0_oh = X_oh[fold0]
X_0_title = X_title[fold0]

df_fold1 = df_train.iloc[fold1].reset_index(drop=1)
X_1_oh = X_oh[fold1]
X_1_title = X_title[fold1]


# saving the data to FFM format

def gen_fft_str(i, oh, title, df, test=False):
    oh_row = oh.getrow(i)
    oh_row = ' '.join('1:%s:1' % t for t in oh_row.indices)

    title_row = title.getrow(i)
    title_row = ' '.join('2:%s:1' % t for t in title_row.indices)

    ttm = df.time_to_midnight.iloc[i]
    dow = df.dow.iloc[i]

    time_row = '3:%s:1 4:%s:1' % (ttm, dow)

    row = oh_row + ' ' + title_row + ' ' + time_row
    if not test:
        lab = df.target.iloc[i]
        return str(lab) + ' ' + row
    else:
        return '0 ' + row


with open('ffm/ffm0_time.txt', 'w') as f:
    n = len(fold0)
    for i in tqdm(range(n)):
        row = gen_fft_str(i, X_0_oh, X_0_title, df_fold0, test=False)
        f.write(row + '\n')

with open('ffm/ffm1_time.txt', 'w') as f:
    n = len(fold1)
    for i in tqdm(range(n)):
        row = gen_fft_str(i, X_1_oh, X_1_title, df_fold1, test=False)
        f.write(row + '\n')

with open('ffm/ffm_test_time.txt', 'w') as f:
    n = len(df_test)
    for i in tqdm(range(n)):
        row = gen_fft_str(i, X_test_oh, X_test_title, df_test, test=True)
        f.write(row + '\n')
