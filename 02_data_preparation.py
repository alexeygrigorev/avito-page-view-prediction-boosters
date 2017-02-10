# coding: utf-8

from collections import Counter

import pandas as pd
import numpy as np
import scipy.sparse as sp

import feather

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


# time processing

df_train.start_time = pd.to_datetime(df_train.start_time)
df_test.start_time = pd.to_datetime(df_test.start_time)

til_0 = df_train.start_time.dt.ceil(freq='d') - df_train.start_time
til_0 = til_0.values / (1000000000 * 60 * 30)
til_0 = til_0.astype('uint32')

df_train['time_to_midnight'] = til_0
df_train['dow'] = df_train_full.start_time.dt.dayofweek


til_0 = df_test.start_time.dt.ceil(freq='d') - df_test.start_time
til_0 = til_0.values / (1000000000 * 60 * 30)
til_0 = til_0.astype('uint32')

df_test['time_to_midnight'] = til_0
df_test['dow'] = df_test.start_time.dt.dayofweek


# saving the results 

feather.write_dataframe(df_train, 'df_train.feather')
feather.write_dataframe(df_test, 'df_test.feather')