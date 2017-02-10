# coding: utf-8

from time import time

import pandas as pd
import numpy as np
import scipy.sparse as sp
import feather

from tqdm import tqdm

import xgboost as xgb

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold

from sklearn.svm import LinearSVR
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_squared_error

def calc_rmse(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return np.sqrt(mse)



df_train_full = feather.read_dataframe('df_train.feather')
df_test = feather.read_dataframe('df_test.feather')


# Title + OH feature matrices

tv = CountVectorizer(ngram_range=(1, 3), min_df=10, binary=True, dtype=np.uint8)
X_title = tv.fit_transform(df_train_full.title)
X_test_title = tv.transform(df_test.title)


ohe_cols = ['owner_type', 'category', 'subcategory', 'param1', 'param2', 'param3', 'region']

dv = DictVectorizer(dtype=np.uint8)
X_oh = dv.fit_transform(df_train_full[ohe_cols].to_dict(orient='records'))
X_test_oh = dv.transform(df_test[ohe_cols].to_dict(orient='records'))

y = df_train_full.target.values


cv = KFold(len(y), n_folds=5, shuffle=True, random_state=1)


#
# simple SVM models
# 


# title model

title_pred = np.zeros_like(y)

C = 0.1

scores = []

for train, val in tqdm(cv):
    lr = LinearSVR(C=C, random_state=1, loss='squared_epsilon_insensitive', dual=False) 
    lr.fit(X_title[train], y[train])
    y_pred = lr.predict(X_title[val])
    title_pred[val] = y_pred

    rmsle = calc_rmse(y[val], y_pred)
    scores.append(rmsle)

print('title model, rmsle=%.4f' % np.mean(scores))


lr_title = LinearSVR(C=C, random_state=1, loss='squared_epsilon_insensitive', dual=False) 
lr_title.fit(X_title, y)

title_pred_test = lr_title.predict(X_test_title)


# OH model

oh_pred = np.zeros_like(y)

C = 0.05
scores = []

for train, val in tqdm(cv):
    lr = LinearSVR(C=C, random_state=1, loss='squared_epsilon_insensitive', dual=False) 
    lr.fit(X_oh[train], y[train])
    y_pred = lr.predict(X_oh[val])
    oh_pred[val] = y_pred

    rmsle = calc_rmse(y[val], y_pred)
    scores.append(rmsle)

print('oh model, rmsle=%.4f' % np.mean(scores))

lr_oh = LinearSVR(C=C, random_state=1, loss='squared_epsilon_insensitive', dual=False) 
lr_oh.fit(X_oh, y)

oh_pred_test = lr_oh.predict(X_test_oh)


# both OH and titme

X_oh_title = sp.hstack([X_title, X_oh], format='csr')
X_oh_title_test = sp.hstack([X_test_title, X_test_oh], format='csr')


oh_title_pred = np.zeros_like(y)

C = 0.05
scores = []

for train, val in tqdm(cv):
    lr = LinearSVR(C=C, random_state=1, loss='squared_epsilon_insensitive', dual=False) 
    lr.fit(X_oh_title[train], y[train])
    y_pred = lr.predict(X_oh_title[val])
    oh_title_pred[val] = y_pred

    rmsle = calc_rmse(y[val], y_pred)
    scores.append(rmsle)

print('oh+title model, rmsle=%.4f' % np.mean(scores))


lr_oh_title = LinearSVR(C=C, random_state=1, loss='squared_epsilon_insensitive', dual=False) 
lr_oh_title.fit(X_oh_title, y)

oh_title_pred_test = lr_oh_title.predict(X_oh_title_test)


#
# ET on SVD + NMF
#

# OH SVD

oh_svd = TruncatedSVD(n_components=30, random_state=1)
X_oh_svd = oh_svd.fit_transform(X_oh)
X_test_oh_svd = oh_svd.transform(X_test_oh)


et_params = dict(
    n_estimators=100,
    criterion='mse',
    max_depth=30,
    min_samples_split=6,
    min_samples_leaf=6,
    max_features=4, 
    bootstrap=False, 
    n_jobs=-1,
    random_state=1
)


et_oh_svd_pred = np.zeros_like(y)
scores = []

for train, val in tqdm(cv):
    et = ExtraTreesRegressor(**et_params)
    et.fit(X_oh_svd[train], y[train])
    y_pred = et.predict(X_oh_svd[val])
    et_oh_svd_pred[val] = y_pred

    rmsle = calc_rmse(y[val], y_pred)
    scores.append(rmsle)

print('et on svd(oh) model, rmsle=%.4f' % np.mean(scores))


et = ExtraTreesRegressor(**et_params)
et.fit(X_oh_svd, y)
et_oh_svd_pred_test = et.predict(X_test_oh_svd)


# Title+OH SVD

t_svd = TruncatedSVD(n_components=10, random_state=1)
X_t_svd = t_svd.fit_transform(X_title)
X_test_t_svd = t_svd.transform(X_test_title)

X_all_svd = np.hstack([X_oh_svd, X_t_svd])
X_test_all_svd = np.hstack([X_test_oh_svd, X_test_t_svd])


et_params = dict(
    n_estimators=100,
    criterion='mse',
    max_depth=30,
    min_samples_split=6,
    min_samples_leaf=6,
    max_features=15, 
    bootstrap=False, 
    n_jobs=-1,
    random_state=1
)


all_svd_pred = np.zeros_like(y)
scores = []

for train, val in tqdm(cv):
    et = ExtraTreesRegressor(**et_params)
    et.fit(X_all_svd[train], y[train])
    y_pred = et.predict(X_all_svd[val])
    all_svd_pred[val] = y_pred

    rmsle = calc_rmse(y[val], y_pred)
    scores.append(rmsle)

print('et on svd(oh + title) model, rmsle=%.4f' % np.mean(scores))


et = ExtraTreesRegressor(**et_params)
et.fit(X_all_svd, y)
all_svd_pred_test = et.predict(X_test_all_svd)



# title+OH NMF

nmf = NMF(n_components=50, random_state=1, alpha=0.1, l1_ratio=1.0)
X_nmf = nmf.fit_transform(X_oh_title.astype('float32'))
X_test_nmf = nmf.transform(X_oh_title_test.astype('float32'))


et_params = dict(
    n_estimators=100,
    criterion='mse',
    max_depth=40,
    min_samples_split=6,
    min_samples_leaf=6,
    max_features=15,
    bootstrap=False, 
    n_jobs=-1,
    random_state=1
)


et_nfm_pred = np.zeros_like(y)

for train, val in tqdm(cv):
    et = ExtraTreesRegressor(**et_params)
    et.fit(X_nmf[train], y[train])
    y_pred = et.predict(X_nmf[val])
    et_nfm_pred[val] = y_pred

    rmsle = calc_rmse(y[val], y_pred)
    scores.append(rmsle)

print('et on nmf(oh + title) model, rmsle=%.4f' % np.mean(scores))

et = ExtraTreesRegressor(**et_params)
et.fit(X_nmf, y)
et_nfm_pred_test = et.predict(X_test_nmf)


# top NMF components as stacking features

nmf_fin = NMF(n_components=5, random_state=1, alpha=0.1, l1_ratio=1.0)
X_nmf_fin = nmf_fin.fit_transform(X_oh_title.astype('float32'))
X_test_nmf_fin = nmf_fin.transform(X_oh_title_test.astype('float32'))


#
# Stacking
# 

# putting everything together


df_train_stack = pd.DataFrame()
df_train_stack['oh'] = oh_pred
df_train_stack['title'] = title_pred
df_train_stack['oh_title'] = oh_title_pred
df_train_stack['et_oh'] = et_oh_svd_pred
df_train_stack['et_oh_title'] = all_svd_pred
df_train_stack['et_nmf'] = et_nfm_pred
df_train_stack['ffm'] = np.load('ffm/ffm_full_pred.npy')

df_train_stack['price'] = df_train_full['price'].values
df_train_stack['time_to_midnight'] = df_train_full['time_to_midnight'].values
df_train_stack['dow'] = df_train_full['dow'].values

for i in range(5):
    df_train_stack['nmf_%d' % i] = X_nmf_fin[:, i]


df_test_stack = pd.DataFrame()
df_test_stack['oh'] = oh_pred_test
df_test_stack['title'] = title_pred_test
df_test_stack['oh_title'] = oh_title_pred_test
df_test_stack['et_oh'] = et_oh_svd_pred_test
df_test_stack['et_oh_title'] = all_svd_pred_test
df_test_stack['et_nmf'] = et_nfm_pred_test
df_test_stack['ffm'] = np.load('ffm/ffm_test_pred.npy')

df_test_stack['price'] = df_test['price'].values
df_test_stack['time_to_midnight'] = df_test['time_to_midnight'].values
df_test_stack['dow'] = df_test['dow'].values

for i in range(5):
    df_test_stack['nmf_%d' % i] = X_test_nmf_fin[:, i]


columns = list(df_train_stack.columns)
X_s = df_train_stack[columns].values
X_test_s = df_test_stack[columns].values


# training XGB

xgb_pars = {
    'eta': 0.1,
    'gamma': 0,
    'max_depth': 10,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 1,
    'lambda': 1,
    'alpha': 0,
    'tree_method': 'approx',
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'nthread': 8,
    'seed': 42,
    'silent': 1
}

n_estimators = 120

dfull = xgb.DMatrix(X_s, label=y, feature_names=columns, missing=np.nan)
watchlist = [(dfull, 'train')]

model = xgb.train(xgb_pars, dfull, num_boost_round=n_estimators, verbose_eval=10,
                  evals=watchlist)

dtest = xgb.DMatrix(X_test_s, feature_names=columns, missing=np.nan)
y_pred = model.predict(dtest)


#
# creating a submission
#

df_sub = pd.DataFrame()
df_sub['id'] = df_test.iloc[:, 0].values
df_sub['item_views'] = np.exp(y_pred) - 1

df_sub.to_csv('submission.csv', sep=';', index=False)
