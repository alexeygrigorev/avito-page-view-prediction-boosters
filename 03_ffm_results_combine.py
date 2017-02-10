import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold

# restore the 2-fold split

cv = KFold(len(y), n_folds=2, shuffle=True, random_state=1)
fold0, fold1 = next(iter(cv))


# read FFM predictions

pred0 = pd.read_csv('ffm/pred_0_time.txt', header=None, dtype='float32')
pred0 = pred0[0].values

pred1 = pd.read_csv('ffm/pred_1_time.txt', header=None, dtype='float32')
pred1 = pred1[0].values

pred_test = pd.read_csv('ffm/pred_test_time.txt', header=None, dtype='float32')
pred_test = pred_test[0].values


# save the results

df_train = pd.DataFrame()

df_train['ffm_pred'] = 0
df_train.iloc[fold0, 0] = pred0
df_train.iloc[fold1, 0] = pred1

df_test = pd.DataFrame()
df_test['ffm_pred'] = pred_test

np.save('ffm/ffm_full_pred.npy', df_train.ffm_pred.values)
np.save('ffm/ffm_test_pred.npy', df_test.ffm_pred.values)