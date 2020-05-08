import pandas as pd
import numpy as np
import pickle
import gc
from reduce_mem_usage import reduce_mem_usage

train = pd.read_csv('../data/train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
test = pd.read_csv('../data/test.csv', dtype={'time': np.float32, 'signal': np.float32})

train['train'] = True
test['train'] = False
tt = pd.concat([train, test], sort=False).reset_index(drop=True)
tt['train'] = tt['train'].astype('bool')

print("Preprocessing...")

tt = tt.sort_values(by=['time']).reset_index(drop=True)
tt.index = ((tt.time * 10_000) - 1).values
tt['batch'] = tt.index // 50_000
tt['batch_index'] = tt.index - (tt.batch * 50_000)

tt['batch_slices'] = tt['batch_index'] // 5_000
tt['batch_slices2'] = tt.apply(lambda r: '_'.join(
    [str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

tt['batch_slices1'] = tt['batch_index'] // 10_000
tt['batch_slices3'] = tt.apply(lambda r: '_'.join(
    [str(r['batch']).zfill(3), str(r['batch_slices1']).zfill(3)]), axis=1)

# 50_000 Batch Features
tt['signal_batch_min'] = tt.groupby('batch')['signal'].transform('min')
tt['signal_batch_max'] = tt.groupby('batch')['signal'].transform('max')
tt['signal_batch_std'] = tt.groupby('batch')['signal'].transform('std')
tt['signal_batch_mean'] = tt.groupby('batch')['signal'].transform('mean')
tt['mean_abs_chg_batch'] = tt.groupby(['batch'])['signal'].transform(
    lambda x: np.mean(np.abs(np.diff(x))))
tt['abs_max_batch'] = tt.groupby(
    ['batch'])['signal'].transform(lambda x: np.max(np.abs(x)))
tt['abs_min_batch'] = tt.groupby(
    ['batch'])['signal'].transform(lambda x: np.min(np.abs(x)))

tt['range_batch'] = tt['signal_batch_max'] - tt['signal_batch_min']
tt['maxtomin_batch'] = tt['signal_batch_max'] / tt['signal_batch_min']
tt['abs_avg_batch'] = (tt['abs_min_batch'] + tt['abs_max_batch']) / 2

# 5_000 Batch Features
tt['signal_batch_5k_min'] = tt.groupby(
    'batch_slices2')['signal'].transform('min')
tt['signal_batch_5k_max'] = tt.groupby(
    'batch_slices2')['signal'].transform('max')
tt['signal_batch_5k_std'] = tt.groupby(
    'batch_slices2')['signal'].transform('std')
tt['signal_batch_5k_mean'] = tt.groupby(
    'batch_slices2')['signal'].transform('mean')
tt['mean_abs_chg_batch_5k'] = tt.groupby(['batch_slices2'])[
    'signal'].transform(lambda x: np.mean(np.abs(np.diff(x))))
tt['abs_max_batch_5k'] = tt.groupby(['batch_slices2'])[
    'signal'].transform(lambda x: np.max(np.abs(x)))
tt['abs_min_batch_5k'] = tt.groupby(['batch_slices2'])[
    'signal'].transform(lambda x: np.min(np.abs(x)))

tt['range_batch_5k'] = tt['signal_batch_5k_max'] - tt['signal_batch_5k_min']
tt['maxtomin_batch_5k'] = tt['signal_batch_5k_max'] / tt['signal_batch_5k_min']
tt['abs_avg_batch_5k'] = (tt['abs_min_batch_5k'] + tt['abs_max_batch_5k']) / 2

# 10_000 Batch Features
tt['signal_batch_10k_min'] = tt.groupby(
    'batch_slices3')['signal'].transform('min')
tt['signal_batch_10k_max'] = tt.groupby(
    'batch_slices3')['signal'].transform('max')
tt['signal_batch_10k_std'] = tt.groupby(
    'batch_slices3')['signal'].transform('std')
tt['signal_batch_10k_mean'] = tt.groupby(
    'batch_slices3')['signal'].transform('mean')
tt['mean_abs_chg_batch_10k'] = tt.groupby(['batch_slices3'])[
    'signal'].transform(lambda x: np.mean(np.abs(np.diff(x))))
tt['abs_max_batch_10k'] = tt.groupby(['batch_slices3'])[
    'signal'].transform(lambda x: np.max(np.abs(x)))
tt['abs_min_batch_10k'] = tt.groupby(['batch_slices3'])[
    'signal'].transform(lambda x: np.min(np.abs(x)))

tt['range_batch_10k'] = tt['signal_batch_10k_max'] - tt['signal_batch_10k_min']
tt['maxtomin_batch_10k'] = tt['signal_batch_10k_max'] / tt['signal_batch_10k_min']
tt['abs_avg_batch_10k'] = (tt['abs_min_batch_10k'] + tt['abs_max_batch_10k']) / 2

# add shifts
tt['signal_shift+1'] = tt.groupby(['batch']).shift(1)['signal']
tt['signal_shift-1'] = tt.groupby(['batch']).shift(-1)['signal']
tt['signal_shift+2'] = tt.groupby(['batch']).shift(2)['signal']
tt['signal_shift-2'] = tt.groupby(['batch']).shift(-2)['signal']
tt['signal_shift+3'] = tt.groupby(['batch']).shift(3)['signal']
tt['signal_shift-3'] = tt.groupby(['batch']).shift(-3)['signal']

for c in ['signal_batch_min', 'signal_batch_max',
          'signal_batch_std', 'signal_batch_mean',
          'mean_abs_chg_batch', 'abs_max_batch',
          'abs_min_batch',
          'range_batch', 'maxtomin_batch', 'abs_avg_batch',
          'signal_shift+1', 'signal_shift-1',
          'signal_batch_5k_min', 'signal_batch_5k_max',
          'signal_batch_5k_std',
          'signal_batch_5k_mean', 'mean_abs_chg_batch_5k',
          'abs_max_batch_5k', 'abs_min_batch_5k',
          'range_batch_5k', 'maxtomin_batch_5k',
          'abs_avg_batch_5k','signal_shift+2','signal_shift-2'] +\
          ['signal_batch_10k_min', 'signal_batch_10k_max',
          'signal_batch_10k_std',
          'signal_batch_10k_mean', 'mean_abs_chg_batch_10k',
          'abs_max_batch_10k', 'abs_min_batch_10k',
          'range_batch_10k', 'maxtomin_batch_10k',
          'abs_avg_batch_10k','signal_shift+3','signal_shift-3']:
    tt[f'{c}_msignal'] = tt[c] - tt['signal']


tt = reduce_mem_usage(tt)
# tt.to_csv("../data/tt.csv", index=False)
pickle.dump(tt, open('../data/tt_10k.pkl', 'wb'))