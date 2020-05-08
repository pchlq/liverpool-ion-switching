import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
from datetime import datetime
import os
import random
from collections import Counter, defaultdict
import time
import pickle
import json
import gc
from reduce_mem_usage import reduce_mem_usage


def get_distribution(y_vals):
        y_distr = Counter(y_vals)
        y_vals_sum = sum([x for x in y_distr.values()])
        return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]

def stratified_group_k_fold(X, y, groups, k, seed=100):
    
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


data_fold = '../data/Best_FE/'

def read_data():
    nr=None #100_000
    print('loading and preparing the data')
    train1 = pd.read_csv(data_fold + 'final_train1.csv', nrows=nr)
    train1 = reduce_mem_usage(train1)
    train2 = pd.read_csv(data_fold + 'final_train2.csv', nrows=nr)
    train2 = reduce_mem_usage(train2)
    train3 = pd.read_csv(data_fold + 'final_train3.csv', nrows=nr)
    train3 = reduce_mem_usage(train3)
    
    train = pd.concat([train1, train2, train3], axis = 1)
    del train1, train2, train3
    gc.collect()
    print('train data loaded')
    
    test1 = pd.read_csv(data_fold + 'final_test1.csv', nrows=nr)
    test1 = reduce_mem_usage(test1)
    test2 = pd.read_csv(data_fold + 'final_test2.csv', nrows=nr)
    test2 = reduce_mem_usage(test2)
    test3 = pd.read_csv(data_fold + 'final_test3.csv', nrows=nr)
    test3 = reduce_mem_usage(test3)
    
    test = pd.concat([test1, test2, test3], axis = 1)
    del test1, test2, test3
    gc.collect()
    print('test data loaded')
    
    return train, test

X, X_test = read_data()
data_test = pd.read_csv('../data/cleaned/test.csv', dtype={'time': np.float32, 'signal': np.float32})
sub = data_test[['time']].copy()
oof_df = X[['signal', 'open_channels']].copy()
print(f'X data have {X.shape[0]} rows and {X.shape[1]} columns.')
print(f'X_test data have {X_test.shape[0]} rows and {X_test.shape[1]} columns.')

y = X['open_channels']
del X['open_channels'], data_test
gc.collect()

print(f'  X.shape =', X.shape)
print('y.shape =', y.shape)
print(f'   test.shape =', X_test.shape)


# with open('../data/xgb_params/best_params.json', 'r') as f:
#     params = json.load(f)

# print("params optuna: ", params)

fold = 1
TOTAL_FOLDS=3
# kfold = KFold(n_splits=TOTAL_FOLDS, shuffle=True, random_state=100)
skf = StratifiedKFold(n_splits = TOTAL_FOLDS, shuffle = True, random_state = 100)

for tr_idx, val_idx in skf.split(X, y):
    print(f'====== Fold {fold:0.0f} of {TOTAL_FOLDS} ======')   # stratified_group_k_fold(X, y, groups, k=TOTAL_FOLDS)
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # train_data = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X.columns)
    # valid_data = xgb.DMatrix(data=X_val, label=y_val, feature_names=X.columns)

    # watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
    # model = xgb.train(dtrain=train_data, 
    #                   evals=watchlist,
    #                   early_stopping_rounds=50, 
    #                   verbose_eval=True, params=params)

    # model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist,
    #                 early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params) OLD

    # preds = model.predict(xgb.DMatrix(X_val, feature_names=X.columns))
    # preds = np.round(np.clip(preds, 0, 10)).astype(int)

    # test_preds = model.predict(xgb.DMatrix(X_test, feature_names=X.columns))
    # test_preds = np.round(np.clip(test_preds, 0, 10)).astype(int)

    model = xgb.XGBRegressor(colsample_bytree= 0.67,
                            learning_rate= 0.266,
                            max_depth= 32, 
                            alpha= 4,
                            reg_lambda=4,
                            gamma=1,
                            min_child_weight=2,
                            eta=0.01,
                            colsample_bylevel=0.86,
                            subsample=0.6,
                            max_leaves=283,
                            tree_method= 'gpu_hist',
                            n_estimators= 1271,
                            metric= ['rmse'],
                            random_state= 100,
                            early_stopping_rounds= 50
                            )

    eval_set=[(X_val, y_val)]
    model.fit(X_tr, y_tr, eval_set=eval_set)
 
    preds = model.predict(X_val)
    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    test_preds = model.predict(X_test)
    test_preds = np.round(np.clip(test_preds, 0, 10)).astype(int)

    oof_df.loc[oof_df.iloc[val_idx].index, 'oof'] = preds
    sub[f'open_channels_fold{fold}'] = test_preds

    f1 = f1_score(oof_df.loc[oof_df.iloc[val_idx].index]['open_channels'],
                oof_df.loc[oof_df.iloc[val_idx].index]['oof'],
                average='macro')
    rmse = np.sqrt(mean_squared_error(oof_df.loc[oof_df.index.isin(val_idx)]['open_channels'],
                                    oof_df.loc[oof_df.index.isin(val_idx)]['oof']))

    print(f'Fold {fold} - validation f1: {f1:0.5f}')
    print(f'Fold {fold} - validation rmse: {rmse:0.5f}')

    fold += 1

oof_f1 = f1_score(oof_df['open_channels'],
                  oof_df['oof'],
                  average='macro')
oof_rmse = np.sqrt(mean_squared_error(oof_df['open_channels'],
                                      oof_df['oof']))

print(f'{oof_f1:0.5f}')
print('saving result...')
######### saving result
s_cols = [s for s in sub.columns if 'open_channels' in s]

sub['open_channels'] = sub[s_cols].median(axis=1).astype(int)
# sub.to_csv(f'./pred_xgb_{oof_f1:0.6}.csv', index=False)
sub[['time', 'open_channels']].to_csv(f'./subm_xgb_{oof_f1:0.6f}.csv',
                                      index=False,
                                      float_format='%0.4f')

# oof_df.to_csv(f'./oof_{MODEL}_{oof_f1:0.6}.csv', index=False)