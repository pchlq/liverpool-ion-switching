import numpy as np
import pandas as pd
import datetime
import gc
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold, train_test_split
from sklearn import metrics
from reduce_mem_usage import reduce_mem_usage

data_fold = '../data/Best_FE/'

def read_data():
    nr= None
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

    # train['batch'] = 0
    # for i in range(0, 10):
    #     train.loc[i * 500_000: 500_000 * (i + 1), 'batch'] = i
    
    submission = pd.read_csv('../data/sample_submission.csv', dtype={'time':str})
    
    return train, test, submission

train, test, submission = read_data()
print(f'Train data have {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test data have {test.shape[0]} rows and {test.shape[1]} columns.')

y_train = train['open_channels']
del train['open_channels']
gc.collect()

print(f'  train.shape =', train.shape)
print('y_train.shape =', y_train.shape)
print(f'   test.shape =', test.shape)

oof_test = submission[['time']].copy()
oof_df = pd.read_csv('../data/train.csv', dtype={'time': np.float32}, usecols=['time'])

TOTAL_FOLDS=5
skf = StratifiedKFold(n_splits = TOTAL_FOLDS, shuffle = True, random_state = 100)
# kfold = KFold(n_splits=TOTAL_FOLDS, shuffle=True, random_state=100)


def run_lgb(train, test, y_train, params=None, cv=skf):

    def MacroF1Metric(preds, dtrain):
        labels = dtrain.get_label()
        preds = np.round(np.clip(preds, 0, 10)).astype(int)
        score = metrics.f1_score(labels, preds, average = 'macro')
        return ('MacroF1Metric', score, True)

    def lgb_Metric(preds, dtrain):
        labels = dtrain.get_label()
        num_labels = 11
        preds = preds.reshape(num_labels, len(preds)//num_labels)
        preds = np.argmax(preds, axis=0)
        score = metrics.f1_score(labels, preds, average="macro")
        return ('KaggleMetric', score, True)
    
    oof_pred = np.zeros(len(train))
    y_pred = np.zeros(len(test))
     
    # groups = train['batch']
    for fold, (tr_ind, val_ind) in enumerate(cv.split(train, y_train)):
        print('fold: ', fold)
        x_train, x_val = train.iloc[tr_ind], train.iloc[val_ind]
        y_trainlgb, y_val = y_train.iloc[tr_ind], y_train.iloc[val_ind]
        
        model = lgb.train(params, lgb.Dataset(x_train, y_trainlgb), 2000, lgb.Dataset(x_val, y_val),
                verbose_eval=100, early_stopping_rounds=200, feval=MacroF1Metric)
        
        oof_pred[val_ind] = model.predict(x_val, num_iteration=model.best_iteration) 
        y_pred += model.predict(test) / cv.n_splits

        oof_test[f'open_channels_fold_{fold}'] = model.predict(test, num_iteration=model.best_iteration)
        oof_df[f'open_channels_fold_{fold}'] = model.predict(train, num_iteration=model.best_iteration)

    rmse_score = np.sqrt(metrics.mean_squared_error(y_train, oof_pred))
    
    oof_pred = np.round(np.clip(oof_pred, 0, 10)).astype(int)
    round_y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)
    f1 = metrics.f1_score(y_train, oof_pred, average = 'macro')

    print("save oof")
    oof_test.to_csv('oof_test.csv', index=False)
    oof_df.to_csv('oof_df.csv', index=False)
    
    print(f'Our oof rmse score is {rmse_score}')
    print(f'Our oof macro f1 score is {f1}')
    return round_y_pred, f1

params = {'boosting_type': 'gbdt',
          'metric': 'rmse',
          'objective': 'regression',
        #   'n_jobs': -1,
          'seed': 100,
          'num_leaves': 280,
          'learning_rate': 0.026623466966581126,
          'max_depth': 73,
          'lambda_l1': 2.959759088169741,
          'lambda_l2': 1.331172832164913,
          'bagging_fraction': 0.9655406551472153,
          'bagging_freq': 9,
          'colsample_bytree': 0.6867118652742716,
          'device': 'gpu',
          'gpu_platform_id': 0,
          'gpu_device_id': 0
          }

# run model and predict
model_date = datetime.datetime.today().strftime("%d%m")
round_y_pred, f1 = run_lgb(train, test, y_train, cv=skf, params=params)
submission['open_channels'] = round_y_pred
submission.to_csv(f'subm_lgb_bf_{model_date}_{f1:0.5f}.csv', index = False)







