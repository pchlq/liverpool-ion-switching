import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
import gc
from reduce_mem_usage import reduce_mem_usage
import optuna
from optuna import Trial
import json
import logging
import joblib


def objective(trial: Trial):

    data_fold = '../data/Best_FE/'
    nr = None

    print('loading and preparing the data')
    train1 = pd.read_csv(data_fold + 'final_train1.csv', nrows=nr)
    train1 = reduce_mem_usage(train1)
    train2 = pd.read_csv(data_fold + 'final_train2.csv', nrows=nr)
    train2 = reduce_mem_usage(train2)
    train3 = pd.read_csv(data_fold + 'final_train3.csv', nrows=nr)
    train3 = reduce_mem_usage(train3)
    train = pd.concat([train1, train2, train3], axis = 1)

    target = train['open_channels']
    del train['open_channels'], train1, train2, train3
    gc.collect()



    # train_x, valid_x, train_y, valid_y = train_test_split(train, target, 
    #                                     stratify=target, test_size=0.25)

    folds = 3
    seed  = 100
    shuffle = True
#   CV = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
    CV = StratifiedKFold(n_splits = folds, shuffle = shuffle, random_state = seed)

    y_val_preds = np.zeros(train.shape[0])

#   gc.collect()
#   catFeatures=[xTrain.columns.get_loc(catCol) for catCol in cat_cols]
    models, scores = [], []
    val_score = 0
    for train_idx, val_idx in CV.split(train, target):
        train_data = train.iloc[train_idx,:], target[train_idx]
        val_data = train.iloc[val_idx,:], target[val_idx]
        model, score = xgb_regeressor_tuner(trial, train_data, val_data, num_rounds=100)
        scores.append(score)
    val_score = np.mean(scores)
    return val_score


def xgb_regeressor_tuner(trial,
                        train,
                        val,
                        catFeatures=None,
                        num_rounds=100):
    

    tree_method = ['auto', 'hist','approx','gpu_hist'] # 'hist', 'exact'
    boosting_list = ['gbtree', 'gblinear']
    objective_list_reg = ['reg:linear', 'reg:gamma', 'reg:tweedie']
    metric_list = ['rmse']
    params ={
        'boosting': trial.suggest_categorical('boosting', boosting_list),
        'tree_method': trial.suggest_categorical('tree_method', tree_method),
        'max_depth': trial.suggest_int('max_depth', 2, 80),
        'max_leaves': trial.suggest_int('max_leaves', 10, 300),
        'reg_alpha': trial.suggest_int('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_int('reg_lambda', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
        'gamma': trial.suggest_int('gamma', 0, 5),
        'learning_rate': trial.suggest_loguniform('learning_rate',0.007,0.5),
        'eval_metric': trial.suggest_categorical('eval_metric', metric_list),
        'objective': trial.suggest_categorical('objective', objective_list_reg),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),
        # 'colsample_bynode': trial.suggest_discrete_uniform('colsample_bynode', 0.1, 1, 0.01),
        'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.1, 1, 0.01),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.05),
        'nthread': -1,
        'n_estimators': trial.suggest_int('n_estimators', 10, 1500),
        'num_boost_round': trial.suggest_int('num_boost_round', 0, 1500),
        'eta': trial.suggest_loguniform('eta',0.007,0.5)
        # 'rate_drop': trial.suggest_discrete_uniform('rate_drop', 0.1, 1, 0.01)
        }
    early_stopping=10

    X_train, y_train = train
    X_val, y_val = val
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid  = xgb.DMatrix(X_val, y_val)

    prune_error = 'val-' + metric_list[0]
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, prune_error)

    model = xgb.train(params, dtrain, 
                    # num_boost_round=num_rounds, 
                    evals=[(dtrain, 'train'), (dvalid, 'val')],
                    verbose_eval=True,
                    early_stopping_rounds=early_stopping,
                    # tree_method='gpu_hist',
                    callbacks=[pruning_callback]
                    )
    
    #predictions
    dval = xgb.DMatrix(X_val)
    pred_val = model.predict(dval)

    # model = xgb.XGBRegressor(params,
    #                         tree_method= 'gpu_hist',
    #                         # n_estimators= 10000,
    #                         random_state= 100,
    #                         early_stopping_rounds= early_stopping,
    #                         callbacks=[pruning_callback]
    #                         )

    # eval_set=[(X_val, y_val)]
    # model.fit(X_train, y_train, eval_set=eval_set)
 
    # preds_val = model.predict(X_val)




    pred_val = np.round( np.clip(pred_val, 0, 10) ).astype(int)
           
    score = f1_score(y_val, pred_val, average='macro')
    print('f1_score :', score)
    return model, score

if __name__ == "__main__":
    
    study_name = 'xgbr_070520'
    hours = 4
    MAX_TIME = hours*60*60  # seconds
    N_TRIALS = 20

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("../data/xgb_params/LOGGER.log", mode="w"))
    optuna.logging.enable_propagation() # Propagate logs to the root logger.
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr.

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        study_name=study_name,
        direction="maximize"
        )

    logger.info("Start optimization.")
    study.optimize(objective, n_trials=N_TRIALS, timeout=MAX_TIME)
    df = study.trials_dataframe()
    df.to_csv('../data/xgb_params/trial_df.csv', index=False)

    with open('../data/xgb_params/best_params.json', 'w') as fout:
        json.dump(study.best_params, fout)


    with open('../data/xgb_params/LOGGER.log') as fout:
        assert fout.readline() == "Start optimization.\n"
        assert fout.readline().startswith("Finished trial#0 with value:")

    joblib.dump(study, '../data/xgb_params/study.pkl')
    print('complete successfully!')