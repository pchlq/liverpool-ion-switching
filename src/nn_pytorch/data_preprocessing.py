import pandas as pd
import numpy as np
import gc
import config
from sklearn.model_selection import GroupKFold

DATA_DIR = "/home/pchlq/workspace/liverpool-ion-switching/data/"


def read_data():
    train = pd.read_csv(DATA_DIR + 'cleaned/train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv(DATA_DIR + 'cleaned/test.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv(DATA_DIR + 'sample_submission.csv', dtype={'time': np.float32})
    
    Y_train_proba = np.load(DATA_DIR + "Y_train_proba.npy")
    Y_test_proba = np.load(DATA_DIR + "Y_test_proba.npy")
    
    for i in range(11):
        train[f"proba_{i}"] = Y_train_proba[:, i]
        test[f"proba_{i}"] = Y_test_proba[:, i]

    return train, test, sub


def create_signal_mod(train):
    left = 3_641_000
    right = 3_829_000
    thresh_dict = {
        3: [0.1, 2.0],
        2: [-1.1, 0.7],
        1: [-2.3, -0.6],
        0: [-3.8, -2],
    }
    train['batch'] = train.index // 500_000
    
    train['signal_mod'] = train['signal'].values
    for ch in train[train['batch']==7]['open_channels'].unique():
        idxs_noisy = (train['open_channels']==ch) & (left<train.index) & (train.index<right)
        idxs_not_noisy = (train['open_channels']==ch) & ~idxs_noisy
        mean = train[idxs_not_noisy]['signal'].mean()

        idxs_outlier = idxs_noisy & (thresh_dict[ch][1]<train['signal'].values)
        train['signal_mod'][idxs_outlier]  = mean
        idxs_outlier = idxs_noisy & (train['signal'].values<thresh_dict[ch][0])
        train['signal_mod'][idxs_outlier]  = mean
    train['signal'] = train['signal_mod'].values
    train.drop(['batch', 'signal_mod'], axis=1, inplace=True)
    
    return train


def denois_train(df):
    left = 3_642_000
    right = 3_825_000
    low = -3.8
    upper = 2
    train = df.copy()
    nois_ind = train.loc[left:right, :].index
    ind_to_nan = train.iloc[nois_ind][(train.signal>upper) | (train.signal<low)].index
    
    tmp = train[train.index.isin(ind_to_nan)].copy()
    MAX_PROBA_COL = tmp.filter(regex='proba').values.argmax(1)
    for i in range(len(tmp)):
        if tmp.loc[tmp.index[i], 'open_channels'] != MAX_PROBA_COL[i]:
            true_col_idx = tmp.loc[tmp.index[i], 'open_channels']
            pred_cod_idx = MAX_PROBA_COL[i]
            # swapping probas
            tmp.loc[tmp.index[i], f'proba_{true_col_idx}'], tmp.loc[tmp.index[i], f'proba_{pred_cod_idx}'] = \
            tmp.loc[tmp.index[i], f'proba_{pred_cod_idx}'], tmp.loc[tmp.index[i], f'proba_{true_col_idx}']
    
    train[train.index.isin(ind_to_nan)] = tmp.values
    train['open_channels'] = train['open_channels'].astype('int16')
    del tmp; gc.collect()
    
    return train


# create batches of 4000 observations
def batching(df, batch_size):
    df['group'] = df.groupby(df.index // batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    return train, test

# get lead and lags features
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size = batch_size)
    # create leads and lags (1, 2, 3 making them 6 features)
    df = lag_with_pct_change(df, [1, 2, 3])
    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
    return df

# fillna with the mean and select features for training
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis=0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features


def split(GROUP_BATCH_SIZE=config.GROUP_BATCH_SIZE, SPLITS=config.SPLITS):
    print('Reading Data Started...')
    train, test, sample_submission = read_data()
    train, test = normalize(train, test)
    print('Reading and Normalizing Data Completed')
    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train, batch_size=GROUP_BATCH_SIZE)
    test = run_feat_engineering(test, batch_size=GROUP_BATCH_SIZE)
    train, test, features = feature_selection(train, test)
    # print(train.head())
    print('Feature Engineering Completed...')

    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=SPLITS)
    splits = [x for x in kf.split(train, train[target], group)]
    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])
        new_splits.append(new_split)
    target_cols = ['open_channels']
    # print(train.head(), train.shape)
    train_tr = np.array(list(train.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[features].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))
    print(train.shape, test.shape, train_tr.shape)
    return train, test, train_tr, new_splits
