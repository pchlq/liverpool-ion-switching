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
import gc
from reduce_mem_usage import reduce_mem_usage


TARGET = "open_channels"


def get_distribution(y_vals):
    y_distr = Counter(y_vals)
    y_vals_sum = sum([x for x in y_distr.values()])
    return [f"{y_distr[i] / y_vals_sum:.2%}" for i in range(np.max(y_vals) + 1)]


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
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
            )
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


FEATURES = [
    "signal",
    "signal_batch_min",
    "signal_batch_max",
    "signal_batch_std",
    "signal_batch_mean",
    "mean_abs_chg_batch",
    #'abs_max_batch',
    #'abs_min_batch',
    #'abs_avg_batch',
    "range_batch",
    "maxtomin_batch",
    "signal_batch_5k_min",
    "signal_batch_5k_max",
    "signal_batch_5k_std",
    "signal_batch_5k_mean",
    "mean_abs_chg_batch_5k",
    "abs_max_batch_5k",
    "abs_min_batch_5k",
    "range_batch_5k",
    "maxtomin_batch_5k",
    "abs_avg_batch_5k",
    "signal_shift+1",
    "signal_shift-1",
    # 'signal_batch_min_msignal',
    "signal_batch_max_msignal",
    "signal_batch_std_msignal",
    # 'signal_batch_mean_msignal',
    "mean_abs_chg_batch_msignal",
    "abs_max_batch_msignal",
    "abs_min_batch_msignal",
    "range_batch_msignal",
    "maxtomin_batch_msignal",
    "abs_avg_batch_msignal",
    "signal_shift+1_msignal",
    "signal_shift-1_msignal",
    "signal_batch_5k_min_msignal",
    "signal_batch_5k_max_msignal",
    "signal_batch_5k_std_msignal",
    "signal_batch_5k_mean_msignal",
    "mean_abs_chg_batch_5k_msignal",
    "abs_max_batch_5k_msignal",
    "abs_min_batch_5k_msignal",
    #'range_batch_5k_msignal',
    "maxtomin_batch_5k_msignal",
    "abs_avg_batch_5k_msignal",
    "signal_shift+2",
    "signal_shift-2",
]  # +\
# ['signal_batch_10k_min',
# 'signal_batch_10k_max',
# 'signal_batch_10k_std',
# 'signal_batch_10k_mean',
# 'mean_abs_chg_batch_10k',
# 'abs_max_batch_10k',
# 'abs_min_batch_10k',
# 'range_batch_10k',
# 'maxtomin_batch_10k',
# 'abs_avg_batch_10k',
# 'signal_batch_10k_min_msignal',
# 'signal_batch_10k_max_msignal',
# 'signal_batch_10k_std_msignal',
# 'signal_batch_10k_mean_msignal',
# 'mean_abs_chg_batch_10k_msignal',
# 'abs_max_batch_10k_msignal',
# 'abs_min_batch_10k_msignal',
# #'range_batch_10k_msignal',
# 'maxtomin_batch_10k_msignal',
# 'abs_avg_batch_10k_msignal',
# 'signal_shift+3',
# 'signal_shift-3']

# print('....: FEATURE LIST :....')
# print([f for f in FEATURES])
#####################################################################
tt = pd.read_csv("../data/tt.csv")  # .head(50_000)
# tt = pickle.load(open('../data/tt_10k.pkl', 'rb'))
tt.head()
print("training->...")

# Training..
tt.fillna(0, inplace=True)
tt = reduce_mem_usage(tt)
tt["train"] = tt["train"].astype("bool")
train = tt.query("train").copy()
test = tt.query("not train").copy()
train["open_channels"] = train["open_channels"].astype(int)
X = train[FEATURES]
X_test = test[FEATURES]
y = train[TARGET].values
sub = test[["time"]].copy()
groups = train["batch"]

print("X shape before: ", X.shape)
# X['avg_cols'] = X.values.mean(axis=1)
# X_test['avg_cols'] = X_test.values.mean(axis=1)


oof_df = train[["signal", "open_channels"]].copy()


params = {
    "objective": "reg:linear",
    "colsample_bytree": 0.3,
    "learning_rate": 0.1,
    "max_depth": 5,
    "alpha": 10,
    "tree_method": "gpu_hist",
    "n_estimators": 100,
    "eval_metric": "rmse",
    "random_state": 100,
    "early_stopping_rounds": 50,
}


fold = 1
TOTAL_FOLDS = 3
# kfold = KFold(n_splits=TOTAL_FOLDS, shuffle=True, random_state=100)
skf = StratifiedKFold(n_splits=TOTAL_FOLDS, shuffle=True, random_state=100)

for tr_idx, val_idx in skf.split(X, y):
    print(
        f"====== Fold {fold:0.0f} of {TOTAL_FOLDS} ======"
    )  # stratified_group_k_fold(X, y, groups, k=TOTAL_FOLDS)
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = xgb.XGBRegressor(
        colsample_bytree=0.3,
        learning_rate=0.04,  # 0.15=0.937
        max_depth=6,
        alpha=3,
        tree_method="gpu_hist",
        n_estimators=10000,
        metric=["rmse"],
        random_state=100,
        early_stopping_rounds=50,
    )

    eval_set = [(X_val, y_val)]
    model.fit(X_tr, y_tr, eval_set=eval_set)

    preds = model.predict(X_val)
    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    test_preds = model.predict(X_test)
    test_preds = np.round(np.clip(test_preds, 0, 10)).astype(int)

    oof_df.loc[oof_df.iloc[val_idx].index, "oof"] = preds
    sub[f"open_channels_fold{fold}"] = test_preds

    f1 = f1_score(
        oof_df.loc[oof_df.iloc[val_idx].index]["open_channels"],
        oof_df.loc[oof_df.iloc[val_idx].index]["oof"],
        average="macro",
    )
    rmse = np.sqrt(
        mean_squared_error(
            oof_df.loc[oof_df.index.isin(val_idx)]["open_channels"],
            oof_df.loc[oof_df.index.isin(val_idx)]["oof"],
        )
    )

    print(f"Fold {fold} - validation f1: {f1:0.5f}")
    print(f"Fold {fold} - validation rmse: {rmse:0.5f}")

    fold += 1

oof_f1 = f1_score(oof_df["open_channels"], oof_df["oof"], average="macro")
oof_rmse = np.sqrt(mean_squared_error(oof_df["open_channels"], oof_df["oof"]))

print(f"{oof_f1:0.5f}")
print("saving result...")
# saving result
s_cols = [s for s in sub.columns if "open_channels" in s]

sub["open_channels"] = sub[s_cols].median(axis=1).astype(int)
# sub.to_csv(f'./pred_xgb_{oof_f1:0.6}.csv', index=False)
sub[["time", "open_channels"]].to_csv(
    f"./subm_xgb_{oof_f1:0.6f}.csv", index=False, float_format="%0.4f"
)

# oof_df.to_csv(f'./oof_{MODEL}_{oof_f1:0.6}.csv', index=False)
