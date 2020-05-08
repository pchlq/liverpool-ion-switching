import pandas as pd
import numpy as np
import argparse
from reduce_mem_usage import reduce_mem_usage


def generate_features(data: pd.DataFrame,
                      batch_sizes: list,
                      window_sizes: list) -> pd.DataFrame:
    """
    Generate features for https://www.kaggle.com/c/liverpool-ion-switching

    Generate various aggregations over the data.

    Args:
        window_sizes: window sizes for rolling features
        batch_sizes: batch sizes for which features are aggregated
        data: original dataframe

    Returns:
        dataframe with generated features
    """
    for batch_size in batch_sizes:
        data['batch'] = ((data['time'] * 10_000) - 1) // batch_size
        data['batch_index'] = ((data['time'] * 10_000) - 1) - (data['batch'] * batch_size)
        data['batch_slices'] = data['batch_index'] // (batch_size / 10)
        data['batch_slices2'] = data['batch'].astype(str).str.zfill(3) + '_' + data['batch_slices'].astype(
            str).str.zfill(3)
        data['batch_slices3'] = data['batch_index'] // (batch_size / 5)
        data['batch_slices4'] = data['batch'].astype(str).str.zfill(3) + '_' + data['batch_slices3'].astype(
            str).str.zfill(3)
        data['batch_slices5'] = data['batch_index'] // (batch_size / 2)
        data['batch_slices6'] = data['batch'].astype(str).str.zfill(3) + '_' + data['batch_slices5'].astype(
            str).str.zfill(3)

        for agg_feature in ['batch', 'batch_slices2', 'batch_slices4', 'batch_slices6']:
            data[f"min_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('min')
            data[f"max_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('max')
            data[f"std_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('std')
            data[f"mean_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('mean')
            data[f"median_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].transform('median')

            data[f"mean_abs_chg_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].apply(
                lambda x: np.mean(np.abs(np.diff(x))))
            data[f"abs_max_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].apply(
                lambda x: np.max(np.abs(x)))
            data[f"abs_min_{agg_feature}_{batch_size}"] = data.groupby(agg_feature)['signal'].apply(
                lambda x: np.min(np.abs(x)))

            data[f"min_{agg_feature}_{batch_size}_diff"] = data[f"min_{agg_feature}_{batch_size}"] - data['signal']
            data[f"max_{agg_feature}_{batch_size}_diff"] = data[f"max_{agg_feature}_{batch_size}"] - data['signal']
            data[f"std_{agg_feature}_{batch_size}_diff"] = data[f"std_{agg_feature}_{batch_size}"] - data['signal']
            data[f"mean_{agg_feature}_{batch_size}_diff"] = data[f"mean_{agg_feature}_{batch_size}"] - data['signal']
            data[f"median_{agg_feature}_{batch_size}_diff"] = data[f"median_{agg_feature}_{batch_size}"] - data[
                'signal']

            data[f"range_{agg_feature}_{batch_size}"] = data[f"max_{agg_feature}_{batch_size}"] - data[
                f"min_{agg_feature}_{batch_size}"]
            data[f"maxtomin_{agg_feature}_{batch_size}"] = data[f"max_{agg_feature}_{batch_size}"] / data[
                f"min_{agg_feature}_{batch_size}"]
            data[f"abs_avg_{agg_feature}_{batch_size}"] = (data[f"abs_min_{agg_feature}_{batch_size}"] + data[
                f"abs_max_{agg_feature}_{batch_size}"]) / 2

            data[f'signal_shift+1_{agg_feature}_{batch_size}'] = data.groupby([agg_feature]).shift(1)['signal']
            data[f'signal_shift-1_{agg_feature}_{batch_size}'] = data.groupby([agg_feature]).shift(-1)['signal']
            data[f'signal_shift+2_{agg_feature}_{batch_size}'] = data.groupby([agg_feature]).shift(2)['signal']
            data[f'signal_shift-2_{agg_feature}_{batch_size}'] = data.groupby([agg_feature]).shift(-2)['signal']

            data[f"signal_shift+1_{agg_feature}_{batch_size}_diff"] = data[f"signal_shift+1_{agg_feature}_{batch_size}"] - data['signal']
            data[f"signal_shift-1_{agg_feature}_{batch_size}_diff"] = data[f"signal_shift-1_{agg_feature}_{batch_size}"] - data['signal']
            data[f"signal_shift+2_{agg_feature}_{batch_size}_diff"] = data[f"signal_shift+2_{agg_feature}_{batch_size}"] - data['signal']
            data[f"signal_shift-2_{agg_feature}_{batch_size}_diff"] = data[f"signal_shift-2_{agg_feature}_{batch_size}"] - data['signal']

        for window in window_sizes:
            window = min(batch_size, window)

            data["rolling_mean_" + str(window) + '_batch_' + str(batch_size)] = \
                data.groupby('batch')['signal'].rolling(window=window).mean().reset_index()['signal']
            data["rolling_std_" + str(window) + '_batch_' + str(batch_size)] = \
                data.groupby('batch')['signal'].rolling(window=window).std().reset_index()['signal']
            data["rolling_min_" + str(window) + '_batch_' + str(batch_size)] = \
                data.groupby('batch')['signal'].rolling(window=window).min().reset_index()['signal']
            data["rolling_max_" + str(window) + '_batch_' + str(batch_size)] = \
                data.groupby('batch')['signal'].rolling(window=window).max().reset_index()['signal']

            data[f'exp_Moving__{window}_{batch_size}'] = data.groupby('batch')['signal'].apply(
                lambda x: x.ewm(alpha=0.5, adjust=False).mean())
        data = reduce_mem_usage(data)
    data.fillna(0, inplace=True)

    return data


def read_data(path: str = ''):
    """
    Read train, test data

    Args:
        path: path to the data

    Returns:
        two dataframes
    """
    train_df = pd.read_csv(f'{path}/train.csv')
    test_df = pd.read_csv(f'{path}/test.csv')
    return train_df, test_df


if __name__ == '__main__':
    """
    Example of usage:
    >>> python generate_data.py --save_format h5
    """

    parser = argparse.ArgumentParser(description="Generate data for ion competition")
    parser.add_argument("--path", help="path to files", type=str, default='data')
    parser.add_argument("--save_format", help="format to save", type=str, default='csv',
                        choices=['csv', 'h5', 'feather'])

    args = parser.parse_args()

    train, test = read_data(args.path)
    batch_sizes = [5000, 10000, 25000, 50000, 100000, 500000]
    window_sizes = [500, 1000, 5000, 10000, 25000]

    generated_train = generate_features(train, batch_sizes, window_sizes)
    with open(f'{args.path}/column_names.txt', 'w') as f:
        for col in generated_train.columns:
            f.write(col + '\n')

    if args.save_format == 'csv':
        generated_train.to_csv(f'{args.path}/generated_train.csv', index=False)
    elif args.save_format == 'h5':
        generated_train.to_hdf(f'{args.path}/generated_train.h5', 'train', append=True)
    elif args.save_format == 'feather':
        generated_train.to_feather(f'{args.path}/generated_train.fts')

    del train, generated_train

    generated_test = generate_features(test, batch_sizes, window_sizes)
    if args.save_format == 'csv':
        generated_test.to_csv(f'{args.path}/generated_test.csv', index=False)
    elif args.save_format == 'h5':
        generated_test.to_hdf(f'{args.path}/generated_test.h5', 'test', append=True)
    elif args.save_format == 'feather':
        generated_test.to_feather(f'{args.path}/generated_test.fts')