#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
This code is derived from AhmetErdem's code
The original code is in https://github.com/aerdem4/kaggle-plasticc

reference: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011
"""

import multiprocessing
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import numba
import click
from tqdm import trange, tqdm

__date__ = '26/12/2018'


# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L9
NUM_PARTITIONS = 500
LOW_PASSBAND_LIMIT = 3
FEATURES = ["A", "B", "t0", "tfall", "trise", "cc", "fit_error", "status",
            "t0_shift"]


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L14.
# # bazin, errorfunc and fit_scipy are developed using:
# # https://github.com/COINtoolbox/ActSNClass/blob/master/examples/1_fit_LC/fit_lc_parametric.py
@numba.jit(nopython=True)
def bazin(time, low_passband, A, B, t0, tfall, trise, cc):
    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
    return (A * X + B) * (1 - cc * low_passband)


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L22.
# @numba.jit(nopython=True)  # <- Errors occur in my environment.
def errfunc(params, time, low_passband, flux, weights):
    return abs(flux - bazin(time, low_passband, *params)) * weights


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L27.
def fit_scipy(time, low_passband, flux, flux_err):
    time -= time[0]
    sn = np.power(flux / flux_err, 2)
    start_point = (sn * flux).argmax()

    t0_init = time[start_point] - time[0]
    amp_init = flux[start_point]
    weights = 1 / (1 + flux_err)
    weights = weights / weights.sum()
    guess = [0, amp_init, t0_init, 40, -5, 0.5]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = least_squares(
            errfunc, guess, args=(time, low_passband, flux, weights),
            method='lm'
        )
    # noinspection PyUnresolvedReferences
    result.t_shift = t0_init - result.x[2]

    return result


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L44.
def yield_data(meta_df, lc_df):
    cols = ["object_id", "mjd", "flux", "flux_err", "low_passband"]
    for i in trange(NUM_PARTITIONS):
        meta_flag = (meta_df["object_id"] % NUM_PARTITIONS) == i
        lc_flag = (lc_df["object_id"] % NUM_PARTITIONS) == i
        yield meta_df[meta_flag]["object_id"].values, lc_df[lc_flag][cols]


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L44.
# This is modified. The object ids are not integers.
def yield_data_augmented(meta_df, lc_df):
    cols = ["object_id", "mjd", "flux", "flux_err", "low_passband"]
    for i in trange(NUM_PARTITIONS):
        object_id_list = meta_df.iloc[i::NUM_PARTITIONS]["object_id"].values
        s = set(object_id_list)
        flag = np.empty(len(lc_df), dtype=np.bool)
        for j, object_id in enumerate(lc_df['object_id'].values):
            flag[j] = object_id in s
        yield object_id_list, lc_df.loc[flag, cols]


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L51.
def get_params(object_id_list, lc_df, result_queue):
    results = {}
    for object_id in object_id_list:
        light_df = lc_df[lc_df["object_id"] == object_id]
        try:
            result = fit_scipy(
                light_df["mjd"].values, light_df["low_passband"].values,
                light_df["flux"].values, light_df["flux_err"].values
            )
            # noinspection PyUnresolvedReferences
            results[object_id] = np.append(
                result.x, [result.cost, result.status, result.t_shift]
            )
        except Exception as e:
            print(e)
            results[object_id] = None
    result_queue.put(results)


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L65.
def parallelize(meta_df, df):
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    for m, d in yield_data(meta_df, df):
        pool.apply_async(get_params, (m, d, result_queue))

    pool.close()
    pool.join()

    return [result_queue.get() for _ in range(NUM_PARTITIONS)]


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L65.
# This is modified to call the function `yield_data_augmented`
def parallelize_augmented(meta_df, df):
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    for m, d in tqdm(yield_data_augmented(meta_df, df), total=NUM_PARTITIONS):
        pool.apply_async(get_params, (m, d, result_queue))

    pool.close()
    pool.join()

    return [result_queue.get() for _ in trange(NUM_PARTITIONS)]


@click.group()
def cmd():
    pass


# The following part is derived from
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L82.
@cmd.command()
@click.option('--data-path', type=click.Path(exists=True),
              default='../../data/raw/test_set.csv')
@click.option('--meta-path', type=click.Path(exists=True),
              default='../../data/raw/test_set_metadata.csv')
@click.option('--output-path', type=click.Path(),
              default='../../data/processed/4th/test_set_bazin_features.pickle'
              )
def raw(data_path, meta_path, output_path):
    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    meta_df = pd.read_csv(meta_path, header=0)
    df = pd.read_csv(data_path, header=0)

    df['low_passband'] = (df['passband'] < LOW_PASSBAND_LIMIT).astype(int)

    result_list = parallelize(meta_df=meta_df, df=df)
    final_result = {}
    for res in result_list:
        final_result.update(res)

    for index, col in enumerate(FEATURES):
        meta_df[col] = meta_df["object_id"].apply(
            lambda x: final_result[x][index])

    # meta_df[["object_id"] + FEATURES].to_csv(output_path, index=False)
    out_df = meta_df[["object_id"] + FEATURES]
    out_df.to_pickle(output_path)


# The following part is derived from
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/bazin.py#L82.
@cmd.command()
@click.option('--hdf5-path', type=click.Path(exists=True),
              default='../../data/processed/4th/train_augment_v3_40.h5')
@click.option('--output-path', type=click.Path(),
              default='../../data/processed/4th/test_set_bazin_features.pickle'
              )
def augmented(hdf5_path, output_path):
    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    df = pd.read_hdf(hdf5_path, 'df')   # type: pd.DataFrame
    meta_df = pd.read_hdf(hdf5_path, 'meta')    # type: pd.DataFrame

    df['low_passband'] = (df['passband'] < LOW_PASSBAND_LIMIT).astype(int)

    result_list = parallelize_augmented(meta_df=meta_df, df=df)
    final_result = {}
    for res in result_list:
        final_result.update(res)

    for index, col in enumerate(FEATURES):
        meta_df[col] = meta_df["object_id"].apply(
            lambda x: final_result[x][index])

    # meta_df[["object_id"] + FEATURES].to_csv(output_path, index=False)
    out_df = meta_df[["object_id"] + FEATURES]
    out_df.to_pickle(output_path)


def main():
    cmd()


if __name__ == '__main__':
    main()
