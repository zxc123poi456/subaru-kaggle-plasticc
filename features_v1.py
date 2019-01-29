#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
This code is derived from AhmetErdem's code.
The original code is in https://github.com/aerdem4/kaggle-plasticc.

reference: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from tqdm import trange

__date__ = '26/12/2018'


# This function is implemented by AhmetErdem
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L7.
def none_or_one(pd_series):
    return pd_series/pd_series


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/features_v1.py#L7.
# Some features are added/modified to run normally.
def extract_features_v1(df):
    df["std_flux"] = df["flux"].values
    df["min_flux"] = df["flux"].values
    df["max_flux"] = df["flux"].values
    df["detected_flux"] = df["flux"] * df["detected"]
    df["flux_sign"] = np.sign(df["flux"])
    df["observation_count"] = 1

    df["detected_mjd_max"] = df["mjd"] * none_or_one(df["detected"])
    df["detected_mjd_min"] = df["mjd"] * none_or_one(df["detected"])
    df["fake_flux"] = df["flux"] - np.sign(df["flux"]) * df["flux_err"]

    df["diff"] = df["flux"] - df.groupby(["object_id", "passband"])["flux"].shift(1)
    df["time_diff"] = df.groupby(["object_id", "detected", "flux_sign"])["mjd"].shift(-1) - df["mjd"]
    df["time_diff_pos"] = df["time_diff"] * none_or_one(df["detected"]) * (df["flux_sign"] == 1)
    df["time_diff_neg"] = df["time_diff"] * none_or_one(df["detected"]) * (df["flux_sign"] == -1)

    aggs = {"detected_mjd_max": "max", "detected_mjd_min": "min", "observation_count": "count",
            "flux": "median", "flux_err": "mean",
            "std_flux": "std", "min_flux": "min", "max_flux": "max",
            "detected_flux": "max", "time_diff_pos": "max", "time_diff_neg": "max", "time_diff": "max",
            "fake_flux": kurtosis, 'detected': 'sum'}

    for i in range(6):
        df["raw_flux" + str(i)] = (df["fake_flux"]) * (df["passband"] == i)
        aggs["raw_flux" + str(i)] = "max"

        df["sn" + str(i)] = np.power(df["flux"] / df["flux_err"], 2) * (df["passband"] == i)
        aggs["sn" + str(i)] = "max"

        df["flux_sn" + str(i)] = df["flux"] * df["sn" + str(i)]
        aggs["flux_sn" + str(i)] = "max"

        df["skew" + str(i)] = (df["fake_flux"]) * ((df["passband"] == i) / (df["passband"] == i).astype(int))
        aggs["skew" + str(i)] = "skew"

        df["f" + str(i)] = (df["flux"]) * (df["passband"] == i)
        aggs["f" + str(i)] = "mean"

        df["d" + str(i)] = (df["detected"]) * (df["passband"] == i)
        aggs["d" + str(i)] = "sum"

        df["dd" + str(i)] = (df["diff"]) * (df["passband"] == i)
        aggs["dd" + str(i)] = "max"

    df = df.groupby("object_id").agg(aggs).reset_index()
    df = df.rename(columns={"detected": "total_detected"})
    df["time_diff_full"] = df["detected_mjd_max"] - df["detected_mjd_min"]
    df["detected_period"] = df["time_diff_full"] / df["total_detected"]
    df["ratio_detected"] = df["total_detected"] / df["observation_count"]
    df["delta_flux"] = df["max_flux"] - df["min_flux"]

    for col in ["sn", "flux_sn", "f", "dd"]:
        total_sum = df[[col + str(i) for i in range(6)]].sum(axis=1)
        for i in range(6):
            df[col + str(i)] /= total_sum

    return df


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data-path', type=click.Path(exists=True),
              default='../../data/raw/test_set.csv')
@click.option('--meta-path', type=click.Path(exists=True),
              default='../../data/raw/test_set_metadata.csv')
@click.option('--output-path', type=click.Path(),
              default='../../data/processed/4th/test_set_features_v1.pickle')
def raw(data_path, meta_path, output_path):
    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    df = pd.read_csv(data_path, header=0)
    if 'test' in data_path:
        feature_list = []
        mod = df['object_id'] % 256
        for i in trange(256):
            tmp = df[mod == i].copy()
            tmp_feature = extract_features_v1(df=tmp)
            feature_list.append(tmp_feature)
        features = pd.concat(feature_list, axis=0)
    else:
        features = extract_features_v1(df=df)
    features.to_pickle(output_path)


@cmd.command()
@click.option('--hdf5-path', type=click.Path(exists=True),
              default='../../data/processed/4th/train_augment_v3_40.h5')
@click.option('--output-path', type=click.Path(),
              default='../../data/processed/4th/test_set_features_v1.pickle')
def augmented(hdf5_path, output_path):
    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    df = pd.read_hdf(hdf5_path, 'df')   # type: pd.DataFrame

    features = extract_features_v1(df=df)
    features.to_pickle(output_path)


def main():
    cmd()


if __name__ == '__main__':
    main()
