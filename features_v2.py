#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
This code is derived from AhmetErdem's code
The original code is in https://github.com/aerdem4/kaggle-plasticc

reference: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from features_v1 import none_or_one

__date__ = '26/12/2018'


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/features_v2.py#L6.
def extract_features_v2(df):
    df["mjd_int"] = df["mjd"].astype(int)

    df = df[df["detected"] == 1].groupby(["object_id", "mjd_int"])["flux"].max().reset_index()
    df["abs_flux"] = np.abs(df["flux"])
    for col in ["first", "last", "deep", "peak"]:
        df[col] = df["flux"].values

    df["mjd_min"] = df["mjd_int"].values
    df["mjd_max"] = df["mjd_int"].values
    max_flux = df.groupby("object_id")["flux"].transform("max")
    df["mjd_peak"] = df["mjd_int"] * (max_flux == df["flux"])
    df["mjd_deep"] = df["mjd_int"] * (df.groupby("object_id")["flux"].transform("min") == df["flux"])

    peak_time = df.groupby("object_id")["mjd_peak"].transform("max")
    period = ((df["mjd_int"] > peak_time) & (df["mjd_int"] < peak_time + 32)).astype(int)
    df["peak_32"] = (none_or_one(period) * df["flux"]) / max_flux

    df = df.groupby("object_id").agg({
        "abs_flux": "max", "first": "first", "last": "last",
        "mjd_int": "count", "peak": lambda ll: np.array(ll).argmax(),
        "deep": lambda ll: np.array(ll).argmin(), "mjd_min": "min",
        "mjd_max": "max", "mjd_peak": "max", "mjd_deep": "max",
        "peak_32": "min"
    }).reset_index()
    df["first"] /= df["abs_flux"]
    df["last"] /= df["abs_flux"]
    df["peak"] /= df["mjd_int"] - 1
    df["deep"] /= df["mjd_int"] - 1
    df["till_peak"] = df["mjd_peak"] - df["mjd_min"]
    df["after_peak"] = df["mjd_max"] - df["mjd_peak"]
    df["deep_peak"] = df["mjd_peak"] - df["mjd_deep"]

    extracted_features = ["first", "last", "peak", "deep", "till_peak",
                          "after_peak", "deep_peak", "peak_32"]

    return df[["object_id"] + extracted_features]


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/feature_extraction/features_v2.py#L6.
# This is a modified version, `sub_id` column is added to the input data.
def extract_features_v2_augmented(df):
    df["mjd_int"] = df["mjd"].astype(int)

    df = df[df["detected"] == 1].groupby(["object_id", 'sub_id', "mjd_int"])["flux"].max().reset_index()
    df["abs_flux"] = np.abs(df["flux"])
    for col in ["first", "last", "deep", "peak"]:
        df[col] = df["flux"].values

    df["mjd_min"] = df["mjd_int"].values
    df["mjd_max"] = df["mjd_int"].values
    max_flux = df.groupby(["object_id", 'sub_id'])["flux"].transform("max")
    df["mjd_peak"] = df["mjd_int"] * (max_flux == df["flux"])
    df["mjd_deep"] = df["mjd_int"] * (df.groupby(["object_id", 'sub_id'])["flux"].transform("min") == df["flux"])

    peak_time = df.groupby(["object_id", 'sub_id'])["mjd_peak"].transform("max")
    period = ((df["mjd_int"] > peak_time) & (df["mjd_int"] < peak_time + 32)).astype(int)
    df["peak_32"] = (none_or_one(period) * df["flux"]) / max_flux

    df = df.groupby(["object_id", 'sub_id']).agg({
        "abs_flux": "max", "first": "first", "last": "last",
        "mjd_int": "count", "peak": lambda ll: np.array(ll).argmax(),
        "deep": lambda ll: np.array(ll).argmin(), "mjd_min": "min",
        "mjd_max": "max", "mjd_peak": "max", "mjd_deep": "max",
        "peak_32": "min"
    }).reset_index()
    df["first"] /= df["abs_flux"]
    df["last"] /= df["abs_flux"]
    df["peak"] /= df["mjd_int"] - 1
    df["deep"] /= df["mjd_int"] - 1
    df["till_peak"] = df["mjd_peak"] - df["mjd_min"]
    df["after_peak"] = df["mjd_max"] - df["mjd_peak"]
    df["deep_peak"] = df["mjd_peak"] - df["mjd_deep"]

    extracted_features = ["first", "last", "peak", "deep", "till_peak",
                          "after_peak", "deep_peak", "peak_32"]

    return df[["object_id", 'sub_id'] + extracted_features]


@click.group()
def cmd():
    pass


@cmd.command()
@click.option('--data-path', type=click.Path(exists=True),
              default='../../data/raw/test_set.csv')
@click.option('--meta-path', type=click.Path(exists=True),
              default='../../data/raw/test_set_metadata.csv')
@click.option('--output-path', type=click.Path(),
              default='../../data/processed/4th/test_set_features_v2.pickle'
              )
def raw(data_path, meta_path, output_path):
    output_path = Path(output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    df = pd.read_csv(data_path, header=0)
    features = extract_features_v2(df=df)
    features.to_pickle(output_path)


@cmd.command()
@click.option('--data-dir', type=click.Path(exists=True),
              default='../../data/interim/augmented')
@click.option('--target', type=click.Choice([
    '15', '42', '52', '62', '64', '67', '88', '90', '95'
]))
def augmented(data_dir, target):
    data_dir = Path(data_dir)
    target = int(target)

    df_list = []
    with pd.HDFStore(data_dir / 'data{}_00.h5'.format(target)) as store:
        for key in store.keys():
            if key == '/redshift':
                continue
            value = store[key]
            df_list.append(value)
    df = pd.concat(df_list, axis=0, ignore_index=True)
    df = extract_features_v2_augmented(df)
    df.to_pickle(data_dir / 'features_v2_{}_00.pickle'.format(target))


def main():
    cmd()


if __name__ == '__main__':
    main()
