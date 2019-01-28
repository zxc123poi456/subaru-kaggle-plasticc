#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
This code is derived from AhmetErdem's code
The original code is in https://github.com/aerdem4/kaggle-plasticc

reference: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

__date__ = '25/12/2018'


NUM_FOLDS = 5


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L15.
def get_hostgal_range(hostgal_photoz):
    return np.clip(hostgal_photoz//0.2, 0, 6)


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/weight_samples.py#L4
class SampleWeighter(object):
    # Example usage:
    # >> sw = SampleWeighter(train_exgal["hostgal_photoz"],
    #                        test_exgal["hostgal_photoz"])
    # >> train_exgal = calculate_weights(train_exgal, False)

    def __init__(self, train_exgal_hp, test_exgal_hp):
        train_exgal_hr = get_hostgal_range(train_exgal_hp)
        test_exgal_hr = get_hostgal_range(test_exgal_hp)

        # noinspection PyUnresolvedReferences
        train_hp_dist = (train_exgal_hr.value_counts() /
                         len(train_exgal_hr)).to_dict()
        # noinspection PyUnresolvedReferences
        test_hp_dist = (test_exgal_hr.value_counts() /
                        len(test_exgal_hr)).to_dict()
        self.weight_list = [
            test_hp_dist[i] / train_hp_dist[i]
            for i in range(int(train_exgal_hr.max()) + 1)
        ]

    def calculate_weights(self, df, is_galactic):
        # gives weights so that test set hostgal_photoz distribution
        # is represented
        if is_galactic:
            df["sample_weight"] = 1.0
        else:
            # noinspection PyUnresolvedReferences
            df["sample_weight"] = (
                get_hostgal_range(df["hostgal_photoz"]).apply(
                    lambda x: self.weight_list[int(x)]
                )
            )
            # df["sample_weight"] = 1.0

        # gives more weights to non-ddf
        # because they are more common in test set
        df["sample_weight"] *= (2 - df["ddf"])

        # normalizes the weights so that each class has total sum of 100
        # (effecting training equally)
        df["sample_weight"] *= (
                100 / df.groupby("target")["sample_weight"].transform("sum")
        )

        # doubles weights for class 15 and class 64
        df["sample_weight"] *= df["target"].apply(
            lambda x: 1 + (x in {15, 64})
        )
        return df


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L27.
def map_classes(df):
    class_list = df["target"].value_counts(ascending=True).index
    class_list = [int(c) for c in class_list]
    class_dict = {}
    for i, c in enumerate(class_list):
        class_dict[c] = i

    df["target"] = df["target"].map(class_dict)
    return df, class_list


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L37
# Features using Gaussian process are added.
# Modified to select data for 2 step classification.
def load_data(feature_data_dir, meta_data_dir, sn_only=False):
    feature_data_dir = Path(feature_data_dir)
    meta_data_dir = Path(meta_data_dir)

    train_df = pd.read_csv(
        meta_data_dir / 'training_set_metadata.csv', header=0
    )
    test_df = pd.read_csv(
        meta_data_dir / 'test_set_metadata.csv', header=0
    )
    for feature_file in ["bazin_features", "features_v1", "features_v2"]:
        train_df = train_df.merge(
            pd.read_pickle(
                feature_data_dir /
                "training_set_{}.pickle".format(feature_file)
            ),
            on="object_id", how="left")
        test_df = test_df.merge(
            pd.read_pickle(
                feature_data_dir / "test_set_{}.pickle".format(feature_file)
            ),
            on="object_id", how="left")
    # GP
    gp_features_path = (
        feature_data_dir / 'training_set_1st_bin1_2DGPR_FE_all_190110_1.csv'
    )
    assert gp_features_path.exists()
    train_df = train_df.merge(
        pd.read_csv(gp_features_path, header=0),
        on="object_id", how="left"
    )
    gp_features_path = (
        feature_data_dir / 'test_set_1st_bin1_2DGPR_FE_all_190110_1.csv'
    )
    assert gp_features_path.exists()
    test_df = test_df.merge(
        pd.read_csv(gp_features_path, header=0),
        on="object_id", how="left"
    )

    # hostgal_calc_df = pd.read_csv("features/hostgal_calc.csv")
    # train_df = train_df.merge(hostgal_calc_df, on="object_id", how="left")
    # test_df = test_df.merge(hostgal_calc_df, on="object_id", how="left")

    train_gal = train_df[train_df["hostgal_photoz"] == 0].copy()
    train_exgal = train_df[train_df["hostgal_photoz"] > 0].copy()
    test_gal = test_df[test_df["hostgal_photoz"] == 0].copy()
    test_exgal = test_df[test_df["hostgal_photoz"] > 0].copy()

    # 超新星だけ
    if sn_only:
        # 精度が悪いので67を追加、95は除外
        sn_class = (42, 52, 62, 67, 90)

        flag = np.zeros(len(train_exgal), dtype=np.bool)
        for c in sn_class:
            flag = np.logical_or(flag, train_exgal['target'] == c)
        train_exgal = train_exgal[flag]

        flag = np.zeros(len(test_exgal), dtype=np.bool)
        for c in sn_class:
            flag = np.logical_or(flag, test_exgal['target'] == c)
        test_exgal = test_exgal[flag]

    sw = SampleWeighter(train_exgal["hostgal_photoz"],
                        test_exgal["hostgal_photoz"])

    train_gal = sw.calculate_weights(train_gal, True)
    train_exgal = sw.calculate_weights(train_exgal, False)

    train_gal, gal_class_list = map_classes(train_gal)
    train_exgal, exgal_class_list = map_classes(train_exgal)

    if sn_only:
        test_df = pd.concat([test_gal, test_exgal], axis=0)
    return (train_gal, train_exgal, test_gal, test_exgal,
            gal_class_list, exgal_class_list,
            test_df[["object_id", "hostgal_photoz"]])


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L37
# Modified to load only training data
def load_data_v2(data_dir, meta_path):
    data_dir = Path(data_dir)

    df = pd.read_csv(meta_path, header=0)

    tmp_list = []
    for f in data_dir.glob('bazin*.pickle'):
        # noinspection PyTypeChecker
        tmp = pd.read_pickle(f)
        tmp_list.append(tmp)
    bazin = pd.concat(tmp_list, axis=0, ignore_index=True)
    df = df.merge(bazin, on='object_id', how='right')

    for name in ('features_v1', 'features_v2'):
        tmp_list = []
        for f in data_dir.glob('{}_*.pickle'.format(name)):
            # noinspection PyTypeChecker
            tmp = pd.read_pickle(f)
            tmp_list.append(tmp)
        features = pd.concat(tmp_list, axis=0, ignore_index=True)
        df = df.merge(features, on=['object_id', 'sub_id'], how='left')

    redshift_list = []
    for f in data_dir.glob('data*.h5'):
        # noinspection PyTypeChecker
        with pd.HDFStore(f) as store:
            tmp = store['/redshift']
            tmp.index.name = 'object_id'
            tmp = pd.melt(
                tmp.reset_index(), id_vars=['object_id'],
                value_vars=tmp.columns, var_name='sub_id',
                value_name='hostgal_calc'
            )
            redshift_list.append(tmp)
    redshift = pd.concat(redshift_list, axis=0)
    redshift = redshift.astype({'sub_id': np.int})
    df = df.merge(redshift, on=['object_id', 'sub_id'], how='left')

    df.to_csv(data_dir / 'features.csv')

    df_galactic = df.copy()
    df_extra = df.copy()

    sw = SampleWeighter(df_galactic["hostgal_photoz"],
                        df_galactic["hostgal_photoz"])

    df_galactic = sw.calculate_weights(df_galactic, True)
    df_extra = sw.calculate_weights(df_extra, False)

    df_galactic, galactic_class_list = map_classes(df_galactic)
    df_extra, extra_class_list = map_classes(df_extra)

    return df_galactic, galactic_class_list, df_extra, extra_class_list


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L21
# training for
def train(df, features, parameters):
    oof_predictions = np.zeros((len(df), parameters['num_classes']))

    if 'sub_id' in df.columns:
        unique_id, index = np.unique(df['object_id'], return_index=True)
        target = df.iloc[index]['target']
    else:
        target = df['target'].values

    skf = StratifiedKFold(5, random_state=42)
    for train_index, validation_index in skf.split(target, target):
        if 'sub_id' in df.columns:
            flag_train = np.zeros(len(df), dtype=np.bool)
            # noinspection PyUnboundLocalVariable
            for i in unique_id[train_index]:
                flag_train = np.logical_or(flag_train, df['object_id'] == i)
            dev = df[flag_train]

            flag_validation = np.zeros(len(df), dtype=np.bool)
            for i in unique_id[validation_index]:
                flag_validation = np.logical_or(flag_validation,
                                                df['object_id'] == i)
            val = df[flag_validation]
        else:
            dev, val = df.iloc[train_index], df.iloc[validation_index]

        lgb_train = lgb.Dataset(dev[features], dev['target'],
                                weight=dev['sample_weight'])
        lgb_validation = lgb.Dataset(val[features], val['target'],
                                     weight=val['sample_weight'])

        model = lgb.train(
            parameters, lgb_train, num_boost_round=200,
            valid_sets=[lgb_train, lgb_validation], early_stopping_rounds=10,
            verbose_eval=50
        )
        if 'sub_id' in df.columns:
            # noinspection PyUnboundLocalVariable
            oof_predictions[flag_validation, :] = model.predict(val[features])
        else:
            oof_predictions[validation_index, :] = model.predict(val[features])

    return oof_predictions


# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L21
# Modified to output accuracy and importance of features
def train_and_predict(train_df, test_df, features, params):
    oof_preds = np.zeros((len(train_df), params["num_class"]))
    test_preds = np.zeros((len(test_df), params["num_class"]))

    skf = StratifiedKFold(NUM_FOLDS, random_state=4)

    importance_list = []
    current_iteration_list = []
    results = {'train_accuracy': [], 'validation_accuracy': [],
               'train_predictions': []}
    for train_index, val_index in skf.split(train_df, train_df["target"]):
        dev_df, val_df = train_df.iloc[train_index], train_df.iloc[val_index]
        lgb_train = lgb.Dataset(
            dev_df[features], dev_df["target"], weight=dev_df["sample_weight"]
        )
        lgb_val = lgb.Dataset(
            val_df[features], val_df["target"], weight=val_df["sample_weight"]
        )

        model = lgb.train(
            params, lgb_train, num_boost_round=200,
            valid_sets=[lgb_train, lgb_val], early_stopping_rounds=10,
            verbose_eval=50
        )
        val_predictions = model.predict(val_df[features])
        oof_preds[val_index, :] = val_predictions

        test_preds += model.predict(test_df[features]) / NUM_FOLDS
        current_iteration_list.append(model.current_iteration())

        val_accuracy = float(
            np.count_nonzero(val_df['target'] ==
                             np.argmax(val_predictions, axis=1)) /
            len(val_predictions)
        )
        # noinspection PyTypeChecker
        results['validation_accuracy'].append(val_accuracy)

        train_predictions = model.predict(dev_df[features])
        dev_accuracy = float(
            np.count_nonzero(dev_df['target'] ==
                             np.argmax(train_predictions, axis=1)) /
            len(train_predictions)
        )
        # noinspection PyTypeChecker
        results['train_accuracy'].append(dev_accuracy)
        # noinspection PyTypeChecker
        results['train_predictions'].append((
            train_predictions, dev_df['target']
        ))

        importance = pd.DataFrame(
            {'split': model.feature_importance('split'),
             'gain': model.feature_importance('gain')},
            index=model.feature_name()
        )
        importance_list.append(importance)
    importance = pd.concat(importance_list, axis=1,
                           keys=list(range(NUM_FOLDS)))
    results['importance'] = importance
    results['current_iteration'] = current_iteration_list

    return oof_preds, test_preds, results


# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L21
# Modified to train with all data
def train_and_predict_all(train_df, test_df, features, params, n_iterations):
    x_test = test_df[features]

    results = {'train_accuracy': [], 'validation_accuracy': [],
               'train_predictions': []}

    x_train = train_df[features]
    lgb_train = lgb.Dataset(
        x_train, train_df["target"], weight=train_df["sample_weight"]
    )
    model = lgb.train(
        params, lgb_train, num_boost_round=n_iterations,
        verbose_eval=50
    )

    test_preds = model.predict(x_test)

    return test_preds, results


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L41
# Add new features, remove features which do not exist
def get_lgb_predictions(train_gal, train_exgal, test_gal, test_exgal,
                        gal_class_list, exgal_class_list, model_dir,
                        exgal_feature_size, use_all):
    bazin = ["A", "B", "tfall", "trise", "cc", "fit_error", "t0_shift"]
    f_flux = (["flux_sn" + str(i) for i in range(6)] +
              ["sn" + str(i) for i in range(6)])
    f_skew = ["skew" + str(i) for i in range(6)]
    f_f = ["f" + str(i) for i in range(6)]
    f_d = ["d" + str(i) for i in range(6)]
    f_dd = ["dd" + str(i) for i in range(6)]
    v3_features = ['first', 'last', 'peak', 'deep', 'till_peak', 'after_peak',
                   'deep_peak', 'peak_32']
    # peak_time = ["peak_time" + str(i) for i in [0, 1, 4, 5]]

    features_gal = (
            ['mwebv', 'flux', 'flux_err', 'fake_flux', 'total_detected',
             'ratio_detected', 'observation_count', 'std_flux', 'min_flux',
             'max_flux', 'delta_flux', 'detected_flux', 'time_diff_pos',
             'time_diff_neg'] +
            f_flux + f_skew + f_f + f_d + f_dd + v3_features + bazin
    )

    # features_exgal = (
    #         ['hostgal_photoz', 'hostgal_photoz_err', 'hostgal_calc', 'mwebv',
    #          'fake_flux', 'time_diff_pos', 'time_diff_neg'] +
    #         f_flux + f_skew + f_f + f_d + v3_features + bazin + peak_time
    # )
    # features_exgal = (
    #         ['hostgal_photoz', 'hostgal_photoz_err', 'mwebv',
    #          'fake_flux', 'time_diff_pos', 'time_diff_neg'] +
    #         f_flux + f_skew + f_f + f_d + v3_features + bazin +
    #         ['M_u', 'dMm10_u', 'dMm5_u', 'dM5_u', 'dM10_u', 'dM15_u',
    #          'dM20_u', 'dM30_u', 'dM40_u', 'dM50_u', 'dM60_u', 'skew_u',
    #          'kurto_u', 'M_g', 'dMm10_g', 'dMm5_g', 'dM5_g', 'dM10_g',
    #          'dM15_g', 'dM20_g', 'dM30_g', 'dM40_g', 'dM50_g', 'dM60_g',
    #          'skew_g', 'kurto_g', 'M_r', 'dMm10_r', 'dMm5_r', 'dM5_r',
    #          'dM10_r', 'dM15_r', 'dM20_r', 'dM30_r', 'dM40_r', 'dM50_r',
    #          'dM60_r', 'skew_r', 'kurto_r', 'M_i', 'dMm10_i', 'dMm5_i',
    #          'dM5_i', 'dM10_i', 'dM15_i', 'dM20_i', 'dM30_i', 'dM40_i',
    #          'dM50_i', 'dM60_i', 'skew_i', 'kurto_i', 'M_z', 'dMm10_z',
    #          'dMm5_z', 'dM5_z', 'dM10_z', 'dM15_z', 'dM20_z', 'dM30_z',
    #          'dM40_z', 'dM50_z', 'dM60_z', 'skew_z', 'kurto_z', 'M_Y',
    #          'dMm10_Y', 'dMm5_Y', 'dM5_Y', 'dM10_Y', 'dM15_Y', 'dM20_Y',
    #          'dM30_Y', 'dM40_Y', 'dM50_Y', 'dM60_Y', 'skew_Y', 'kurto_Y',
    #          'c_gr', 'c_ri', 'c_iz']
    # )
    # 重要度(split)で降順に並んでいる
    features_exgal = [
        'M_z', 'M_Y', 'cc', 'tfall', 'trise', 'fit_error', 'M_r',
        'fake_flux', 'time_diff_pos', 'c_gr', 'M_u', 'M_i', 'M_g', 'B',
        'hostgal_photoz', 'till_peak', 'skew1', 'f0', 'flux_sn0', 'c_iz',
        'c_ri', 'skew_r', 'dMm10_r', 't0_shift', 'flux_sn5', 'after_peak',
        'f1', 'flux_sn1', 'time_diff_neg', 'dMm10_g', 'kurto_z', 'f2',
        'dM60_i', 'sn0', 'd1', 'skew4', 'peak_32', 'skew_u', 'A', 'f3',
        'kurto_g', 'sn4', 'skew_i', 'dM40_z', 'dM60_z', 'skew0', 'dMm10_i',
        'kurto_r', 'first', 'dM30_r', 'skew5', 'deep_peak', 'skew_g',
        'skew_Y', 'kurto_Y', 'dM20_g', 'd5', 'kurto_u', 'dMm5_g', 'd2',
        'flux_sn4', 'last', 'sn1', 'sn2', 'dM50_u', 'sn3', 'dM50_i',
        'dM60_Y', 'dM10_g', 'd3', 'dM40_Y', 'dM20_r', 'f5', 'skew3',
        'dM60_g', 'kurto_i', 'flux_sn3', 'peak', 'dMm10_u', 'dMm5_r',
        'hostgal_photoz_err', 'dM50_z', 'sn5', 'dM50_g', 'dM50_Y', 'dM5_r',
        'flux_sn2', 'skew_z', 'dM20_Y', 'dM30_z', 'dM60_u', 'dM40_r',
        'dMm5_Y', 'dM10_Y', 'dMm5_i', 'dM5_i', 'mwebv', 'dM30_u', 'dM40_u',
        'dM15_z', 'dM15_i', 'skew2', 'dM20_u', 'dM30_g', 'dM60_r',
        'dM20_z', 'dM5_g', 'dM5_u', 'dM15_r', 'dMm5_u', 'dM10_z', 'dM10_r',
        'dM30_i', 'dM20_i', 'deep', 'dM30_Y', 'dM50_r', 'f4', 'dM5_z',
        'dM5_Y', 'dMm10_z', 'dM15_u', 'dM15_g', 'dM10_i', 'dM40_g',
        'dMm10_Y', 'd4', 'dM10_u', 'd0', 'dMm5_z', 'dM40_i', 'dM15_Y'
    ]
    if 0 < exgal_feature_size < len(features_exgal):
        features_exgal = features_exgal[:exgal_feature_size]

    params_gal = {"objective": "multiclass",
                  "num_class": len(gal_class_list),
                  "min_data_in_leaf": 200,
                  "num_leaves": 5,
                  "feature_fraction": 0.7
                  }

    params_exgal = {"objective": "multiclass",
                    "num_class": len(exgal_class_list),
                    "min_data_in_leaf": 200,
                    "num_leaves": 5,
                    "feature_fraction": 0.7
                    }

    if use_all:
        print("GALACTIC MODEL")
        test_preds_gal, results_gal = train_and_predict_all(
            train_gal, test_gal, features_gal, params_gal, n_iterations=150
        )
        print("EXTRAGALACTIC MODEL")
        test_preds_exgal, results_exgal = train_and_predict_all(
            train_exgal, test_exgal, features_exgal, params_exgal,
            n_iterations=150
        )

        return test_preds_gal, test_preds_exgal
    else:
        print("GALACTIC MODEL")
        oof_preds_gal, test_preds_gal, results_gal = train_and_predict(
            train_gal, test_gal, features_gal, params_gal
        )
        print("EXTRAGALACTIC MODEL")
        oof_preds_exgal, test_preds_exgal, results_exgal = train_and_predict(
            train_exgal, test_exgal, features_exgal, params_exgal
        )

        evaluate(train_gal, train_exgal, oof_preds_gal, oof_preds_exgal,
                 model_dir)

        gal_train, gal_label = [], []
        for p, t in results_gal['train_predictions']:
            gal_train.append(p)
            gal_label.append(t)
        gal_train = np.vstack(gal_train)
        gal_label = np.hstack(gal_label)
        draw_confusion_matrix(
            target=gal_label, prediction=gal_train, class_list=gal_class_list,
            path=model_dir / 'train_confusion_matrix_galactic.png'
        )
        exgal_train, exgal_label = [], []
        for p, t in results_exgal['train_predictions']:
            exgal_train.append(p)
            exgal_label.append(t)
        exgal_train = np.vstack(exgal_train)
        exgal_label = np.hstack(exgal_label)
        draw_confusion_matrix(
            target=exgal_label, prediction=exgal_train,
            class_list=exgal_class_list,
            path=model_dir / 'train_confusion_matrix_extra_galactic.png'
        )
        del results_gal['train_predictions']
        del results_exgal['train_predictions']

        results_gal['importance'].to_csv(model_dir / 'importance_galactic.csv')
        results_exgal['importance'].to_csv(
            model_dir / 'importance_extra_galactic.csv'
        )
        del results_gal['importance'], results_exgal['importance']

        with (model_dir / 'galactic_result.json').open('w') as f:
            json.dump(results_gal, f, sort_keys=True, indent=4)
        with (model_dir / 'extra_galactic_result.json').open('w') as f:
            json.dump(results_exgal, f, sort_keys=True, indent=4)

        return oof_preds_gal, oof_preds_exgal, test_preds_gal, test_preds_exgal


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L41
# Modified or added new features
def get_lgb_predictions_augmented(df_galactic, galactic_class_list,
                                  df_extra, extra_class_list):
    bazin = ["A", "B", "tfall", "trise", "cc", "fit_error", "t0_shift"]
    f_flux = ["flux_sn" + str(i) for i in range(6)] + ["sn" + str(i) for i in
                                                       range(6)]
    f_skew = ["skew" + str(i) for i in range(6)]
    f_f = ["f" + str(i) for i in range(6)]
    f_d = ["d" + str(i) for i in range(6)]
    f_dd = ["dd" + str(i) for i in range(6)]
    v3_features = ['first', 'last', 'peak', 'deep', 'till_peak', 'after_peak',
                   'deep_peak', 'peak_32']
    # peak_time = ["peak_time" + str(i) for i in [0, 1, 4, 5]]

    features_gal = (
            ['mwebv', 'flux', 'flux_err', 'fake_flux', 'total_detected',
             'ratio_detected', 'observation_count', 'std_flux', 'min_flux',
             'max_flux', 'delta_flux', 'detected_flux', 'time_diff_pos',
             'time_diff_neg'] +
            f_flux + f_skew + f_f + f_d + f_dd + v3_features + bazin
    )
    print('feature size (galactic): {}'.format(len(features_gal)))

    # features_exgal = (
    #         ['hostgal_photoz', 'hostgal_photoz_err', 'hostgal_calc', 'mwebv',
    #          'fake_flux', 'time_diff_pos', 'time_diff_neg'] +
    #         f_flux + f_skew + f_f + f_d + v3_features + bazin + peak_time
    # )
    # features_exgal = (
    #         ['hostgal_photoz', 'hostgal_photoz_err', 'hostgal_photoz',
    #          'mwebv', 'fake_flux', 'time_diff_pos', 'time_diff_neg'] +
    #         f_flux + f_skew + f_f + f_d + v3_features + bazin
    # )
    features_exgal = (
            ['hostgal_photoz', 'hostgal_photoz_err', 'mwebv',
             'fake_flux', 'time_diff_pos', 'time_diff_neg'] +
            f_flux + f_skew + f_f + f_d + v3_features + bazin +
            ['M_u', 'dMm10_u', 'dMm5_u', 'dM5_u', 'dM10_u', 'dM15_u',
             'dM20_u', 'dM30_u', 'dM40_u', 'dM50_u', 'dM60_u', 'skew_u',
             'kurto_u', 'M_g', 'dMm10_g', 'dMm5_g', 'dM5_g', 'dM10_g',
             'dM15_g', 'dM20_g', 'dM30_g', 'dM40_g', 'dM50_g', 'dM60_g',
             'skew_g', 'kurto_g', 'M_r', 'dMm10_r', 'dMm5_r', 'dM5_r',
             'dM10_r', 'dM15_r', 'dM20_r', 'dM30_r', 'dM40_r', 'dM50_r',
             'dM60_r', 'skew_r', 'kurto_r', 'M_i', 'dMm10_i', 'dMm5_i',
             'dM5_i', 'dM10_i', 'dM15_i', 'dM20_i', 'dM30_i', 'dM40_i',
             'dM50_i', 'dM60_i', 'skew_i', 'kurto_i', 'M_z', 'dMm10_z',
             'dMm5_z', 'dM5_z', 'dM10_z', 'dM15_z', 'dM20_z', 'dM30_z',
             'dM40_z', 'dM50_z', 'dM60_z', 'skew_z', 'kurto_z', 'M_Y',
             'dMm10_Y', 'dMm5_Y', 'dM5_Y', 'dM10_Y', 'dM15_Y', 'dM20_Y',
             'dM30_Y', 'dM40_Y', 'dM50_Y', 'dM60_Y', 'skew_Y', 'kurto_Y',
             'c_gr', 'c_ri', 'c_iz']
    )
    print('feature size (extra galactic): {}'.format(len(features_exgal)))

    params_gal = {
        "objective": "multiclass", "num_classes": len(galactic_class_list),
        "min_data_in_leaf": 200, "num_leaves": 5, "feature_fraction": 0.7
    }

    params_exgal = {
        "objective": "multiclass", "num_classes": len(extra_class_list),
        "min_data_in_leaf": 200, "num_leaves": 5, "feature_fraction": 0.7
    }

    print("GALACTIC MODEL")
    oof_preds_gal = train(df_galactic, features_gal, params_gal)
    print("EXTRAGALACTIC MODEL")
    oof_preds_exgal = train(df_extra, features_exgal, params_exgal)

    if 'sub_id' in df_extra.columns:
        oof_preds_gal = pd.DataFrame(
            oof_preds_gal,
            index=df_galactic.set_index(['object_id', 'sub_id']).index,
            columns=galactic_class_list
        )
        oof_preds_exgal = pd.DataFrame(
            oof_preds_exgal,
            index=df_extra.set_index(['object_id', 'sub_id']).index,
            columns=extra_class_list
        )
    else:
        oof_preds_gal = pd.DataFrame(
            oof_preds_gal,
            index=df_galactic.set_index('object_id').index,
            columns=galactic_class_list
        )
        oof_preds_exgal = pd.DataFrame(
            oof_preds_exgal,
            index=df_extra.set_index('object_id').index,
            columns=extra_class_list
        )
    return oof_preds_gal, oof_preds_exgal


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L11
# Modified to output the result as json file
def evaluate(train_gal, train_exgal, oof_preds_gal, oof_preds_exgal,
             model_dir):
    gal_loss = log_loss(train_gal["target"], np.round(oof_preds_gal, 4),
                        sample_weight=train_gal["sample_weight"])
    exgal_loss = log_loss(train_exgal["target"], np.round(oof_preds_exgal, 4),
                          sample_weight=train_exgal["sample_weight"])
    print("Galactic CV: {}".format(gal_loss))
    print("Extragalactic CV: {}".format(exgal_loss))
    total_loss = (5 / (14 + 2)) * gal_loss + ((9 + 2) / (14 + 2)) * exgal_loss
    print("Overall CV: {}".format(total_loss))

    d = {'galactic_log_loss': float(gal_loss),
         'extra_galactic_log_loss': float(exgal_loss),
         'log_loss': float(total_loss)}
    with (model_dir / 'result.json').open('w') as f:
        json.dump(d, f, indent=4, sort_keys=True)


# Example usage: test_preds_exgal = get_meta_preds(train_exgal,
# oof_preds_exgal, test_preds_exgal, 0.2)
def get_meta_preds(train_df, oof_preds, test_preds, c):
    lr = LogisticRegression(
        C=c, intercept_scaling=0.1, multi_class="multinomial", solver="lbfgs"
    )
    lr.fit(safe_log(oof_preds), train_df["target"],
           sample_weight=train_df["sample_weight"])
    return lr.predict_proba(safe_log(test_preds))


# This function is implemented by AhmetErdem
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L15
def safe_log(x):
    return np.log(np.clip(x, 1e-4, None))


# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L88
# This function is implemented by AhmetErdem
def submit(test_df, test_preds_gal, test_preds_exgal, gal_class_list,
           exgal_class_list, sub_file):
    all_classes = [c for c in gal_class_list] + [c for c in exgal_class_list]

    gal_indices = np.where(test_df["hostgal_photoz"] == 0)[0]
    exgal_indices = np.where(test_df["hostgal_photoz"] > 0)[0]

    test_preds = np.zeros((test_df.shape[0], len(all_classes)))
    test_preds[gal_indices, :] = np.hstack((
        np.clip(test_preds_gal, 1e-4, None),
        np.zeros((test_preds_gal.shape[0], len(exgal_class_list)))
    ))
    test_preds[exgal_indices, :] = np.hstack((
        np.zeros((test_preds_exgal.shape[0], len(gal_class_list))),
        np.clip(test_preds_exgal, 1e-4, None)
    ))

    estimated99 = get_class99_proba(test_df, test_preds, all_classes)

    sub_df = pd.DataFrame(
        index=test_df['object_id'],
        data=np.round(test_preds * (1 - estimated99), 4),
        columns=['class_%d' % i for i in all_classes]
    )
    sub_df["class_99"] = estimated99

    sub_df.to_csv(sub_file)


# This function is implemented by AhmetErdem.
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L15
def get_class99_proba(test_df, test_preds, all_classes):
    base = 0.02

    high99 = (get_hostgal_range(test_df["hostgal_photoz"]) == 0)

    low99 = is_labeled_as(test_preds, all_classes, 15)
    for label in [64, 67, 88, 90]:
        low99 = low99 | is_labeled_as(test_preds, all_classes, label)
    class99 = 0.22 - 0.18 * low99 + 0.13 * high99 - base

    not_sure = (test_preds.max(axis=1) < 0.9)
    filt = (test_df["hostgal_photoz"] > 0) & not_sure

    return (base + (class99 * filt).values).reshape(-1, 1)


# This function is implemented by AhmetErdem.
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L68
def is_labeled_as(preds, class_list, label):
    return preds.argmax(axis=1) == np.where(np.array(class_list) == label)[0]


def draw_confusion_matrix(target, prediction, class_list, path):
    y_pred = np.argmax(prediction, axis=1)
    cm = confusion_matrix(y_true=target, y_pred=y_pred)

    cm = cm / np.sum(cm, axis=1, keepdims=True)
    annotation = np.around(cm, 2)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm, xticklabels=class_list, yticklabels=class_list,
                cmap='Blues', annot=annotation, lw=0.5, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')
    fig.tight_layout()

    fig.savefig(path, bbox_inches='tight')
    plt.close()


@click.group()
def cmd():
    pass


# This function is derived from
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L85
@cmd.command()
@click.option('--feature-dir', type=click.Path(exists=True),
              default='../data/processed/4th')
@click.option('--meta-dir', type=click.Path(exists=True),
              default='../data/raw')
@click.option('--model-dir', type=click.Path())
@click.option('--exgal-feature-size', type=int, default=-1)
@click.option('--use-all', is_flag=True)
@click.option('--use-meta-prediction', is_flag=True)
def raw(feature_dir, meta_dir, model_dir, exgal_feature_size, use_all,
        use_meta_prediction):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    with (model_dir / 'parameters.json').open('w') as f:
        parameters = {
            'feature_dir': feature_dir, 'meta_dir': meta_dir,
            'exgal_feature_size': exgal_feature_size,
            'use_all': use_all, 'use_meta_prediction': use_meta_prediction
        }
        json.dump(parameters, f, indent=4, sort_keys=True)

    (train_gal, train_exgal, test_gal, test_exgal,
     gal_class_list, exgal_class_list, test_df) = load_data(
        feature_data_dir=feature_dir, meta_data_dir=meta_dir
    )
    if use_all:
        test_preds_gal, test_preds_exgal = get_lgb_predictions(
            train_gal, train_exgal, test_gal, test_exgal,
            gal_class_list=gal_class_list, exgal_class_list=exgal_class_list,
            model_dir=model_dir, exgal_feature_size=exgal_feature_size,
            use_all=use_all
        )
    else:
        (oof_preds_gal, oof_preds_exgal,
         test_preds_gal, test_preds_exgal) = get_lgb_predictions(
            train_gal, train_exgal, test_gal, test_exgal,
            gal_class_list=gal_class_list, exgal_class_list=exgal_class_list,
            model_dir=model_dir, exgal_feature_size=exgal_feature_size,
            use_all=use_all
        )

        if use_meta_prediction:
            test_preds_gal = get_meta_preds(
                train_gal, oof_preds_gal, test_preds_gal, 0.2
            )
            test_preds_exgal = get_meta_preds(
                train_exgal, oof_preds_exgal, test_preds_exgal, 0.2
            )

    submit(test_df, test_preds_gal, test_preds_exgal,
           gal_class_list, exgal_class_list,
           str(model_dir / "submission_lgb.csv"))

    if not use_all:
        tmp = np.copy(train_gal['target'].values)

        # noinspection PyUnboundLocalVariable
        draw_confusion_matrix(
            target=tmp, prediction=oof_preds_gal,
            class_list=gal_class_list,
            path=model_dir / 'confusion_matrix_galactic.png'
        )
        # noinspection PyUnboundLocalVariable
        draw_confusion_matrix(
            target=train_exgal['target'].values, prediction=oof_preds_exgal,
            class_list=exgal_class_list,
            path=model_dir / 'confusion_matrix_extra_galactic.png'
        )


# This function is derived from
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L85
@click.command()
@click.option('--data-dir', type=click.Path(exists=True),
              default='../data/interim/gp2d')
@click.option('--model-dir', type=click.Path(),
              default='../models/gp2d-feature')
def augmented(data_dir, model_dir):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    (df_galactic, galactic_class_list,
     df_extra, extra_class_list) = load_data_v2(
        data_dir=data_dir,
        meta_path='../data/raw/training_set_metadata.csv'
    )

    oof_preds_gal, oof_preds_exgal = get_lgb_predictions_augmented(
        df_galactic=df_galactic, galactic_class_list=galactic_class_list,
        df_extra=df_extra, extra_class_list=extra_class_list
    )

    evaluate(df_galactic, df_extra, oof_preds_gal, oof_preds_exgal, model_dir)
    tmp = np.copy(df_galactic['target'].values)
    draw_confusion_matrix(
        target=tmp, prediction=oof_preds_gal.values,
        class_list=galactic_class_list,
        path=model_dir / 'confusion_matrix_galactic.png'
    )
    draw_confusion_matrix(
        target=df_extra['target'].values, prediction=oof_preds_exgal.values,
        class_list=extra_class_list,
        path=model_dir / 'confusion_matrix_extra_galactic.png'
    )

    oof_preds_gal.to_pickle(model_dir / 'predictions_galactic.pickle')
    oof_preds_exgal.to_pickle(model_dir / 'predictions_extra_galactic.pickle')


def main():
    cmd()


if __name__ == '__main__':
    main()
