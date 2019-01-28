#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
This code is derived from AhmetErdem's code
The original code is in https://github.com/aerdem4/kaggle-plasticc

reference: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011

We modified the classification process of extra galactic objects.
We found 2 step classification is better than 1 step classification.
"""

import json
from pathlib import Path

import click
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from tqdm import tqdm

from plasticc_simple3 import (
    SampleWeighter, map_classes, draw_confusion_matrix, get_meta_preds, submit
)
from plasticc_simple3_sn import (
    train_and_predict, get_feature_names
)

try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt
    import seaborn as sns

__date__ = '10/01/2019'


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L37
# Add new features.
def load_training_data(feature_data_dir, meta_data_dir, augmented):
    if augmented:
        train_df = pd.read_hdf(
            feature_data_dir / 'train_augment_v3_40.h5',
            'meta'
        )  # type: pd.DataFrame
        train_df.sort_values('object_id', inplace=True)

        bazin = pd.read_pickle(
            feature_data_dir /
            'training_set_bazin_features_kyle_final_augmented.pickle'
        )
        bazin.sort_values('object_id', inplace=True)
        assert np.all(np.round(bazin['object_id'].values, 2) ==
                      np.round(train_df['object_id'].values, 2))
        bazin['object_id'] = train_df['object_id'].values

        v1 = pd.read_pickle(
            feature_data_dir /
            'training_set_features_v1_kyle_final_augmented.pickle'
        )
        v1.sort_values('object_id', inplace=True)
        assert np.all(np.round(v1['object_id'].values, 2) ==
                      np.round(train_df['object_id'].values, 2))
        v1['object_id'] = train_df['object_id'].values

        v2 = pd.read_pickle(
            feature_data_dir /
            'training_set_features_v2_kyle_final_augmented.pickle'
        )
        v2.sort_values('object_id', inplace=True)
        assert np.all(np.round(v2['object_id'].values, 2) ==
                      np.round(train_df['object_id'].values, 2))
        v2['object_id'] = train_df['object_id'].values

        # GP補間の特徴量
        gp = pd.read_csv(
            feature_data_dir /
            'training_set_1st_bin1_2DGPR_FE_all_190110_1.csv',
            header=0
        )
        gp.sort_values('object_id', inplace=True)
        assert np.all(np.round(gp['object_id'].values, 2) ==
                      np.round(train_df['object_id'].values, 2))
        gp['object_id'] = train_df['object_id'].values

        for data in (bazin, v1, v2, gp):
            train_df = train_df.merge(data, on='object_id', how='left')
    else:
        train_df = pd.read_csv(
            meta_data_dir / 'training_set_metadata.csv', header=0
        )

        for feature_file in ["bazin_features", "features_v1", "features_v2"]:
            train_df = train_df.merge(
                pd.read_pickle(
                    feature_data_dir /
                    "training_set_{}.pickle".format(feature_file)
                ),
                on="object_id", how="left"
            )
        gp_features_path = (
                feature_data_dir /
                'training_set_1st_bin1_2DGPR_FE_all_190110_1.csv'
        )
        assert gp_features_path.exists()
        train_df = train_df.merge(
            pd.read_csv(gp_features_path, header=0),
            on="object_id", how="left"
        )

    train_gal = train_df[train_df["hostgal_photoz"] == 0].copy()
    train_exgal = train_df[train_df["hostgal_photoz"] > 0].copy()

    return train_gal, train_exgal


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L37
# Add new features
def load_test_data(feature_data_dir, meta_data_dir):
    test_df = pd.read_csv(
        meta_data_dir / 'test_set_metadata.csv', header=0
    )
    for feature_file in ["bazin_features", "features_v1", "features_v2"]:
        test_df = test_df.merge(
            pd.read_pickle(
                feature_data_dir /
                "test_set_{}.pickle".format(feature_file)
            ),
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

    test_gal = test_df[test_df["hostgal_photoz"] == 0].copy()
    test_exgal = test_df[test_df["hostgal_photoz"] > 0].copy()

    return test_gal, test_exgal, test_df


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/util.py#L37
# Modified for 2 step classification
def load_data(feature_data_dir, meta_data_dir, augmented):
    feature_data_dir = Path(feature_data_dir)
    meta_data_dir = Path(meta_data_dir)

    train_gal, train_exgal = load_training_data(
        feature_data_dir=feature_data_dir, meta_data_dir=meta_data_dir,
        augmented=augmented
    )
    test_gal, test_exgal, test_df = load_test_data(
        feature_data_dir=feature_data_dir, meta_data_dir=meta_data_dir
    )

    sw = SampleWeighter(train_exgal["hostgal_photoz"],
                        test_exgal["hostgal_photoz"])

    train_gal = sw.calculate_weights(train_gal, True)
    train_exgal = sw.calculate_weights(train_exgal, False)

    # 精度が悪いので67を追加、95は除外
    sn_class = (42, 52, 62, 67, 90)

    # 超新星のクラスを選択
    flag = np.zeros(len(train_exgal), dtype=np.bool)
    for c in sn_class:
        flag = np.logical_or(flag, train_exgal['target'] == c)
    train_exgal_sn = train_exgal[flag].copy()

    # 超新星を一つのクラスにまとめる
    train_exgal_all = train_exgal.copy()
    ex_class = np.unique(train_exgal_all['target']).astype(np.int)
    m = {c: c if c not in sn_class else 98 for c in ex_class}
    train_exgal_all['target'] = train_exgal_all['target'].map(m)

    train_gal, gal_class_list = map_classes(train_gal)
    train_exgal, exgal_class_list = map_classes(train_exgal)
    train_exgal_stage1, exgal_class_list_stage1 = map_classes(train_exgal_all)
    train_exgal_stage2, exgal_class_list_stage2 = map_classes(train_exgal_sn)

    r = (train_gal, train_exgal, train_exgal_stage1, train_exgal_stage2,
         test_gal, test_exgal, gal_class_list, exgal_class_list,
         exgal_class_list_stage1, exgal_class_list_stage2,
         test_df[["object_id", "hostgal_photoz"]])
    return r


# The original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L21
# training with all data
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

    results['importance'] = None
    results['current_iteration'] = None

    return test_preds, results


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L41
# This function processes galactic part.
def get_lgb_predictions_galactic(train_gal, test_gal, gal_class_list,
                                 features_gal, augmented, use_all):
    params_gal = {
        "objective": "multiclass", "num_class": len(gal_class_list),
        "min_data_in_leaf": 200, "num_leaves": 5, "feature_fraction": 0.7
    }
    print("GALACTIC MODEL")
    print('galactic: {}'.format(len(features_gal)))
    if use_all:
        test_preds_gal, results_gal = train_and_predict_all(
            train_df=train_gal, test_df=test_gal, features=features_gal,
            params=params_gal, n_iterations=150
        )

        return test_preds_gal, results_gal
    else:
        oof_preds_gal, test_preds_gal, results_gal = train_and_predict(
            train_gal, test_gal, features_gal, params_gal, exgal=False,
            n_neighbors=None, standard_scaler=False,
            distance_scale='random', n_jobs=-1,
            feature_augmentation=False, augmented=augmented,
            weight_scale=None, nn_dir=None, sn_only=False
        )

        return oof_preds_gal, test_preds_gal, results_gal


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L41
# This function processes extra galactic part.
# The classification task consists 2 step processes.
# The hyper parameters of Light GBM are optimized.
def get_lgb_predictions_extra_galactic(
        train_exgal_stage1, train_exgal_stage2, train_exgal, test_exgal,
        exgal_class_list_stage1, exgal_class_list_stage2, exgal_class_list,
        features_exgal, augmented, features_exgal_stage1,
        features_exgal_stage2, use_all):
    if features_exgal_stage1 is None:
        features_exgal_stage1 = features_exgal
    if features_exgal_stage2 is None:
        features_exgal_stage2 = features_exgal

    params_exgal_state1 = {
        "objective": "multiclass", "num_class": len(exgal_class_list_stage1),
        "bagging_fraction": 0.3449433546210914,
        "bagging_freq": 1,
        "boosting": "gbdt",
        "feature_fraction": 0.30541823648351074,
        "lambda_l1": 0.007670156255141235,
        "lambda_l2": 0.17760897231258513,
        "max_bin": 33,
        "min_data_in_leaf": 167,
        "min_gain_to_split": 0.00018320992685689547,
        "min_sum_hessian_in_leaf": 0.0006129085010632326,
        "num_leaves": 39
    }
    params_exgal_state2 = params_exgal_state1.copy()
    params_exgal_state2['num_class'] = len(exgal_class_list_stage2)
    weight_scale = (2.234986962900447, 1.8472313855496796, 1.3372275699359146,
                    1.1017368580861138, 1.0)

    print("EXTRAGALACTIC MODEL")
    print('galactic: {}'.format(len(features_exgal)))
    if use_all:
        (test_preds_exgal_state1,
         results_exgal_stage1) = train_and_predict_all(
            train_exgal_stage1, test_exgal, features_exgal_stage1,
            params_exgal_state1, n_iterations=160
        )
        (test_preds_exgal_state2,
         results_exgal_stage2) = train_and_predict_all(
            train_exgal_stage2, test_exgal, features_exgal_stage2,
            params_exgal_state2, n_iterations=180
        )

        (test_preds_exgal_state1, test_preds_exgal_state2,
         test_preds_exgal) = merge_test_exgal(
            test_preds_exgal_state1=test_preds_exgal_state1,
            test_preds_exgal_state2=test_preds_exgal_state2,
            test_exgal=test_exgal,
            exgal_class_list_stage1=exgal_class_list_stage1,
            exgal_class_list_stage2=exgal_class_list_stage2,
            exgal_class_list=exgal_class_list
        )

        r = (test_preds_exgal_state1, test_preds_exgal_state2,
             test_preds_exgal, results_exgal_stage1, results_exgal_stage2)
        return r
    else:
        (oof_preds_exgal_state1, test_preds_exgal_state1,
         results_exgal_stage1) = train_and_predict(
            train_exgal_stage1, test_exgal, features_exgal_stage1,
            params_exgal_state1,
            exgal=True, n_neighbors=None, standard_scaler=False,
            distance_scale='random', n_jobs=-1, feature_augmentation=False,
            augmented=augmented, weight_scale=weight_scale, nn_dir=Path('.'),
            sn_only=False
        )
        (oof_preds_exgal_state2, test_preds_exgal_state2,
         results_exgal_stage2) = train_and_predict(
            train_exgal_stage2, test_exgal, features_exgal_stage2,
            params_exgal_state2,
            exgal=True, n_neighbors=None, standard_scaler=False,
            distance_scale='random', n_jobs=-1, feature_augmentation=False,
            augmented=augmented, weight_scale=weight_scale, nn_dir=Path('.'),
            sn_only=True
        )

        (oof_preds_exgal_state1, oof_preds_exgal_state2,
         oof_preds_exgal) = merge_oof_exgal(
            oof_preds_exgal_state1=oof_preds_exgal_state1,
            oof_preds_exgal_state2=oof_preds_exgal_state2,
            train_exgal_stage1=train_exgal_stage1,
            train_exgal_stage2=train_exgal_stage2,
            train_exgal=train_exgal,
            exgal_class_list_stage1=exgal_class_list_stage1,
            exgal_class_list_stage2=exgal_class_list_stage2,
            exgal_class_list=exgal_class_list
        )

        (test_preds_exgal_state1, test_preds_exgal_state2,
         test_preds_exgal) = merge_test_exgal(
            test_preds_exgal_state1=test_preds_exgal_state1,
            test_preds_exgal_state2=test_preds_exgal_state2,
            test_exgal=test_exgal,
            exgal_class_list_stage1=exgal_class_list_stage1,
            exgal_class_list_stage2=exgal_class_list_stage2,
            exgal_class_list=exgal_class_list
        )

        r = (oof_preds_exgal_state1, oof_preds_exgal_state2, oof_preds_exgal,
             test_preds_exgal_state1, test_preds_exgal_state2,
             test_preds_exgal, results_exgal_stage1, results_exgal_stage2)
        return r


def merge_oof_exgal(oof_preds_exgal_state1, oof_preds_exgal_state2,
                    train_exgal_stage1, train_exgal_stage2, train_exgal,
                    exgal_class_list_stage1, exgal_class_list_stage2,
                    exgal_class_list):
    oof_preds_exgal_state1 = pd.DataFrame(
        oof_preds_exgal_state1, index=train_exgal_stage1['object_id'],
        columns=exgal_class_list_stage1
    )
    oof_preds_exgal_state2 = pd.DataFrame(
        oof_preds_exgal_state2, index=train_exgal_stage2['object_id'],
        columns=exgal_class_list_stage2
    )

    oof_preds_exgal = pd.DataFrame(
        np.empty((len(train_exgal), len(exgal_class_list))),
        index=train_exgal['object_id'], columns=exgal_class_list
    )
    print('merging oof predictions')
    n = len(exgal_class_list_stage2)
    tmp = train_exgal_stage1.set_index('object_id')
    for object_id in tqdm(train_exgal['object_id'].values):
        target = exgal_class_list_stage1[tmp.loc[object_id, 'target']]
        for c1 in exgal_class_list_stage1:
            v1 = oof_preds_exgal_state1.loc[object_id, c1]
            if c1 == 98:
                if target == 98:
                    # stage2がある
                    for c2 in exgal_class_list_stage2:
                        v2 = oof_preds_exgal_state2.loc[object_id, c2]
                        oof_preds_exgal.loc[object_id, c2] = v1 * v2
                else:
                    # stage2がないので、一様に振り分ける
                    for c2 in exgal_class_list_stage2:
                        oof_preds_exgal.loc[object_id, c2] = v1 / n
            else:
                oof_preds_exgal.loc[object_id, c1] = v1

    return oof_preds_exgal_state1, oof_preds_exgal_state2, oof_preds_exgal


def merge_test_exgal(test_preds_exgal_state1, test_preds_exgal_state2,
                     test_exgal, exgal_class_list_stage1,
                     exgal_class_list_stage2, exgal_class_list):
    test_preds_exgal_state1 = pd.DataFrame(
        test_preds_exgal_state1, index=test_exgal['object_id'],
        columns=exgal_class_list_stage1
    )
    test_preds_exgal_state2 = pd.DataFrame(
        test_preds_exgal_state2, index=test_exgal['object_id'],
        columns=exgal_class_list_stage2
    )

    test_preds_exgal = test_preds_exgal_state1.copy()
    p = test_preds_exgal_state1[98]
    for c in exgal_class_list_stage2:
        test_preds_exgal[c] = p * test_preds_exgal_state2[c]
    # 並び替え
    test_preds_exgal = test_preds_exgal[exgal_class_list]

    return test_preds_exgal_state1, test_preds_exgal_state2, test_preds_exgal


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L11
# Modified to output the result as json file
def evaluate(train_gal, train_exgal, train_exgal_stage1, train_exgal_stage2,
             oof_preds_gal, oof_preds_exgal, oof_preds_exgal_state1,
             oof_preds_exgal_state2, model_dir):
    gal_loss = log_loss(train_gal["target"], np.round(oof_preds_gal, 4),
                        sample_weight=train_gal["sample_weight"])
    exgal_loss = log_loss(train_exgal["target"], np.round(oof_preds_exgal, 4),
                          sample_weight=train_exgal["sample_weight"])
    print("Galactic CV: {}".format(gal_loss))
    print("Extragalactic CV: {}".format(exgal_loss))
    total = (5 / 16) * gal_loss + (11 / 16) * exgal_loss
    print("Overall CV: {}".format(total))

    exgal_loss_stage1 = log_loss(
        train_exgal_stage1['target'], np.round(oof_preds_exgal_state1, 4),
        sample_weight=train_exgal_stage1['sample_weight']
    )
    exgal_loss_stage2 = log_loss(
        train_exgal_stage2['target'], np.round(oof_preds_exgal_state2, 4),
        sample_weight=train_exgal_stage2['sample_weight']
    )
    print("Extragalactic CV (stage1): {}".format(exgal_loss_stage1))
    print("Extragalactic CV (stage2): {}".format(exgal_loss_stage2))

    d = {'galactic_log_loss': float(gal_loss),
         'extra_galactic_log_loss': float(exgal_loss),
         'log_loss': float(total),
         'extra_galactic_log_loss_stage1': float(exgal_loss_stage1),
         'extra_galactic_log_loss_stage2': float(exgal_loss_stage2)}
    with (model_dir / 'result.json').open('w') as f:
        json.dump(d, f, indent=4, sort_keys=True)


# the original code is
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L41
# Add new features, remove features which do not exist
def get_lgb_predictions(train_gal, train_exgal_stage1, train_exgal_stage2,
                        train_exgal, test_gal, test_exgal, gal_class_list,
                        exgal_class_list_stage1, exgal_class_list_stage2,
                        exgal_class_list, model_dir, augmented,
                        exgal_feature_size1, exgal_feature_size2, use_all):
    (features_gal, features_exgal,
     features_exgal_state1, features_exgal_state2) = get_feature_names()

    if 0 < exgal_feature_size1 < len(features_exgal):
        features_exgal_state1 = features_exgal_state1[:exgal_feature_size1]
    if 0 < exgal_feature_size2 < len(features_exgal):
        features_exgal_state2 = features_exgal_state2[:exgal_feature_size2]

    if use_all:
        (test_preds_exgal_state1, test_preds_exgal_state2, test_preds_exgal,
         results_exgal_state1, results_exgal_state2
         ) = get_lgb_predictions_extra_galactic(
            train_exgal_stage1=train_exgal_stage1,
            train_exgal_stage2=train_exgal_stage2,
            train_exgal=train_exgal,
            test_exgal=test_exgal,
            exgal_class_list_stage1=exgal_class_list_stage1,
            exgal_class_list_stage2=exgal_class_list_stage2,
            exgal_class_list=exgal_class_list,
            features_exgal=features_exgal,
            augmented=augmented,
            features_exgal_stage1=features_exgal_state1,
            features_exgal_stage2=features_exgal_state2,
            use_all=use_all
        )
        test_preds_gal, results_gal = get_lgb_predictions_galactic(
            train_gal=train_gal, test_gal=test_gal,
            gal_class_list=gal_class_list,
            features_gal=features_gal, augmented=augmented,
            use_all=use_all
        )

        return test_preds_gal, test_preds_exgal
    else:
        (oof_preds_exgal_state1, oof_preds_exgal_state2, oof_preds_exgal,
         test_preds_exgal_state1, test_preds_exgal_state2, test_preds_exgal,
         results_exgal_state1, results_exgal_state2
         ) = get_lgb_predictions_extra_galactic(
            train_exgal_stage1=train_exgal_stage1,
            train_exgal_stage2=train_exgal_stage2,
            train_exgal=train_exgal,
            test_exgal=test_exgal,
            exgal_class_list_stage1=exgal_class_list_stage1,
            exgal_class_list_stage2=exgal_class_list_stage2,
            exgal_class_list=exgal_class_list,
            features_exgal=features_exgal,
            augmented=augmented,
            features_exgal_stage1=features_exgal_state1,
            features_exgal_stage2=features_exgal_state2,
            use_all=use_all
        )
        (oof_preds_gal, test_preds_gal,
         results_gal) = get_lgb_predictions_galactic(
            train_gal=train_gal, test_gal=test_gal,
            gal_class_list=gal_class_list,
            features_gal=features_gal, augmented=augmented,
            use_all=use_all
        )

        evaluate(train_gal=train_gal,
                 train_exgal=train_exgal,
                 train_exgal_stage1=train_exgal_stage1,
                 train_exgal_stage2=train_exgal_stage2,
                 oof_preds_gal=oof_preds_gal,
                 oof_preds_exgal=oof_preds_exgal.values,
                 oof_preds_exgal_state1=oof_preds_exgal_state1.values,
                 oof_preds_exgal_state2=oof_preds_exgal_state2.values,
                 model_dir=model_dir)

        output_confusion_matrix(
            results=results_gal, class_list=gal_class_list,
            path=model_dir / 'train_confusion_matrix_galactic.png'
        )
        output_confusion_matrix(
            results=results_exgal_state1, class_list=exgal_class_list_stage1,
            path=model_dir / 'train_confusion_matrix_extra_galactic_stage1.png'
        )
        output_confusion_matrix(
            results=results_exgal_state2, class_list=exgal_class_list_stage2,
            path=model_dir / 'train_confusion_matrix_extra_galactic_stage2.png'
        )
        del results_gal['train_predictions']
        del results_exgal_state1['train_predictions']
        del results_exgal_state2['train_predictions']

        results_gal['importance'].to_csv(model_dir / 'importance_galactic.csv')
        results_exgal_state1['importance'].to_csv(
            model_dir / 'importance_extra_galactic_stage1.csv'
        )
        results_exgal_state2['importance'].to_csv(
            model_dir / 'importance_extra_galactic_stage2.csv'
        )
        del results_gal['importance']
        del results_exgal_state1['importance']
        del results_exgal_state2['importance']

        results = {'galactic': results_gal,
                   'extra_galactic_stage1': results_exgal_state1,
                   'extra_galactic_stage2': results_exgal_state2}
        with (model_dir / 'cv_result.json').open('w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

        r = (oof_preds_gal, oof_preds_exgal, oof_preds_exgal_state1,
             oof_preds_exgal_state2, test_preds_gal, test_preds_exgal)
        return r


def output_confusion_matrix(results, class_list, path):
    data, label = [], []
    for p, t in results['train_predictions']:
        data.append(p)
        label.append(t)
    data = np.vstack(data)
    label = np.hstack(label)
    draw_confusion_matrix(
        target=label, prediction=data, class_list=class_list, path=path
    )


# This function is derived from
# https://github.com/aerdem4/kaggle-plasticc/blob/master/models/lgb.py#L85
# Some running options are added.
# The predicted values are saved for stacking.
@click.command()
@click.option('--feature-dir', type=click.Path(exists=True),
              default='../data/processed/4th')
@click.option('--meta-dir', type=click.Path(exists=True),
              default='../data/raw')
@click.option('--model-dir', type=click.Path())
@click.option('--augmented', is_flag=True)
@click.option('--exgal-feature-size1', type=int, default=-1)
@click.option('--exgal-feature-size2', type=int, default=-1)
@click.option('--use-all', is_flag=True)
@click.option('--use-meta-prediction', is_flag=True)
def cmd(feature_dir, meta_dir, model_dir, augmented, exgal_feature_size1,
        exgal_feature_size2, use_all, use_meta_prediction):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    with (model_dir / 'parameters.json').open('w') as f:
        parameters = {
            'feature_dir': feature_dir, 'meta_dir': meta_dir,
            'exgal_feature_size1': exgal_feature_size1,
            'exgal_feature_size2': exgal_feature_size2,
            'use_all': use_all, 'use_meta_prediction': use_meta_prediction
        }
        json.dump(parameters, f, indent=4, sort_keys=True)

    (train_gal, train_exgal, train_exgal_stage1, train_exgal_stage2,
     test_gal, test_exgal, gal_class_list, exgal_class_list,
     exgal_class_list_stage1, exgal_class_list_stage2, test_df) = load_data(
        feature_data_dir=feature_dir, meta_data_dir=meta_dir,
        augmented=augmented
    )
    if use_all:
        test_preds_gal, test_preds_exgal = get_lgb_predictions(
            train_gal=train_gal, train_exgal=train_exgal,
            train_exgal_stage1=train_exgal_stage1,
            train_exgal_stage2=train_exgal_stage2,
            test_gal=test_gal, test_exgal=test_exgal,
            gal_class_list=gal_class_list, exgal_class_list=exgal_class_list,
            model_dir=model_dir, augmented=augmented,
            exgal_class_list_stage1=exgal_class_list_stage1,
            exgal_class_list_stage2=exgal_class_list_stage2,
            exgal_feature_size1=exgal_feature_size1,
            exgal_feature_size2=exgal_feature_size2, use_all=use_all
        )

        # 2段階目用
        test_gal_raw = pd.DataFrame(
            test_preds_gal,
            index=test_df.loc[test_df['hostgal_photoz'] == 0, 'object_id'],
            columns=gal_class_list
        )
        test_exgal_raw = pd.DataFrame(
            test_preds_exgal.values,
            index=test_df.loc[test_df['hostgal_photoz'] > 0, 'object_id'],
            columns=exgal_class_list
        )
    else:
        (oof_preds_gal, oof_preds_exgal, oof_preds_exgal_stage1,
         oof_preds_exgal_stage2, test_preds_gal, test_preds_exgal
         ) = get_lgb_predictions(
            train_gal=train_gal, train_exgal=train_exgal,
            train_exgal_stage1=train_exgal_stage1,
            train_exgal_stage2=train_exgal_stage2,
            test_gal=test_gal, test_exgal=test_exgal,
            gal_class_list=gal_class_list, exgal_class_list=exgal_class_list,
            model_dir=model_dir, augmented=augmented,
            exgal_class_list_stage1=exgal_class_list_stage1,
            exgal_class_list_stage2=exgal_class_list_stage2,
            exgal_feature_size1=exgal_feature_size1,
            exgal_feature_size2=exgal_feature_size2, use_all=use_all
        )

        # 2段階目用
        test_gal_raw = pd.DataFrame(
            test_preds_gal,
            index=test_df.loc[test_df['hostgal_photoz'] == 0, 'object_id'],
            columns=gal_class_list
        )
        test_exgal_raw = pd.DataFrame(
            test_preds_exgal.values,
            index=test_df.loc[test_df['hostgal_photoz'] > 0, 'object_id'],
            columns=exgal_class_list
        )
        oof_preds_gal = pd.DataFrame(
            oof_preds_gal, index=train_gal['object_id'], columns=gal_class_list
        )

        if use_meta_prediction:
            test_preds_gal = get_meta_preds(
                train_gal, oof_preds_gal.values, test_preds_gal, 0.2
            )
            test_preds_exgal = get_meta_preds(
                train_exgal, oof_preds_exgal.values,
                test_preds_exgal.values, 0.2
            )
        else:
            test_preds_exgal = test_preds_exgal.values

    submit(test_df, test_preds_gal, test_preds_exgal,
           gal_class_list, exgal_class_list,
           str(model_dir / "submission_lgb.csv"))

    if use_all:
        tmp = {
            'galactic': {
                'target': train_gal['target'],
                'class_list': gal_class_list
            },
            'extra_galactic': {
                'target': train_exgal['target'],
                'class_list': exgal_class_list
            },
            'extra_galactic_stage1': {
                'target': train_exgal_stage1['target'],
                'class_list': exgal_class_list_stage1
            },
            'extra_galactic_stage2': {
                'target': train_exgal_stage2['target'],
                'class_list': exgal_class_list_stage2
            },
            'test_galactic': test_gal_raw,
            'test_extra_galactic': test_exgal_raw
        }
    else:
        # noinspection PyUnboundLocalVariable
        tmp = {
            'galactic': {
                'target': train_gal['target'], 'prediction': oof_preds_gal,
                'class_list': gal_class_list
            },
            'extra_galactic': {
                'target': train_exgal['target'], 'prediction': oof_preds_exgal,
                'class_list': exgal_class_list
            },
            'extra_galactic_stage1': {
                'target': train_exgal_stage1['target'],
                'prediction': oof_preds_exgal_stage1,
                'class_list': exgal_class_list_stage1
            },
            'extra_galactic_stage2': {
                'target': train_exgal_stage2['target'],
                'prediction': oof_preds_exgal_stage2,
                'class_list': exgal_class_list_stage2
            },
            'test_galactic': test_gal_raw,
            'test_extra_galactic': test_exgal_raw
        }
    joblib.dump(tmp, str(model_dir / 'prediction.pickle'))

    # tmp = joblib.load(str(model_dir / 'prediction.pickle'))
    #
    # train_gal = tmp['galactic']
    # oof_preds_gal = tmp['galactic']['prediction']
    # gal_class_list = tmp['galactic']['class_list']
    #
    # train_exgal = tmp['extra_galactic']
    # oof_preds_exgal = tmp['extra_galactic']['prediction']
    # exgal_class_list = tmp['extra_galactic']['class_list']
    #
    # train_exgal_stage1 = tmp['extra_galactic_stage1']
    # oof_preds_exgal_stage1 = tmp['extra_galactic_stage1']['prediction']
    # exgal_class_list_stage1 = tmp['extra_galactic_stage1']['class_list']
    #
    # train_exgal_stage2 = tmp['extra_galactic_stage2']
    # oof_preds_exgal_stage2 = tmp['extra_galactic_stage2']['prediction']
    # exgal_class_list_stage2 = tmp['extra_galactic_stage2']['class_list']

    if not use_all:
        draw_confusion_matrix(
            target=train_gal['target'].values,
            prediction=oof_preds_gal.values,
            class_list=gal_class_list,
            path=model_dir / 'confusion_matrix_galactic.png'
        )
        draw_confusion_matrix(
            target=train_exgal['target'].values,
            prediction=oof_preds_exgal.values,
            class_list=exgal_class_list,
            path=model_dir / 'confusion_matrix_extra_galactic.png'
        )
        draw_confusion_matrix(
            target=train_exgal_stage1['target'].values,
            prediction=oof_preds_exgal_stage1.values,
            class_list=exgal_class_list_stage1,
            path=model_dir / 'confusion_matrix_extra_galactic_stage1.png'
        )
        draw_confusion_matrix(
            target=train_exgal_stage2['target'].values,
            prediction=oof_preds_exgal_stage2.values,
            class_list=exgal_class_list_stage2,
            path=model_dir / 'confusion_matrix_extra_galactic_stage2.png'
        )


def main():
    cmd()


if __name__ == '__main__':
    main()
