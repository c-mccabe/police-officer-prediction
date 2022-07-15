import pandas as pd
import numpy as np
import math


def get_aggregated_metrics_by_feature(df, feature):
    visits_df = df.groupby(feature) \
        .agg({'label': 'sum'}) \
        .reset_index() \
        .rename(columns={'label': f'{feature}_attendance'})

    accidents_df = df.groupby(feature) \
        .agg({'label': 'count'}) \
        .reset_index() \
        .rename(columns={'label': f'{feature}_accidents'})

    out_df = pd.merge(visits_df, accidents_df, on=feature)
    out_df[f'{feature}_attendance_rate'] = out_df[f'{feature}_attendance'] / out_df[f'{feature}_accidents']

    return out_df


def with_aggregated_metrics_by_feature(df, feature_df, feature):
    metrics_df = get_aggregated_metrics_by_feature(feature_df, feature)
    return pd.merge(df, metrics_df, on=feature, how='left')


def transform_seasonality_features(df, feature, period):
    df[f'transformed_{feature}'] = np.cos(2 * math.pi * df[feature] / period)
    return df


def with_is_rural(df):
    df['is_rural'] = df['urban_or_rural_area'] == 2
    return df


def with_is_severe(df):
    df['is_severe'] = np.isin(df['accident_severity'], [1, 2])
    return df


def with_is_weekend(df):
    df['is_weekend'] = np.isin(df['day_of_week'], [1, 7])
    return df


def train_test_split(df):
    train_df = df[df['date_time'] < '2020-11-01']
    test_df = df[df['date_time'] >= '2020-11-01']

    return train_df, test_df


def create_training_datasets(df):
    df = transform_seasonality_features(df, feature='hour_of_day', period=24)
    df = with_is_rural(df)
    df = with_is_severe(df)
    df = with_is_weekend(df)

    train_df, test_df = train_test_split(df)

    train_df = with_aggregated_metrics_by_feature(train_df, train_df, 'police_force')
    test_df = with_aggregated_metrics_by_feature(test_df, train_df, 'police_force')

    return train_df, test_df
