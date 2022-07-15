import pandas as pd


def get_attribute_map(key, attribute):
    return key[key['Attribute'] == attribute][['Code', 'Label']].set_index('Code').to_dict()['Label']


def add_descriptions_from_key(df, key):
    descriptor_df = df.copy()
    key['Attribute'] = key['Attribute'].str.lower()
    for attribute in key['Attribute'].unique():
        attribute_map = get_attribute_map(key, attribute)
        descriptor_df[attribute] = df[attribute].map(attribute_map)
    return descriptor_df


def with_prediction_label(df):
    df['label'] = 0
    df.loc[df['did_police_officer_attend_scene_of_accident'] == 1, 'label'] = 1
    return df


def with_date_time(df):
    df['date_time'] = pd.to_datetime(df['date'])
    return df


def with_hour_of_day(df):
    df['hour_of_day'] = df['date_time'].dt.hour
    return df


def with_month_of_year(df):
    df['month_of_year'] = df['date_time'].dt.month
    return df


def preprocess_df(df):
    df = with_prediction_label(df)
    df = with_date_time(df)
    df = with_hour_of_day(df)
    df = with_month_of_year(df)

    return df


def preprocess_key(key):
    key = key[key['Dataset'] == 'accidents']
    key.loc[:, 'Attribute'] = key['Attribute'].str.lower()
    clean_mapping = {'local_authority_(district)': 'local_authority_district',
                     'local_authority_(highway)': 'local_authority_highway',
                     'pedestrian_crossing-human_control': 'pedestrian_crossing_human_control',
                     'pedestrian_crossing-physical_facilities': 'pedestrian_crossing_physical_facilities',
                     '1st_road_class': 'first_road_class',
                     '2nd_road_class': 'second_road_class'}
    key.loc[:, 'Attribute'] = key['Attribute'].replace(clean_mapping)
    return key


def preprocess_df_and_key(df, key):
    df = preprocess_df(df)
    key = preprocess_key(key)
    return df, key
