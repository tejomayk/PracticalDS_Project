import pandas as pd


def prepare_data(df):

    df.dropna(inplace=True)
    datetime_columns = [
        'INCIDENT_DATETIME',
        'INCIDENT_CLOSE_DATETIME'
    ]

    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y %I:%M:%S %p")

    df['hour_of_day'] = df['INCIDENT_DATETIME'].dt.hour
    df['day_of_week'] = df['INCIDENT_DATETIME'].dt.dayofweek
    df['month'] = df['INCIDENT_DATETIME'].dt.month

    df['total_incident_duration'] = (
        df['INCIDENT_CLOSE_DATETIME'] - df['INCIDENT_DATETIME']
    ).dt.total_seconds() / 60

    df = df[df['INCIDENT_RESPONSE_SECONDS_QY'] <= 600]
    df = df[df['INCIDENT_RESPONSE_SECONDS_QY'] > 0]

    feature_columns = [
        'INITIAL_SEVERITY_LEVEL_CODE',
        'FINAL_SEVERITY_LEVEL_CODE',
        'DISPATCH_RESPONSE_SECONDS_QY',
        'INITIAL_CALL_TYPE',
        'FINAL_CALL_TYPE',
        'hour_of_day',
        'day_of_week',
        'month',
        'BOROUGH',
        'ZIPCODE',
        'POLICEPRECINCT',
        'HELD_INDICATOR',
        'SPECIAL_EVENT_INDICATOR',
        'STANDBY_INDICATOR',
        'TRANSFER_INDICATOR',
        'INCIDENT_DISPATCH_AREA',
        'CONGRESSIONALDISTRICT',
        'CITYCOUNCILDISTRICT'
    ]

    target = 'INCIDENT_RESPONSE_SECONDS_QY'

    return df[feature_columns], df[target]
