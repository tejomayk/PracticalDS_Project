from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# from torch_regressor import TorchRegressor


def create_model_pipeline():

    numeric_features = [
        'INITIAL_SEVERITY_LEVEL_CODE',
        'FINAL_SEVERITY_LEVEL_CODE',
        'hour_of_day',
        'day_of_week',
        'month',
        'ZIPCODE',
        'POLICEPRECINCT'
    ]

    categorical_features = [
        'BOROUGH',
        'HELD_INDICATOR',
        'SPECIAL_EVENT_INDICATOR',
        'STANDBY_INDICATOR',
        'TRANSFER_INDICATOR'
    ]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100))
    ])

    return model_pipeline
