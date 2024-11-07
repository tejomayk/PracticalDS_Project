from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
from torch_regressor import TorchRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import RadiusNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor


def create_model_pipeline():

    numerical_features = [
        'DISPATCH_RESPONSE_SECONDS_QY',
        'INCIDENT_TRAVEL_TM_SECONDS_QY'
    ]

    nominal_features = [
        'SPECIAL_EVENT_INDICATOR',
        'STANDBY_INDICATOR',
        'TRANSFER_INDICATOR',
        'INITIAL_CALL_TYPE',
        'FINAL_CALL_TYPE',
        'HELD_INDICATOR',
        'BOROUGH',
        'INCIDENT_DISPATCH_AREA',
        'CONGRESSIONALDISTRICT',
        'CITYCOUNCILDISTRICT',

    ]

    ordinal_features = [
        'INITIAL_SEVERITY_LEVEL_CODE',
        'FINAL_SEVERITY_LEVEL_CODE',
        'hour_of_day',
        'day_of_week',
        'ZIPCODE',
        'POLICEPRECINCT',
        'month'
    ]

    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('num', numerical_transformer, numerical_features)
        ])

    pca = Pipeline(steps=[
        ('pca', PCA(n_components=10))
    ])

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', pca),
        ('regressor', TorchRegressor())
    ])

    return model_pipeline
