import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import prepare_data
from pipeline import create_model_pipeline
from sklearn.model_selection import train_test_split


def train_and_evaluate_model(df):

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = create_model_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return pipeline, {
        'RMSE': rmse,
        'R2': r2,
        'Test_Size': len(y_test)
    }
