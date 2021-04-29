from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dataset.train_test import get_multimodal_mean
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

def mean_absolute_percentage_error(y_pred, y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(abs((y_true - y_pred)/y_true)) * 100

def linear_regression_mean(path):

    for _ in range(20):
        X_train, X_test, y_train, y_test = get_multimodal_mean()

        y_train = y_train.replace([1,2,3,4,5],[70,76,82,88,94])
        y_test = y_test.replace([1,2,3,4,5],[70,76,82,88,94])

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        dataset = pd.read_csv(path)
        dataset = dataset.replace(['inf', '-inf'], np.nan)
        dataset = dataset.dropna(how="any")
        model = smf.ols(formula="label ~ Angry_avr+Fear_avr+Happy_avr+Normal_avr+Sadness_avr+Surprise_avr+BodyMoving_avr+pitch_avr+yaw_avr+HandOcclusion_avr", data=dataset)
        result = model.fit()
        print(result.summary())

        print("mse: ", mean_squared_error(y_pred, y_test))
        print("mape: ", mean_absolute_percentage_error(y_pred, y_test))
        print(smf)

linear_regression_mean("./dataset/mean_dataset.csv")