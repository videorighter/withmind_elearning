from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_squared_error
from dataset.train_test import get_multimodal_mean_var, get_multimodal_ci_five, get_multimodal_mean
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

def mean_absolute_percentage_error(y_pred, y_true):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(abs((y_true - y_pred)/y_true)) * 100

def li_svr_five(path):

    for _ in range(20):
        X_train, X_test, y_train, y_test = get_multimodal_ci_five(path)

        y_train = y_train.replace([1,2,3,4,5],[70,76,82,88,94])
        y_test = y_test.replace([1,2,3,4,5],[70,76,82,88,94])

        scaler = StandardScaler()
        svr = make_pipeline(StandardScaler(),
                            LinearSVR(epsilon=0.5, tol=1e-5))
        svr.fit(scaler.fit_transform(X_train), y_train)
        y_pred = svr.predict(scaler.fit_transform(X_test))

        print("mse: ", mean_squared_error(y_pred, y_test))
        print("mape: ", mean_absolute_percentage_error(y_pred, y_test))


if __name__ == "__main__":
    li_svr_five("./dataset/condition_indicators_dataset_five.csv")