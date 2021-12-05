'''
bagging algorithm
'''

# -*- coding: utf-8 -*-
import numpy as np
import keras

'''
X = feature dataset
y = label dataset
ratio = bootstrap 비율
'''
def create_bootstrap(X, y, ratio):

    ind = np.random.choice(len(X), replace=True, size=int(len(X)*ratio))

    return X[ind, :], y[ind]

'''
X = feature dataset
y = label dataset
ratio = bootstrap 비율
mlp_batch = multi-layer perceptron batch size
mlp_epoch = multi-layer perceptron epoch size
'''
def bagging_cls(X, y, model_list, bootstrap_n, ratio, mlp_batch, mlp_epoch):

    bagging_model_list = []

    for model in model_list:
        for _ in range(bootstrap_n):
            newX, newy = create_bootstrap(X, y, ratio)
            if isinstance(model, keras.engine.sequential.Sequential):
                _ = model.fit(newX, newy, mlp_batch, mlp_epoch)
                bagging_model_list.append(model)
            else:
                bagging_model_list.append(model.fit(newX, newy))

    return bagging_model_list


def voting(bagging_model_list, X):

    y_array = np.zeros((len(X), len(bagging_model_list)))

    for i, model in enumerate(bagging_model_list):
        if isinstance(model, keras.engine.sequential.Sequential):
            y_array[:, i] = model.predict_classes(X).flatten()
        else:
            y_array[:, i] = model.predict(X).flatten()

    unique, row_count = np.unique(y_array, return_inverse=True)
    row_count = row_count.reshape(-1, len(bagging_model_list))

    voting = np.apply_along_axis(
        func1d=lambda x: np.bincount(x, minlength=len(bagging_model_list)),
        axis=1,
        arr=row_count.astype(int)).argmax(axis=1)

    voting_to_class = np.apply_along_axis(func1d=lambda x: unique[x],
                                          axis=1,
                                          arr=voting.reshape(-1, 1))

    return voting_to_class.reshape(-1)