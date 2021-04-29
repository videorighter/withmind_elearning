'''
앙상블 모델
multi-layer perceptron, random forest, support vector machine 세 모델 중 정확도가 가장 높은 모델을 자동선정하여 결과 도출
'''


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from model.bagging import bagging_cls, voting
from dataset.train_test import get_multimodal_ci_remove_34_over, get_multimodal_ci_remove_234_over
import pickle
import time


def mlp(units):
    model = Sequential()
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu', input_dim=96))
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def rf():
    params = {'n_estimators': [50],
              'max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              'min_samples_leaf': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              'min_samples_split': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
              }

    rf_clf = RandomForestClassifier(n_estimators=50, random_state=1)
    grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=5, n_jobs=-1)

    return grid_cv

def svm():
    svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("linear_svc", LinearSVC(C=0.1, loss='hinge', random_state=42))])

    return svm_clf

def ensemble(model_list, dataset):


    X_train, X_test, y_train, y_test = dataset
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    model_list = model_list

    bagging_model_list = bagging_cls(X_train, y_train.values, model_list,
                                     bootstrap_n=5, ratio=0.8,
                                     mlp_batch=256, mlp_epoch=50)

    pred = voting(bagging_model_list, X_test)
    cm = confusion_matrix(y_test, pred)

    print(model_list)
    print(f"confusion matrix: \n"
          f"{cm}")
    print(classification_report(y_test.values, pred,
                                target_names=['Boring', 'Interesting'],
                                digits=4))


    # 결과로 나온 모델을 bin 파일로 저장

    with open("bagging_model_file.bin", "wb") as f:
        pickle.dump(bagging_model_list, f)

    return bagging_model_list

# mlp_clf = mlp(192)
# rf_clf = rf()
# sv_clf = svm()


# start = time.time()
# ensemble([mlp_clf, rf_clf, sv_clf], get_multimodal_ci_remove_34_over("./dataset/condition_indicators_dataset_remove_34_over2.csv"))
# print(f"time: {time.time() - start}")
# start = time.time()
# ensemble([mlp_clf, rf_clf, sv_clf], get_multimodal_ci_remove_234_over("./dataset/condition_indicators_dataset_remove_234_over2.csv"))
# print(f"time: {time.time() - start}")


'''
ensemble([mlp_clf, rf_clf, sv_clf], get_multimodal_ci_remove_34_over())
mlp_batch=256, mlp_epoch=50
confusion matrix: 
[[514  99]
 [ 93 764]]
              precision    recall  f1-score   support

      Boring     0.8468    0.8385    0.8426       613
 Interesting     0.8853    0.8915    0.8884       857

    accuracy                         0.8694      1470
   macro avg     0.8660    0.8650    0.8655      1470
weighted avg     0.8692    0.8694    0.8693      1470

time: 1551.2831637859344

ensemble([mlp_clf, rf_clf, sv_clf], get_multimodal_ci_remove_234_over())
mlp_batch=256, mlp_epoch=50
confusion matrix: 
[[350  57]
 [ 61 780]]
              precision    recall  f1-score   support

      Boring     0.8516    0.8600    0.8557       407
 Interesting     0.9319    0.9275    0.9297       841

    accuracy                         0.9054      1248
   macro avg     0.8917    0.8937    0.8927      1248
weighted avg     0.9057    0.9054    0.9056      1248

time: 1314.1192948818207
'''
