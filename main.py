import preprocessing
from dataset import new_dataset
from model.ensemble import *
from dataset.train_test import get_multimodal_ci_remove_234_over
import time

def main():

    start = time.time()

    # overlapping된 데이터셋 생성
    # 처음 파라미터 1은 데이터셋을 60초 단위로 쪼개서 상태지표로 계산한 데이터셋을 만들기 위함
    # 2는 30초 단위로 쪼개기 위함

    preprocessing.preprocessing_overlap(1, './label/labeling.csv', f'preprocessed/')
    preprocessing.preprocessing_overlap(2, './label/labeling.csv', f'preprocessed/')

    # 만약 15초 단위로 쪼갠 데이터를 추가하고 싶다면 아래 코드를 활성화
    # preprocessing.preprocessing_overlap(3, './label/labeling.csv', f'preprocessed/')

    # 첫 번째 경로 = 데이터셋으로 만들 데이터들의 경로
    # 두 번째 경로 = 데이터셋으로 만든 데이터를 저장할 경로(파일명 포함)
    dataset_remove_234 = new_dataset.condition_indicators_dataset(
        "C:/Users/user/PycharmProjects/Withmind_final/preprocessed/*",
        "C:/Users/user/PycharmProjects/Withmind_final/dataset/condition_indicators_dataset_remove_234_over2.csv")

    mlp_clf = mlp(192)
    rf_clf = rf()
    sv_clf = svm()

    # 파라미터 경로에는 앞서 생성한 데이터셋 경로
    ensemble([mlp_clf, rf_clf, sv_clf], get_multimodal_ci_remove_234_over(
        "./dataset/condition_indicators_dataset_remove_234_over2.csv"))

    # 학습에 걸린 시간
    print(f"time: {time.time() - start}")

    # 이 main.py 실행시 최종 모델의 정확도가 나오며, 보다 높은 정확도를 가진 모델 결과가 나올 경우,
    # bagging_model_file.bin에 저장되므로 추후 로드하여 사용 가능


if __name__ == "__main__":
    main()