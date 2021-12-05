'''
해당 코드는 초기 분석 결과를 도출하기 위한
선형 모델 비교 코드
'''

import preprocessing
from dataset import new_dataset
from model import linear_regression, svr
import time


def main():

    preprocessing.preprocessing_overlap(1, './label/labeling.csv', f'preprocessed/')
    preprocessing.preprocessing_overlap(2, './label/labeling.csv', f'preprocessed/')
    new_dataset.new_dataset("./dataset/mean_dataset.csv")
    new_dataset.condition_indicators_dataset_five()

    start = time.time()
    linear_regression.linear_regression_mean("./dataset/mean_dataset.csv")
    svr.li_svr_five("./dataset/condition_indicators_dataset_five.csv")
    print(f"time: {time.time() - start}")


if __name__ == "__main__":
    main()