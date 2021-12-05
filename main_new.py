import preprocessing
from dataset import new_dataset
from model.ensemble import *


def main():
    # 저장한 모델 bin 파일을 불러와서 list 변수로 저장
    with open("bagging_model_file.bin", "rb") as f:
        bagging_model_list = []
        while True:
            try:
                data = pickle.load(f)
            except EOFError:
                break

            bagging_model_list.append(data)


    # 첫번째 경로 = 파라미터에 새로운 데이터 경로 입력
    # 두번째 경로 = 파라미터에 전처리 한 데이터를 저장할 폴더 경로 입력
    preprocessing.prep_overlap_newdata(1,
                                        "C:/Users/user/PycharmProjects/Withmind_final/test_data/*",
                                        "C:/Users/user/PycharmProjects/Withmind_final/test_pre")
    preprocessing.prep_overlap_newdata(2,
                                        "C:/Users/user/PycharmProjects/Withmind_final/test_data/*",
                                        "C:/Users/user/PycharmProjects/Withmind_final/test_pre")

    # 첫 번째 경로 = 데이터셋으로 만들 데이터들의 경로
    # 두 번째 경로 = 데이터셋으로 만든 데이터를 저장할 경로(파일명 포함)
    newdataset = new_dataset.condition_indicators_newdataset(
        "C:/Users/user/PycharmProjects/Withmind_final/test_pre/*",
        "C:/Users/user/PycharmProjects/Withmind_final/test_dataset.csv")


    '''
    pred는 새롭게 전처리 된 데이터에 대한 예측 결과
    순서는 데이터를 입력한 순서
    예를 들어, 첫 번째 데이터가 특정 영상의 ~분부터 ~분까지 데이터에 대한 상태지표 전처리 데이터라면,
    pred의 첫 번째 숫자 결과는 그 데이터에 대한 집중도 예측을 보여줌
    pred는 list 변수이며 추후 애플리케이션에 이용할 정보임
    '''
    pred = voting(bagging_model_list[0], newdataset)


if __name__ == "__main__":
    main()