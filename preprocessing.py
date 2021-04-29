# -*- coding: utf-8 -*-
'''
초기 전처리
1분단위로 나눠 csv 파일로 저장
'''

import pandas as pd
import os
from glob import glob
from label import labeling
from unicodedata import normalize
from os.path import basename


def preprocessing_overlap(num, path, folder):
    label = labeling.labeling(r"C:\Users\user\PycharmProjects\Withmind_final\label\labeling.csv")
    labeling_keys = list(label.keys())
    labeling_values = list(label.values())

    labeling_values_over = []
    for i in range(len(labeling_values)):
        labeling_values_per = []
        for j in labeling_values[i]:
            for k in range(int(num)):
                labeling_values_per.append(j)
        for k in range(int(num)):
            del labeling_values_per[-1]
        labeling_values_over.append(labeling_values_per)
        label[labeling_keys[i]] = labeling_values_over[i]

    if not os.path.exists(folder):
        os.mkdir(folder) # 폴더 생성

    for data in glob(path):
        print(data) # 파일 경로 리스트
        df = pd.read_csv(data)
        df = df[(df.ValidFace == True) & (df.expression__Angry != 0) & (df.expression__Fear != 0) &
                (df.expression__Happy != 0) & (df.expression__Normal != 0) & (df.expression__Sadness != 0) &
                (df.expression__Surprise != 0)]
        df = df.dropna(how='any')
        df = df.reset_index(drop=True)

        df["label"] = 0

        for key, value in label.items():
            key = normalize('NFC', key)
            if key in data:
                for j in range(len(value)):
                    df2 = df[(df.time >= j * 60000/num) & (df.time < (j+1) * 60000/num)].copy(deep=True)
                    df2["label"] = value[j]
                    if len(df2) != 0:
                        df2.to_csv(f"{folder}/{num}_{key}_480_640_{j+1}.csv")

#preprocessing_overlap(2)


def prep_overlap_newdata(num, path, folder):

    if not os.path.exists(folder):
        os.mkdir(folder) # 폴더 생성

    for data in glob(path):
        print(data) # 파일 경로 리스트
        name = basename(data)[:-12]
        df = pd.read_csv(data)
        df = df[(df.ValidFace == True) & (df.expression__Angry != 0) & (df.expression__Fear != 0) &
                (df.expression__Happy != 0) & (df.expression__Normal != 0) & (df.expression__Sadness != 0) &
                (df.expression__Surprise != 0)]
        df = df.dropna(how='any')
        df = df.reset_index(drop=True)

        for j in range(len(df) % 60000 + 1):
            df2 = df[(df.time >= j * 60000/num) & (df.time < (j+1) * 60000/num)].copy(deep=True)

            if len(df2) != 0:
                df2.to_csv(f"{folder}/{num}_{name}_480_640_{j+1}.csv")