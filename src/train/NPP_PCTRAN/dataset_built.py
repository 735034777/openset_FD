import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.train.NPP_PCTRAN.utils.read_data import get_data
from src.config import *
SOURCE_FILE_PATH =  BASE_FILE_PATH +"/data/NPP"


def read_mat_files(filepath=SOURCE_FILE_PATH+r"\100\LOCAC50.csv"):
    df = get_data(filepath)
    return df

def save_SDOS_dataset(save_path,
        folder_path = r'H:\project\data\NPP'
):
    # 指定包含MAT文件的文件夹路径
    # folder_path = 'H:\project\data\cwru\CaseWesternReserveUniversityData'

    # 读取MAT文件并创建DataFrame
    print("data generating...")
    df = read_mat_files()
    num_class = len(pd.unique(df["label"]))
    all_labels = [i for i in range(num_class)]
    # 随机生成训练集和测试集lables，并打印区别
    num_labels_to_select = random.randint(3, 5)
    trainlabels = random.sample(all_labels, num_labels_to_select)
    num_labels_to_select = random.randint(5, 20)
    testlabels = random.sample(all_labels, num_labels_to_select)
    if USE_FIX_LABELS:
        trainlabels = [0,4,3,5,6,9]
        testlabels = [4,7,6,9]
    # trainlabels.append("normal")
    # testlabels.append("normal")
    # all_labels.append("normal")
    # print(list(set(trainlabels).symmetric_difference(set(testlabels))))  # 应该先求交后求差

    create_testdataset(df, all_labels, trainlabels, testlabels,save_path, phase="train")
    create_testdataset(df, all_labels, trainlabels, testlabels, save_path,phase="test")
    del df
    return trainlabels,testlabels

def create_testdataset(raw_data,all_labels,train_labels,test_labels,save_path,phase = "test"):
    #初始化中间变量

    data = []
    labels = []
    testdata = []
    testlabels = []
    train = []
    for i in raw_data["label"].unique():
        train.append(raw_data[raw_data["label"]==i])
    for i in range(len(train)):
        if phase =="train" and i in train_labels:
            segments = segment_time_series(train[i][train[i].columns.drop(["TIME","label"])])
            data.extend(segments)
            labels.extend([i]*len(segments))
            # segments = segment_time_series(test.iloc[i,0])
            # testdata.extend(segments)
            # testlabels.extend([test.iloc[i,1]]*len(segments))
        if phase =="test" and i in test_labels:#加入测试阶段测试集数据，包括normal
            segments = segment_time_series(train[i][train[i].columns.drop(["TIME","label"])])
            testdata.extend(segments)
            testlabels.extend([i]*len(segments))
    if phase =="train":
        data, testdata, labels, testlabels = train_test_split(np.array(data), np.array(labels), test_size=0.15, random_state=42)
        np.save(save_path+r"\train_train"+"_x.npy", np.array(data))
        np.save(save_path+r"\train_train"+"_y.npy", np.array(labels))
        np.save(save_path+r"\train_test"+"_x.npy", np.array(testdata))
        np.save(save_path+r"\train_test"+"_y.npy", np.array(testlabels))
    if phase =="test":
        np.save(save_path+r"\test_test"+"_x.npy", np.array(testdata))
        np.save(save_path+r"\test_test"+"_y.npy", np.array(testlabels))



def segment_time_series(time_series):
    #设置参数
    from src.train.NPP_PCTRAN.config import SEGMENT_LENGTH,STEP
    segment_length = SEGMENT_LENGTH
    step = STEP
#     初始化中间变量
    segments = []
    segments_fft = []
    for i in range(0, len(time_series), step):
        segment = time_series[i:i + segment_length]
        if len(segment) == segment_length:
            segments.append(segment.values)
    return segments


if __name__ =="__main__":
    data = read_mat_files()
    segments = segment_time_series(data)
    trainlabels,testlabels = save_SDOS_dataset()
    print()