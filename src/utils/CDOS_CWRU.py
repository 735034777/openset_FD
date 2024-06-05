import os
from src.config import *
import re
import random
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.utils.SDOS_CWRU import extract_info_from_filename,read_mat_files,validate_OR,build_dataset,segment_time_series
from src.config import *





def read_mat_files(folder_path):
    data = []
    labels = []
    speed = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".mat"):
            infomationOfFilename = extract_info_from_filename(filename)
            # print(infomationOfFilename)
            file_path = os.path.join(folder_path, filename)
            mat_data = loadmat(file_path)
            #             print(mat_data)

            # 这里假设MAT文件中有一个数组变量名为'data'，你可以根据实际情况修改
            mat_variable_data = mat_data.get('X' + str(infomationOfFilename["Trial Number"]) + "_DE_time", None)
            if infomationOfFilename["Trial Number"] >= 100:
                mat_variable_data = mat_data.get('X' + str(infomationOfFilename["Trial Number"]) + "_DE_time", None)
                mat_speed = mat_data.get('X' + str(infomationOfFilename["Trial Number"]) + "RPM", None)
            else:
                mat_variable_data = mat_data.get('X' + "0" + str(infomationOfFilename["Trial Number"]) + "_DE_time",
                                                 None)
                mat_speed = mat_data.get('X' + "0" + str(infomationOfFilename["Trial Number"]) + "RPM", None)
            if mat_variable_data is not None:
                # 将数组数据转换为一维数组或一维列表，具体取决于你的需求
                flattened_data = mat_variable_data.flatten()
                data.append(flattened_data)
                labels.append(filename)
                if mat_speed is not None:
                    speed.append(mat_speed[0][0])
                else:
                    speed.append(0)

    df = pd.DataFrame({'data': data, 'label': labels, "speed": speed})
    return df


def validate_OR(input_str):
    #识别str是否是out ring的故障类型
    pattern = re.compile(r'^[A-Z]{2}\d{3}@\d{1,2}$')
    return bool(pattern.match(input_str))


def build_dataset(df, all_labels,load):
    # 读取MAT文件并创建DataFrame
    # 设置参数
    # load = load
    filepath = "train.csv"

    #     all_labels_idex = [0,1,2,3,4,5,6,7,8,9]
    train_ratio = 0.7
    # 初始化中间变量
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    # 构建测试集和训练集，使用采样频率为12k的drive_end数据
    for i in range(df.shape[0]):
        file_info = extract_info_from_filename(df.loc[i]["label"])
        if file_info["Type"] != "Normal":
            if file_info['Sample Frequence'] == 12 and file_info['Drive End'] == "Drive_End":
                if file_info['Load'] == load:
                    if validate_OR(file_info['Fault Info']):
                        # 使用OR的6方向
                        if file_info['Fault Info'].split('@')[1] == "3" or file_info['Fault Info'].split('@')[
                            1] == "12":
                            continue
                        #                     print(file_info)
                    #                     if file_info['Fault Info'] in trainlabels:
                    train_data.append(df.loc[i]["data"][:int(len(df.loc[i]["data"]) * train_ratio)])
                    train_label.append(all_labels.index(
                        file_info["Fault Info"]))  # 使用故障类型在all_labels中的index作为y值，下同
                    #                     if file_info['Fault Info'] in testlabels:
                    test_data.append(df.loc[i]["data"][int(len(df.loc[i]["data"]) * train_ratio) + 1:])
                    test_label.append(all_labels.index(file_info["Fault Info"]))
        # 将normal数据加入训练集,normal的代号为9
        if file_info["Type"] == "Normal" and file_info["Load"] == load:
            train_data.append(df.loc[i]["data"][:int(len(df.loc[i]["data"]) * train_ratio)])
            train_label.append(9)
            test_data.append(df.loc[i]["data"][int(len(df.loc[i]["data"]) * train_ratio) + 1:])
            test_label.append(9)
    #     输出csv
    train = pd.DataFrame({"data": train_data, "label": train_label})
    train.to_csv(filepath)
    test = pd.DataFrame({"data": test_data, "label": test_label})
    test.to_csv("test.csv")
    # 返回dataframe类型的训练集和测试集数据
    return train, test



def segment_time_series(time_series):
    #设置参数
    segment_length = 1024
    step = 60
#     初始化中间变量
    segments = []
    segments_fft = []
    for i in range(0, len(time_series), step):
        segment = time_series[i:i + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
    return segments

def create_testdataset(train,test,all_labels,train_labels,test_labels,save_path,phase = "test"):
    #初始化中间变量

    data = []
    labels = []
    testdata = []
    testlabels = []
    for i in range(train.shape[0]):
        if phase =="train" and all_labels[train.iloc[i,1]] in train_labels:
            segments = segment_time_series(train.iloc[i,0])
            data.extend(segments)
            labels.extend([train.iloc[i,1]]*len(segments))
            segments = segment_time_series(test.iloc[i,0])
            testdata.extend(segments)
            testlabels.extend([test.iloc[i,1]]*len(segments))
        if phase =="test" and all_labels[test.iloc[i,1]] in test_labels:#加入测试阶段测试集数据，包括normal
            segments = segment_time_series(test.iloc[i,0])
            testdata.extend(segments)
            testlabels.extend([test.iloc[i,1]]*len(segments))
    if phase =="train":
        np.save(save_path+r"\train_train"+"_x.npy", np.array(data))
        np.save(save_path+r"\train_train"+"_y.npy", np.array(labels))
        np.save(save_path+r"\train_test"+"_x.npy", np.array(testdata))
        np.save(save_path+r"\train_test"+"_y.npy", np.array(testlabels))
    if phase =="test":
        np.save(save_path+r"\test_test"+"_x.npy", np.array(testdata))
        np.save(save_path+r"\test_test"+"_y.npy", np.array(testlabels))

def save_CDOS_dataset(save_path,
        folder_path = 'H:\project\data\cwru\CaseWesternReserveUniversityData'
):
    # 指定包含MAT文件的文件夹路径
    # folder_path = 'H:\project\data\cwru\CaseWesternReserveUniversityData'

    # 读取MAT文件并创建DataFrame
    print("data generating...")
    df = read_mat_files(folder_path)

    all_labels = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007@6', 'OR014@6', 'OR021@6']
    # 随机生成训练集和测试集lables，并打印区别
    num_labels_to_select = random.randint(2, 5)
    trainlabels = random.sample(all_labels, num_labels_to_select)
    num_labels_to_select = random.randint(5, 9)
    testlabels = random.sample(all_labels, num_labels_to_select)
    trainlabels.append("normal")
    testlabels.append("normal")
    all_labels.append("normal")
    # print(list(set(trainlabels).symmetric_difference(set(testlabels))))  # 应该先求交后求差
    train, test = build_dataset(df, all_labels,load=0)
    create_testdataset(train, test, all_labels, trainlabels, testlabels,save_path, phase="train")
    _, test = build_dataset(df, all_labels, load=2)
    create_testdataset(train, test, all_labels, trainlabels, testlabels, save_path,phase="test")
    del train,test,_
    return trainlabels,testlabels

if __name__=="__main__":
    save_CDOS_dataset(BASE_FILE_PATH+"/data")

