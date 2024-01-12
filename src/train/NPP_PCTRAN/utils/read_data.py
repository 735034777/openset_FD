#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 20:45
# @Author  : Wang Yushun
# @Site    :
# @File    : LSTM.py
# @Software: PyCharm
# @describe:训练GCNHiGRU所用的函数

import os
import pandas as pd
from src.train.NPP_PCTRAN.config import *
SOURCE_FILE_PATH =  BASE_FILE_PATH +r"\data\NPP"
import os
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
# from sko.GA import GA
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
# from models.GCNModel import Model

import torchmetrics


def seed_torch(seed=1029):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def evaluate(y_true, y_hat, label='test'):
    acc = accuracy_score(y_true,y_hat)
    pre = precision_score(y_true,y_hat,average="micro")
    recall = recall_score(y_true,y_hat,average="micro")
    # print('{} set acc:{}, pre:{}, recall:{}'.format(label, acc, pre,recall))
    return acc,pre,recall

def get_data(filepath=SOURCE_FILE_PATH+"\LOCAC50.csv"
             ,IF_fft=False,IF_normalize=True,highest_time=2300):
    data = pd.read_csv(filepath)
    import os
    import re
    def listdir(path, list_name):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name)
            elif os.path.splitext(file_path)[1] == '.csv':
                list_name.append(file_path)

    label_list = []

    listdir(SOURCE_FILE_PATH, label_list)
    # np.savetxt('./data/label_num_map.csv', label_list)
    label_num = [i for i in range(len(label_list))]
    label_num_map = dict(zip(label_list, label_num))
    label_num_map_str = yaml.dump(label_num_map)
    with open(SOURCE_FILE_PATH+'./label_num_map.yaml',"w") as f:
        f.write(label_num_map_str)
    # define column names for easy indexing
    index_names = ['Time']
    # sensor_names = ['s_{}'.format(i) for i in range(1,97)]
    sensor_names = data.columns
    label = ["label"]
    # col_names = index_names + sensor_names
    col_names = sensor_names
    # read data
    data = pd.DataFrame()
    columns_to_normalize = [col for col in sensor_names if col != "TIME"]
    for file in label_list:
        # print(file)
        temdata = pd.read_csv(file, header=None, names=col_names)
        temdata = temdata.drop(0)
        for sensor in temdata.columns:
            temdata[sensor] = pd.to_numeric(temdata[sensor])
        temdata = temdata[temdata["TIME"] <= MAXTIMESTAMP]


        if IF_normalize:
            scaler = MinMaxScaler()
            temdata[columns_to_normalize] = pd.DataFrame(scaler.fit_transform(temdata[columns_to_normalize]), columns=columns_to_normalize)
        temdata[label] = label_num_map[file]
        data = pd.concat([data, temdata[:-1]], axis=0)

    # if IF_normalize:
    #     scaler = MinMaxScaler()
    #     data[columns_to_normalize] = pd.DataFrame(scaler.fit_transform(data[columns_to_normalize]),
    #                                                  columns=columns_to_normalize)
    # for sensor in data.columns:
    #     data[sensor] = pd.to_numeric(data[sensor])
    data = data.loc[:, (data != data.iloc[0]).any()]  # 删除单一值列
    data = data.dropna(axis=1)

    # graph,remain_sensors = get_graph()
    # data = data[data["TIME"] < highest_time]
    # sensors = data.columns.drop(["TIME", "label"])
    # if IF_normalize:
    #     # sensors = data.columns.drop(["TIME","label"])
    #     data[sensors] = (data[sensors] - data[sensors].min()) / (
    #                 data[sensors].max() - data[sensors].min())
    # if IF_fft:
    #     from scipy import fft
    #     labels = data['label'].unique()
    #     temdatablow400 = pd.DataFrame()
    #     for l in labels:
    #         data_fft_blow400 = pd.DataFrame()
    #         for sensor in data.columns.drop(["label","TIME"]):
    #             data_fft_blow400[sensor] = np.abs(fft.rfft(data[data["label"] == l][sensor].values))
    #         data_fft_blow400["label"] = l
    #         # data_fft_blow400["id"] = np.linspace(0,data_fft_blow400.shape[0],num=data_fft_blow400.shape[0],endpoint=False,dtype=np.int)
    #         temdatablow400 = pd.concat([temdatablow400, data_fft_blow400], axis=0)
    #     data = temdatablow400
    # temdata = data.loc[:,remain_sensors+["label"]]
    # # temdata["label"]=data["label"]
    # # temdata.set_index("id")
    # # temdata.to_csv("./data/data.csv")
    return data


def get_graph(filepath=SOURCE_FILE_PATH+r"\GRAPH\nppgraph.csv"):
    graph = pd.read_csv(filepath)

    remain_sensors = list(graph["name"])
    graph = graph.drop("name", axis=1)
    return graph,remain_sensors
















if __name__=="__main__":
    seed_torch()
    data=get_data()
    pass



