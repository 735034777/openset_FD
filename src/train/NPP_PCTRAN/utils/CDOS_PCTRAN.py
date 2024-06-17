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
from src.config import *
from src.train.NPP_PCTRAN.dataset_built import save_SDOS_dataset,create_testdataset,read_mat_files

from src.train.NPP_PCTRAN.config import *
import src.train.NPP_PCTRAN.config as config
from itertools import combinations
# SOURCE_FILE_PATH =  BASE_FILE_PATH +"/data/NPP"


def save_CDOS_dataset(save_path,trainlabels,testlabels,index,
        folder_path = r'H:\project\data\NPP'
):
    # 指定包含MAT文件的文件夹路径
    # folder_path = 'H:\project\data\cwru\CaseWesternReserveUniversityData'

    # 读取MAT文件并创建DataFrame

    tem_sorce_file_path = config.SOURCE_FILE_PATH
    config.SOURCE_FILE_PATH = tem_sorce_file_path+'\\' + index[0]

    df = read_mat_files(config.SOURCE_FILE_PATH+r"\10.csv")
    # num_class = len(pd.unique(df["label"]))
    all_labels = [i for i in range(26)]
    # # 随机生成训练集和测试集lables，并打印区别
    # num_labels_to_select = random.randint(3, 20)
    # trainlabels = random.sample(all_labels, num_labels_to_select)
    # num_labels_to_select = random.randint(3, 20)
    # testlabels = random.sample(all_labels, num_labels_to_select)
    # trainlabels.append("normal")
    # testlabels.append("normal")
    # all_labels.append("normal")
    # print(list(set(trainlabels).symmetric_difference(set(testlabels))))  # 应该先求交后求差

    create_testdataset(df, all_labels, trainlabels, testlabels,save_path, phase="train")
    config.SOURCE_FILE_PATH = tem_sorce_file_path +"\\" +index[1]
    df = read_mat_files(config.SOURCE_FILE_PATH +r"\LOCAC50.csv")
    create_testdataset(df, all_labels, trainlabels, testlabels, save_path,phase="test")
    config.SOURCE_FILE_PATH = tem_sorce_file_path
    del df

    return trainlabels,testlabels

class LabelGenerator:
    def __init__(self):
        self.index = 0
        self.openfault_subsets = []

    def generate_labels(self):
        num_class = 26
        all_labels = list(range(num_class))
        num_labels_to_select = random.randint(3, 5)
        trainlabels = random.sample(all_labels, num_labels_to_select)
        USE_FIX_LABELS = True
        if USE_FIX_LABELS:
            trainlabels =[9, 19, 13, 17, 12, 4, 18, 7, 15, 1, 20, 24]
            openfault = list(set(all_labels) - set(trainlabels))
            if not self.openfault_subsets:  # 只在第一次调用时生成子集
                self.openfault_subsets = list(combinations(openfault, 2))
            testlabels = trainlabels+list(self.openfault_subsets[self.index % len(self.openfault_subsets)])
            testlabels = [21, 9, 13, 12, 14, 24, 15, 4, 1, 7]
            self.index += 1
        else:
            testlabels = []
        return trainlabels, testlabels


def generate_labels():
    num_class = 26
    all_labels = [i for i in range(num_class)]
    # 随机生成训练集和测试集lables，并打印区别
    num_labels_to_select = random.randint(3, 5)
    trainlabels = random.sample(all_labels, num_labels_to_select)
    num_labels_to_select = random.randint(5, 24)
    testlabels = random.sample(all_labels, num_labels_to_select)
    if USE_FIX_LABELS:
        trainlabels=[9, 19, 13, 17, 12, 4, 18, 7, 15, 1, 20, 24]
        openfault = list(set(all_labels)-set(trainlabels))
        openfault_subsets = list(combinations(openfault, 4))
        testlabels = [21, 9, 13, 12, 14, 24, 15, 4, 1, 7]
    return trainlabels,testlabels







if __name__=="__main__":
    save_CDOS_dataset(BASE_FILE_PATH+"/data")

