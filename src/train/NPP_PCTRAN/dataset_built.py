import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.train.NPP_PCTRAN.utils.read_data import get_data
# from src.config import *
from src.config import *
from itertools import combinations
import yaml

SOURCE_FILE_PATH =  BASE_FILE_PATH +"/data/NPP"


class SDOSDatasetManager:
    def __init__(self, folder_path, save_path):
        self.folder_path = folder_path
        self.save_path = save_path
        self.USE_FIX_LABELS = True
        self.IF_normalize = False
        self.MAXTIMESTAMP = 1500
        self.index =0
        self.openfault_subsets = []
        print("SDOS Dataset Manager initialized.")

    def read_mat_files(self):
        # 这里假设read_mat_files的功能是读取MAT文件并返回DataFrame
        # 实际实现应该根据实际情况进行修改
        # 示例数据
        import os
        path = os.path.join(self.folder_path,"10.csv")
        data = pd.read_csv(path)

        def listdir(path, list_name):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isdir(file_path):
                    listdir(file_path, list_name)
                elif os.path.splitext(file_path)[1] == '.csv':
                    list_name.append(file_path)

        label_list = []
        # path = os.path.join(self.folder_path,)

        listdir(self.folder_path, label_list)
        # np.savetxt('./data/label_num_map.csv', label_list)
        label_num = [i for i in range(len(label_list))]
        label_num_map = dict(zip(label_list, label_num))
        label_num_map_str = yaml.dump(label_num_map)
        with open(SOURCE_FILE_PATH + './label_num_map.yaml', "w") as f:
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
            temdata = temdata[temdata["TIME"] <= self.MAXTIMESTAMP]

            # if self.IF_normalize:
            #     scaler = MinMaxScaler()
            #     temdata[columns_to_normalize] = pd.DataFrame(scaler.fit_transform(temdata[columns_to_normalize]),
            #                                                  columns=columns_to_normalize)
            temdata[label] = label_num_map[file]
            data = pd.concat([data, temdata[:-1]], axis=0)


        data = data.loc[:, (data != data.iloc[0]).any()]  # 删除单一值列
        data = data.dropna(axis=1)


        return data

    def create_testdataset(self, df, all_labels, trainlabels, testlabels, phase):
        # 这里应该实现数据集的创建和保存逻辑
        pass

    def generate_datasets(self):
        print("Data generating...")
        df = self.read_mat_files()
        num_class = len(pd.unique(df["label"]))
        all_labels = list(range(num_class))

        if self.USE_FIX_LABELS:
            trainlabels = [8, 16, 19, 22]
            openfault = list(set(all_labels) - set(trainlabels))
            if not self.openfault_subsets:  # 只在第一次调用时生成子集
                self.openfault_subsets = list(combinations(openfault, 18))
            # openfault_subsets = list(combinations(openfault, 18))
            testlabels = trainlabels + list(self.openfault_subsets[self.index % len(self.openfault_subsets)])
            self.index += 1
        else:
            num_labels_to_select = random.randint(3, 5)
            trainlabels = random.sample(all_labels, num_labels_to_select)
            num_labels_to_select = random.randint(5, 20)
            testlabels = random.sample(all_labels, num_labels_to_select)

        create_testdataset(df, all_labels, trainlabels, testlabels,self.save_path, "train")
        create_testdataset(df, all_labels, trainlabels, testlabels,self.save_path, "test")
        del df
        return trainlabels, testlabels

def read_mat_files(filepath=SOURCE_FILE_PATH+r"\LOCAC50.csv"):
    df = get_data(filepath + r"\LOCAC50.csv")
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
        trainlabels=[8, 16, 19, 22]
        openfault = list(set(all_labels)-set(trainlabels))
        openfault_subsets = list(combinations(openfault, 18))
        testlabels = trainlabels+openfault_subsets
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