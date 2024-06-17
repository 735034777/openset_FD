import sys
# sys.path.append(r"H:\project\imbalanced_data\openset_FD")
import os
# print(os.getcwd())

# print(sys.path)
import src.config as config
import os,sys
import pandas as pd
from src.config import BASE_FILE_PATH
sys.path.append('../../utils')
from src.train.NPP_PCTRAN.utils.evt_fitting import calculate_openmax_accuracy,calculate_openness
from src.train.NPP_PCTRAN.dataset_built import save_SDOS_dataset
from src.train.NPP_PCTRAN.utils.train_base_model import train_base_model
from src.utils.tools import delete_file
from itertools import combinations
from dataset_built import SDOSDatasetManager
import numpy as np
import torch
import random


# 设置随机种子
# random.seed(142)
# np.random.seed(142)
# torch.manual_seed(142)

FOLDER_PATH = r'H:\project\imbalanced_data\openset_FD'
SAVE_PATH = r"H:\project\imbalanced_data\openset_FD\data"

def train_SDOS_PCTRAN(result_path):
    results = pd.read_csv(result_path,index_col=0)
    metric_types_list = ["cosine","lmnn","euclidean","manhattan"]
    metric_types_list = ["cosine","euclidean","manhattan"]
    # metric_types_list = ["lmnn"]
    manager = SDOSDatasetManager(folder_path=SAVE_PATH+r"\NPP\100", save_path=SAVE_PATH)
    trainlabels,testlabels = manager.generate_datasets()
    open_fault = set(testlabels)-set(trainlabels)
    openness = calculate_openness(len(trainlabels), len(set(trainlabels+testlabels)))
    accuracy = train_base_model()
    for i in range(200):
        tailsize = 5


        # trainlabels,testlabels = save_SDOS_dataset(SAVE_PATH,FOLDER_PATH)
        # trainlabels,testlabels = manager.generate_datasets()

        # openness = calculate_openness(len(trainlabels), len(set(trainlabels+testlabels)))
        # accuracy = train_base_model()
        for metric_types in metric_types_list:
            for j in range(1):
                if j == 0:
                    config.HIGH_DIMENSION_OUTPUT = False
                else:
                    config.HIGH_DIMENSION_OUTPUT = True
                for k in range(0,1):
                    new_index = len(results)
                    # config.LMNN_LR = 5*10**(-1*j)
                    testaccuracy,precision,recall,f1,youdens_index,soft_accuracy,soft_metrix = calculate_openmax_accuracy(metric_types,tailsize)

                    try:
                        testaccuracy,precision,recall,f1,youdens_index,soft_accuracy,soft_metrix = calculate_openmax_accuracy(metric_types,tailsize)
                    except:
                        continue
                    # open_fault = set(testlabels)-set(trainlabels)
                    formatted_open_fault = ', '.join(map(str, open_fault))
                    if config.HIGH_DIMENSION_OUTPUT==True:
                        metric_types_record = metric_types+"_HDV"
                    else:
                        metric_types_record=metric_types
                    new_row = [trainlabels,testlabels,metric_types_record,tailsize,open_fault,openness,accuracy,testaccuracy,precision,
                               recall,f1,youdens_index,soft_accuracy,soft_metrix]
                    print("metric is {},tail size is {}, open_fault is {{{}}},accuracy is {:.5f}".format(metric_types,tailsize,open_fault,testaccuracy))
                    # 确定新行的索引位置
                    # new_index = len(results)
                    # 扩展DataFramel
                    results = results.reindex(results.index.tolist() + [new_index])
                    # results.append(pd.Series(new_row, index=results.columns), ignore_index=True)
                    # 使用iloc填充新行
                    results.iloc[new_index] = new_row
                if os.path.exists(BASE_FILE_PATH+"/src/data/CWRU_lmnn_model.pkl"):
                    os.remove(BASE_FILE_PATH+"/src/data/CWRU_lmnn_model.pkl")
        # results.append(dict(zip(results.columns,new_row)),ignore_index=True)
        results.to_csv(result_path)


if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore")
    result = r"H:\project\imbalanced_data\openset_FD\src\train\NPP_PCTRAN\SDOS_DISSCUSSION_TAILSIZE.csv"
    # config.HIGH_DIMENSION_OUTPUT = True
    train_SDOS_PCTRAN(result)
    # sys.exit()