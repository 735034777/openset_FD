from src.utils.SDOS_CWRU import SNR_CWRU_Data
import pandas as pd
import os
from src.utils.train_base_model import train_base_model
from src.evt_fitting import calculate_openmax_accuracy,calculate_openness
import random
import numpy as np
import torch


# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE_FILE_PATH = r"H:\project\imbalanced_data\openset_FD"


def train_SDOS_CWRU(result_path):
    results = pd.read_csv(result_path,index_col=0)
    metric_types_list = ["cosine","lmnn","euclidean","manhattan"]
    metric_types_list = ["cosine","euclidean","manhattan"]
    # metric_types_list = ["lmnn"]
    for i in range(1):
        snr_cwru_data = SNR_CWRU_Data("H:\project\imbalanced_data\openset_FD\data",
                                      "H:\project\data\cwru\CaseWesternReserveUniversityData",SNR=-10*(i+1))
        trainlabels,testlabels = snr_cwru_data.save_CDOS_dataset()

        openness = snr_cwru_data.openness
        accuracy = train_base_model()
        for metric_types in metric_types_list:

            for i in range(0,1):
                tailsize = 5
                new_index = len(results)
                # config.LMNN_LR = 5*10**(-1*j)
                testaccuracy,precision,recall,f1,youdens_index,soft_accuracy,soft_metrix = calculate_openmax_accuracy(metric_types,tailsize)
                open_fault = set(testlabels)-set(trainlabels)
                formatted_open_fault = ', '.join(map(str, open_fault))

                metric_types_record=metric_types
                new_row = [trainlabels,testlabels,metric_types_record,tailsize,open_fault,openness,accuracy,testaccuracy,precision,
                           recall,f1,youdens_index,soft_accuracy,soft_metrix,snr_cwru_data.SNR]
                print("metric is {},tail size is {}, open_fault is {{{}}},accuracy is {:.5f}".format(metric_types,tailsize,open_fault,testaccuracy))

                results = results.reindex(results.index.tolist() + [new_index])

                results.iloc[new_index] = new_row
            if os.path.exists(BASE_FILE_PATH+"/src/data/CWRU_lmnn_model.pkl"):
                os.remove(BASE_FILE_PATH+"/src/data/CWRU_lmnn_model.pkl")
        # results.append(dict(zip(results.columns,new_row)),ignore_index=True)
        results.to_csv(result_path)

if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore")
    result = "SDOS_RESULTS_tailsize.csv"
    # config.HIGH_DIMENSION_OUTPUT = True
    train_SDOS_CWRU(result)
    # sys.exit()
