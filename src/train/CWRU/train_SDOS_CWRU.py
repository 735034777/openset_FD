import src.config as config
import os,sys
import pandas as pd
from src.config import BASE_FILE_PATH
sys.path.append('../../utils')
from src.evt_fitting import calculate_openmax_accuracy,calculate_openness
from src.utils.SDOS_CWRU import save_SDOS_dataset
from src.utils.train_base_model import train_base_model



FOLDER_PATH = r'H:\project\data\cwru\CaseWesternReserveUniversityData'
SAVE_PATH = r"H:\project\imbalanced_data\openset_FD\data"


def train_SDOS_CWRU(result_path):
    results = pd.read_csv(result_path)
    metric_types_list = ["cosine","lmnn","euclidean","manhattan"]
    # metric_types_list = ["lmnn"]
    for i in range(60):
        trainlabels,testlabels = save_SDOS_dataset(SAVE_PATH,FOLDER_PATH)

        openness = calculate_openness(len(trainlabels), len(set(trainlabels+testlabels)))
        accuracy = train_base_model()
        for metric_types in metric_types_list:
            for i in range(1,8):
                for j in range(1):
                    new_index = len(results)
                    # config.LMNN_LR = 5*10**(-1*j)
                    testaccuracy,precision,recall,f1,youdens_index,soft_accuracy,soft_metrix = calculate_openmax_accuracy(metric_types,5*i)
                    open_fault = set(testlabels)-set(trainlabels)
                    new_row = [new_index,trainlabels,testlabels,metric_types,5*i,open_fault,openness,accuracy,testaccuracy,precision,
                               recall,f1,youdens_index,soft_accuracy,soft_metrix]
                    print("metric is {},tail size is {}, LR is {:.8f},accuracy is {:.5f}".format(metric_types,5*i,config.LMNN_LR,testaccuracy))
                    # 确定新行的索引位置
                    # new_index = len(results)
                    # 扩展DataFrame
                    # results = results.reindex(results.index.tolist() + [new_index])
                    results.append(pd.Series(new_row, index=results.columns), ignore_index=True)
                    # 使用iloc填充新行
                    # results.iloc[new_index] = new_row
        # results.append(dict(zip(results.columns,new_row)),ignore_index=True)
        results.to_csv(result_path)

if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore")
    result = "SDOS_RESULTS_DIFF_METRIC.csv"
    train_SDOS_CWRU(result)

