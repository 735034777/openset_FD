import os,sys
import pandas as pd
from src.config import *
sys.path.append('../../utils')
from src.evt_fitting import calculate_openmax_accuracy,calculate_openness
from src.utils.SDOS_CWRU import save_SDOS_dataset
from src.utils.train_base_model import train_base_model
from src.config import *


FOLDER_PATH = r'H:\project\data\cwru\CaseWesternReserveUniversityData'
SAVE_PATH = r"H:\project\imbalanced_data\openset_FD\data"


def train_SDOS_CWRU(result_path):
    results = pd.read_csv(result_path)
    for i in range(60):
        trainlabels,testlabels = save_SDOS_dataset(SAVE_PATH,FOLDER_PATH)

        openness = calculate_openness(len(trainlabels), len(set(trainlabels+testlabels)))
        accuracy = train_base_model()
        testaccuracy,precision,recall,f1,youdens_index = calculate_openmax_accuracy(5)
        open_fault = set(testlabels)-set(trainlabels)
        results.append(dict(zip(results.columns,[trainlabels,testlabels,open_fault,openness,accuracy,testaccuracy,precision,recall,f1,youdens_index])),ignore_index=True)
        results.to_csv(result_path)

if __name__=="__main__":
    result = "SDOS_RESULTS.csv"
    train_SDOS_CWRU("SDOS_RESULTS.csv")

