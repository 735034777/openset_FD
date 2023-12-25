import os, sys, pickle, glob
# import os.path as path
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
import torch
import torch.nn as nn
import numpy as np
from utils.train_base_model import load_model,load_data
from utils.openmax import create_model
from evt_fitting import calculate_openmax_accuracy

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def main():
    np.random.seed(1234)

    # 获取数据
    data = load_data(phase="test")
    #分割数据
    x_train,x_test,y_train,y_test,dim = data
    #载入模型
    model = load_model(dim,phase="test")
    #计算mean和distance,并保存其文件
    create_model(model,data)

    accuracy = calculate_openmax_accuracy(i)


if __name__ =="__main__":
    main()

