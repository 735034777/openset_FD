import os
import torch
import torch.nn as nn
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def main():
    np.random.seed(1234)
    #载入模型
    model = load_model()
    # 获取数据
    data = get_train_test()
    #分割数据
    x_train,x_test,y_train,y_test = data
    #计算mean和distance,并保存其文件
    create_model(model,data)

    N=100 #测试100个测试集数据
    for i in range(N):
        #生成随机测试集序号
        random_char = np.randint(0,len(x_test))

        test_x1 = x_test[random_char]
        test_y1 = y_test[random_char]

        #计算测试集的激活值
        activation = compute_activation(model, test_x1)

        #计算openmax激活值
        softmax, openmax = compute_openmax(model, activation)

