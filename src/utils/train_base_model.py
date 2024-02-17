import os, sys, pickle, glob
# import os.path as path
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
from src.config import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import argparse
from CNN_LSTM import CustomModel
from src.utils.II_LOSS import ii_loss

model_save_path = BASE_FILE_PATH+r"\src\train\CWRU"

def train_base_model():
    print("training base model...")
    accuracy = main()
    return accuracy


def main():
    #读取配置
    config = load_config()
    # 载入数据
    dataloader,testdataloader,valdataloader,dim = load_data()
    #载入模型
    model = load_model(dim)
    #训练模型
    accuracy = train_model(model,dataloader,testdataloader,valdataloader,num_classes=dim,num_epochs=BASE_MOEDL_EPOCH, lr=BASE_MODEL_LR,model_save_path='best_model.pth')
    return accuracy

def train_model(model, dataloader, testdataloader, valdataloader,num_epochs=BASE_MOEDL_EPOCH, lr=BASE_MODEL_LR,
                model_save_path=model_save_path+'\\best_model.pth',num_classes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if BASE_MODEL_LOSS_TYPE == "II_LOSS":
        criterion = ii_loss
    if BASE_MODEL_LOSS_TYPE=="CI":
        cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs= model(inputs)
            if BASE_MODEL_LOSS_TYPE == "II_LOSS":
                loss = criterion(outputs[1], labels,num_classes)
            elif BASE_MODEL_LOSS_TYPE=="CI":
                ii_l = ii_loss(outputs[1], labels,num_classes)
                cross_loss = cross_entropy(outputs[1], labels)
                loss = 0.9*cross_loss+0.1*ii_l
            else:
                loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in valdataloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                if BASE_MODEL_LOSS_TYPE == "II_LOSS":
                    val_loss += criterion(val_outputs[1], val_labels,num_classes).item()
                elif BASE_MODEL_LOSS_TYPE == "CI":
                    ii_l = ii_loss(outputs[1], labels, num_classes)
                    cross_loss = cross_entropy(outputs[0], labels)
                    loss = 0.9*cross_loss + 0.1 * ii_l
                    val_loss+=loss
                else:
                    val_loss += criterion(val_outputs[0], val_labels).item()

        val_loss /= len(valdataloader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)

    print(f'Training complete. Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})')

    # Load the best model for testing
    model.load_state_dict(torch.load(model_save_path))

    # Testing
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_inputs, test_labels in testdataloader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs= model(test_inputs)
            _, predicted = torch.max(test_outputs[0].data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    del model
    del dataloader, testdataloader, valdataloader
    return accuracy






def load_model(dim,phase="train"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if phase == "train":
        model = CustomModel(dim)
        return model
    if phase =="test":
        # model_save_path = 'best_model.pth'
        model = CustomModel(dim)
        state_dict  = torch.load(model_save_path+'\\best_model.pth')
        # 加载参数
        model.load_state_dict(state_dict)
        return model

def remap_values(arr, K, N,mapping = None):
    if mapping == None:
        """
        Remap values in the array from the range 0-K to 0-N where K > N.
        """
        # Extract unique values and sort them
        unique_values = np.unique(arr)
        if len(unique_values) > N:
            raise ValueError("The number of unique values in the array is greater than N.")

        # Create a mapping from old values to new values
        mapping = {old_val: new_val for new_val, old_val in enumerate(unique_values)}

        # Apply the mapping
        remapped_array = np.array([mapping[val] for val in arr])
    else:
        # Apply the mapping
        remapped_array = np.array([mapping[val] for val in arr])

    return remapped_array,mapping

def map_test_values_not_in_train(y_train, y_test):
    """
    创建一个映射表，将 y_test 中存在但 y_train 中不存在的值映射到一个新的范围。
    新范围从 y_train 中的唯一值数量开始，到 y_train 中的唯一值数量加上 y_test 独有的值的数量。

    :param y_train: 训练集数组
    :param y_test: 测试集数组
    :return: 映射表
    """
    # 找出 y_test 中有而 y_train 中没有的 unique 值
    unique_test_not_in_train = np.setdiff1d(np.unique(y_test), np.unique(y_train))

    # 计算 N（y_train 中 unique 值的数量）
    N = len(np.unique(y_train))

    # 创建映射表
    unique_values = np.unique(y_train)

    # Create a mapping from old values to new values
    train_mapping = {old_val: new_val for new_val, old_val in enumerate(unique_values)}
    mapping = {val: i + N for i, val in enumerate(unique_test_not_in_train)}
    merged_mapping = train_mapping.copy()  # 复制第一个映射表
    merged_mapping.update(mapping)  # 更新映射表，加入第二个映射表的内容
    try:
        # remapped_array = []
        # for val in y_test:
        #     remapped_array.append(merged_mapping[val])
        remapped_array = np.array([merged_mapping[val] for val in y_test])
    except KeyError:
        # print(val)
        sys.exit()

    return remapped_array,merged_mapping




def load_data(phase="train"):
    if phase=="train":
        X_train = np.load(BASE_FILE_PATH+"/data/train_train_x.npy",allow_pickle=True)
        y_train = np.load(BASE_FILE_PATH+"/data/train_train_y.npy",allow_pickle=True)
        X_temp = np.load(BASE_FILE_PATH+"/data/train_test_x.npy",allow_pickle=True)
        y_temp = np.load(BASE_FILE_PATH+"/data/train_test_y.npy",allow_pickle=True)
        dim = len(np.unique(y_train))
        #将0-10的数值映射到0-6，并保存映射表
        y_train, mapping  = remap_values(y_train,10,dim)
        y_temp,_ = remap_values(y_temp,10,dim,mapping)

        # 再次划分为验证集和测试集
        # X_train,_,y_train,_= train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # test_x = np.load("../../data/testdataset_x.npy", allow_pickle=True)
        # test_y = np.load("../../data/testdataset_y.npy", allow_pickle=True)
        # 转换 NumPy 数组为 PyTorch Tensor
        x = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        # 创建一个 TensorDataset
        dataset = TensorDataset(x, y)
        testdataset = TensorDataset(X_test, y_test)
        valdataset = TensorDataset(X_val, y_val)
        # 创建一个 DataLoader
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)
        valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

        return dataloader,testdataloader,valdataloader,dim
    if phase=="test":
        X_train = np.load(BASE_FILE_PATH+"/data/train_train_x.npy", allow_pickle=True)
        y_train = np.load(BASE_FILE_PATH+"/data/train_train_y.npy", allow_pickle=True)
        X_test = np.load(BASE_FILE_PATH+"/data/test_test_x.npy", allow_pickle=True)
        y_test = np.load(BASE_FILE_PATH+"/data/test_test_y.npy", allow_pickle=True)
        dim =  len(np.unique(y_train))

        #找出y_train和y_test之间的不同值，并将其映射到6-10
        y_test,merged_mapping = map_test_values_not_in_train(y_train,y_test)
        #将0-10的数值映射到0-6，并保存映射表
        y_train, mapping  = remap_values(y_train,10,dim,merged_mapping)


        # 转换 NumPy 数组为 PyTorch Tensor

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        # 创建一个 TensorDataset
        # dataset = TensorDataset(x, y)
        # testdataset = TensorDataset(X_test, y_test)

        # 创建一个 DataLoader
        # batch_size = 64
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)

        return X_train, X_test,y_train,y_test,dim


def load_config():
    parser = argparse.ArgumentParser(description='Neural Network Training Parameters')

    # 添加命令行参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层大小')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--data_path', type=str, default='data.csv', help='数据集路径')

    args = parser.parse_args()
    return args

if __name__ =="__main__":
    main()