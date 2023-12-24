import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import argparse
from CNN_LSTM import CustomModel

def main():
    #读取配置
    config = load_config()
    # 载入数据
    dataloader,testdataloader,valdataloader,dim = load_data()
    #载入模型
    model = load_model(dim)
    #训练模型
    train_model(model,dataloader,testdataloader,valdataloader,num_epochs=100, lr=0.0001,model_save_path='best_model.pth')


def train_model(model, dataloader, testdataloader, valdataloader,num_epochs=10, lr=0.001,
                model_save_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs,_ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in valdataloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
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
            test_outputs,_ = model(test_inputs)
            _, predicted = torch.max(test_outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')






def load_model(dim,phase="train"):
    if phase == "train":
        model = CustomModel(dim)
        return model
    if phase =="test":
        model_save_path = 'best_model.pth'
        model = torch.load(model_save_path)
        return model




def load_data(phase="train"):
    if phase=="train":
        X_train = np.load("../../data/train_train_x.npy",allow_pickle=True)
        y_train = np.load("../../data/train_train_y.npy",allow_pickle=True)
        X_temp = np.load("../../data/train_test_x.npy",allow_pickle=True)
        y_temp = np.load("../../data/train_test_y.npy",allow_pickle=True)
        dim = len(np.unique(y_train))
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
        X_train = np.load("../../data/train_train_x.npy", allow_pickle=True)
        y_train = np.load("../../data/train_train_y.npy", allow_pickle=True)
        X_test = np.load("../../data/test_test_x.npy", allow_pickle=True)
        y_test = np.load("../../data/test_test_y.npy", allow_pickle=True)
        dim =  len(np.unique(y_train))

        # 转换 NumPy 数组为 PyTorch Tensor
        x = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
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