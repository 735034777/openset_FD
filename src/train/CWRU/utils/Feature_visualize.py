import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from src.utils.train_base_model import load_model,load_data,model_save_path,train_model
from src.train.CWRU.utils.CWRU_CONFIG import *
from src.train.CWRU.utils.Loss_Functions import *


def train_model(model,data_loader,valdataloader,criterion):
    num_epochs = NUMBER_OF_EPOCH
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=CWRU_LR)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output,features,_ = model(data)
            loss = criterion(output, target)
            loss.backward()
            val_loss = 0
            with torch.no_grad():
                for val_inputs, val_labels in valdataloader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs,val_features,_ = model(val_inputs)
                    val_loss += criterion(val_outputs, val_labels).item()
            val_loss /= len(valdataloader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path+'\\best_model.pth')

    print(f'Training complete. Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})')



def visualize(model, title,dataloader):
    model.cuda()
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.cuda()
            _,feature, _ = model(data)
            features.extend(feature.cpu().numpy())
            labels.extend(target.numpy())

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(np.array(features))

    # plt.figure(figsize=(8, 8))
    for i in range(10):
        idxs = [idx for idx, label in enumerate(labels) if label == i]
        plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], label=str(i))
    plt.legend()
    plt.title(title)
    # plt.show()

def visualize_test(model, title,data,target):
    model.cuda()
    model.eval()
    features = []
    labels = []
    dataset = TensorDataset(data, target)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for data, target in dataloader:
            data = data.cuda()
            _,feature, _ = model(data)
            features.extend(feature.cpu().numpy())
            labels.extend(target.numpy())

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(np.array(features))

    # plt.figure(figsize=(8, 8))
    for i in range(10):
        idxs = [idx for idx, label in enumerate(labels) if label == i]
        plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], label=str(i))
    plt.legend()
    plt.title(title)
    # plt.show()
    # return plt

if __name__=="__main__":
    dataloader, testdataloader,valdataloader, dim = load_data()
    model = load_model(dim,"test")
    model = load_model(dim)
    criterion = get_ii_loss()
    train_model(model,dataloader,testdataloader,criterion)

    # 创建一个母图
    plt.figure()

    # 在母图上添加第一个子图
    plt.subplot(1, 2, 1)  # 1行2列的第1个位置
    visualize(model,"Cross_entropy Loss",dataloader)

    # 在母图上添加第二个子图
    plt.subplot(1, 2, 2)  # 1行2列的第2个位置
    X_train, X_test, y_train, y_test, dim = load_data("test")
    visualize_test(model,"Test Cross_entropy Loss",X_test,y_test)
    # 显示最终的母图
    plt.show()