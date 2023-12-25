import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

# 1. 构建数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 假设你有一个包含开集的训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_data, open_set_data= train_test_split(train_dataset, train_size=0.8, random_state=42, stratify=train_dataset.targets)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
open_set_loader = DataLoader(open_set_data, batch_size=64, shuffle=False)

# 2. 构建模型
class OpenMaxModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OpenMaxModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 3. 定义 OpenMax 损失函数
class OpenMaxLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(OpenMaxLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predicted_scores, known_class_scores, unknown_class_scores):
        # 计算 softmax 的对数概率
        log_probs_known = torch.log(predicted_scores)
        log_probs_unknown = torch.log(1.0 - predicted_scores)

        # 计算 OpenMax 损失
        openmax_loss = -torch.mean(log_probs_known) - self.alpha * torch.mean(log_probs_unknown)

        return openmax_loss

# 4. 训练模型
input_size = 28 * 28  # MNIST 图像大小
hidden_size = 256
output_size = 10  # 10 类别用于训练
model = OpenMaxModel(input_size, hidden_size, output_size)
criterion = OpenMaxLoss(alpha=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.1)

def train_openmax_model(model, train_loader, optimizer, criterion, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1)  # 将图像数据展平
            optimizer.zero_grad()
            outputs = model(inputs)
            known_class_scores = outputs[:, :output_size-1]  # 最后一列为未知类别
            unknown_class_scores = outputs[:, output_size-1:]
            loss = criterion(outputs[:, :-1], known_class_scores, unknown_class_scores)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss/len(train_loader)}")

train_openmax_model(model, train_loader, optimizer, criterion, epochs=5)

# 5. 在开集上测试
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in open_set_loader:
        inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data[:, :-1], 1)  # 仅考虑已知类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on Open Set: {correct / total}")
