import os, sys, pickle, glob
# import os.path as path
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

import torch
import torch.nn as nn
import src.config as config

class CustomModel(nn.Module):
    def __init__(self,dim):
        super(CustomModel, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, 50, kernel_size=20, stride=2),
            nn.Tanh(),
            nn.Conv1d(50, 30, kernel_size=10, stride=2),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(1, 50, kernel_size=6, stride=1),
            nn.Tanh(),
            nn.Conv1d(50, 40, kernel_size=6, stride=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(40, 30, kernel_size=6, stride=1),
            nn.Tanh(),
            nn.Conv1d(30, 30, kernel_size=6, stride=2),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm1 = nn.LSTM(3690, 120, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(120, 60, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(120, 30)
        self.fc2 = nn.Linear(120,dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension (batch_size, 1, 250)

        ec1_outputs = self.encoder1(x)
        ec2_outputs = self.encoder2(x)

        # Reshape encoder outputs
        ec1_outputs = ec1_outputs.view(ec1_outputs.size(0), -1)
        ec2_outputs = ec2_outputs.view(ec2_outputs.size(0), -1)
        # 在维度 1 上填充较小的张量，以匹配大小
        min_size = min(ec1_outputs.size(1), ec2_outputs.size(1))
        ec1_outputs = ec1_outputs[:, :min_size]
        ec2_outputs = ec2_outputs[:, :min_size]

        # Element-wise multiplication
        encoder = torch.mul(ec1_outputs, ec2_outputs)

        # LSTM
        lstm_out, _ = self.lstm1(encoder.unsqueeze(0))
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out.squeeze(0)

        # Global pooling
        # lstm_out = torch.mean(lstm_out, dim=2)

        # Fully connected layers
        # fc_out1 = self.fc1(lstm_out)
        fc_out2 = self.fc2(lstm_out)
        fc_out = self.softmax(fc_out2)
        fc_out1 = torch.tensor([])
        # if config.HIGH_DIMENSION_OUTPUT == True:
        #     #使用高维向量是只输出高维激活向量
        #     fc_out2 = torch.tensor([])#
        return fc_out,fc_out2,fc_out1

# # 创建模型实例
#
# model = CustomModel()
