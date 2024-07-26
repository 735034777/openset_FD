"""
File: CWRU_shape_exp
Author: admin
Date Created: 2024/7/10
Last Modified: 2024/7/10

Description:
    Describe the file purpose and main functionality.


"""

import torch
import torch.nn as nn

# 定义模型类
class CustomModel(nn.Module):
    def __init__(self, dim):
        super(CustomModel, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(5, 5), stride=(2, 2)), # Adjust kernel size and stride
            nn.Tanh(),
            nn.Conv2d(50, 30, kernel_size=(5, 5), stride=(2, 2)), # Adjust kernel size and stride
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(3, 3), stride=(1, 1)), # Adjust kernel size and stride
            nn.Tanh(),
            nn.Conv2d(50, 40, kernel_size=(3, 3), stride=(1, 1)), # Adjust kernel size and stride
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2)), # Adjust kernel size
            nn.Conv2d(40, 30, kernel_size=(3, 3), stride=(1, 1)), # Adjust kernel size and stride
            nn.Tanh(),
            nn.Conv2d(30, 30, kernel_size=(3, 3), stride=(2, 2)), # Adjust kernel size and stride
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.lstm1 = nn.LSTM(1350, 120, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(120, 60, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_origin = x
        x = x.unsqueeze(1)  # Add a channel dimension
        print("Input shape:", x.shape)

        print("=== Encoder 1 ===")
        for i, layer in enumerate(self.encoder1):
            x = layer(x)
            print(f"Layer {i} output shape: {x.shape}")

        ec1_outputs = x
        x = x_origin.unsqueeze(1)  # Reset to initial input for encoder2

        print("=== Encoder 2 ===")
        for i, layer in enumerate(self.encoder2):
            x = layer(x)
            print(f"Layer {i} output shape: {x.shape}")

        ec2_outputs = x

        # Reshape encoder outputs to flatten them
        ec1_outputs = ec1_outputs.view(ec1_outputs.size(0), -1)
        ec2_outputs = ec2_outputs.view(ec2_outputs.size(0), -1)
        print("Flattened Encoder1 output shape:", ec1_outputs.shape)
        print("Flattened Encoder2 output shape:", ec2_outputs.shape)

        # Match sizes of the two encoder outputs
        min_size = min(ec1_outputs.size(1), ec2_outputs.size(1))
        ec1_outputs = ec1_outputs[:, :min_size]
        ec2_outputs = ec2_outputs[:, :min_size]

        # Element-wise multiplication
        encoder = torch.mul(ec1_outputs, ec2_outputs)
        print("Combined encoder output shape:", encoder.shape)

        # LSTM processing
        lstm_out, _ = self.lstm1(encoder.unsqueeze(0))
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out.squeeze(0)
        print("LSTM output shape:", lstm_out.shape)

        # Fully connected and softmax
        fc_out2 = self.fc2(lstm_out)
        fc_out = self.softmax(fc_out2)
        print("Final output shape:", fc_out.shape)

        fc_out1 = torch.tensor([])  # Unused tensor, just for placeholder
        return fc_out, fc_out2, fc_out1

# 创建模型实例
model = CustomModel(dim=10)

# 生成数据
input_data = torch.randn(64, 50,82)

# 运行模型
outputs = model(input_data)

