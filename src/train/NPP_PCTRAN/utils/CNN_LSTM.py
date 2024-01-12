import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, dim):
        super(CustomModel, self).__init__()

        # Adjusted for 2D input
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(5, 5), stride=(2, 2)), # Adjust kernel size and stride
            nn.Tanh(),
            nn.Conv2d(50, 30, kernel_size=(5, 5), stride=(2, 2)), # Adjust kernel size and stride
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2)) # Adjust kernel size
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
            nn.MaxPool2d(kernel_size=(2, 2)) # Adjust kernel size
        )

        # LSTM and other layers remain the same
        self.lstm1 = nn.LSTM(1350, 120, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(120, 60, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension (batch_size, 1, 50, 82)

        ec1_outputs = self.encoder1(x)
        ec2_outputs = self.encoder2(x)

        # Reshape for LSTM input
        # This part will need adjustment based on the output shape of your encoders
        ec1_outputs = ec1_outputs.view(ec1_outputs.size(0), -1)
        ec2_outputs = ec2_outputs.view(ec2_outputs.size(0), -1)

        # Matching sizes of the outputs from both encoders
        min_size = min(ec1_outputs.size(1), ec2_outputs.size(1))
        ec1_outputs = ec1_outputs[:, :min_size]
        ec2_outputs = ec2_outputs[:, :min_size]

        # Element-wise multiplication
        encoder = torch.mul(ec1_outputs, ec2_outputs)

        # LSTM layers
        lstm_out, _ = self.lstm1(encoder.unsqueeze(0))
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out.squeeze(0)

        # Fully connected layer
        fc_out2 = self.fc2(lstm_out)
        fc_out = self.softmax(fc_out2)
        fc_out1 = torch.tensor([])

        return fc_out, fc_out2, fc_out1
