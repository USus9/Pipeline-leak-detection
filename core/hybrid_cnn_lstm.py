import torch
import torch.nn as nn

class HybridCNNLSTM(nn.Module):
    """CNN on scalogram + LSTM on raw sequence."""
    def __init__(self, seq_len: int, cnn_filters=(32, 64), lstm_hidden=128, lstm_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        cnn_out_features = cnn_filters[1]
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_features + lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # regression output (VB prediction)
        )

    def forward(self, scalogram: torch.Tensor, sequence: torch.Tensor):
        # scalogram shape (B,1,H,W)
        cnn_feat = self.cnn(scalogram).flatten(1)
        # sequence shape (B, seq_len)
        seq = sequence.unsqueeze(-1)  # (B, seq_len, 1)
        lstm_out, _ = self.lstm(seq)
        lstm_feat = lstm_out[:,-1,:]
        concat = torch.cat([cnn_feat, lstm_feat], dim=1)
        return self.fc(concat).squeeze(1)
