import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import yaml, os
from core.hybrid_cnn_lstm import HybridCNNLSTM

class ModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.train_cfg = cfg['training']
        arch_cfg = cfg['architecture']
        self.model = HybridCNNLSTM(
            seq_len=self.train_cfg['sequence_length'],
            cnn_filters=tuple(arch_cfg['cnn_filters']),
            lstm_hidden=arch_cfg['lstm_hidden'],
            lstm_layers=arch_cfg['lstm_layers'],
            dropout=arch_cfg['dropout'],
            bidirectional=arch_cfg['bidirectional']
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, scalograms, sequences, targets):
        X1 = torch.tensor(scalograms, dtype=torch.float32)
        X2 = torch.tensor(sequences, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        ds = TensorDataset(X1, X2, y)
        val_size = int(len(ds)*self.train_cfg['validation_split'])
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=self.train_cfg['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.train_cfg['batch_size'])

        optim = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg['learning_rate'])
        crit = torch.nn.MSELoss()

        best_val = float('inf')
        for epoch in range(self.train_cfg['epochs']):
            self.model.train()
            for s1,s2,yy in train_loader:
                s1,s2,yy = s1.to(self.device), s2.to(self.device), yy.to(self.device)
                optim.zero_grad()
                pred = self.model(s1, s2)
                loss = crit(pred, yy)
                loss.backward()
                optim.step()
            # validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for s1,s2,yy in val_loader:
                    s1,s2,yy = s1.to(self.device), s2.to(self.device), yy.to(self.device)
                    pred = self.model(s1,s2)
                    val_loss += crit(pred, yy).item() * len(yy)
            val_loss /= len(val_ds)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), 'model_best.pth')
            print(f"Epoch {epoch+1}/{self.train_cfg['epochs']} - val_loss {val_loss:.4f}")
        # load best
        self.model.load_state_dict(torch.load('model_best.pth'))
        return self.model
