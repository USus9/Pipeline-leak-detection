import pandas as pd
import numpy as np
from core.cwt_processor import CWTProcessor

class DataPipeline:
    def __init__(self, numeric_features, sequence_length=128):
        self.numeric_features = numeric_features
        self.seq_len = sequence_length
        self.cwt = CWTProcessor()

    def load(self, path):
        df = pd.read_csv(path)
        df['LocalTime'] = pd.to_datetime(df['LocalTime'], errors='coerce')
        df = df.dropna(subset=['LocalTime'])
        return df.sort_values('LocalTime').reset_index(drop=True)

    def prepare_sequences(self, df):
        scalograms, sequences, targets = [], [], []
        vb_series = df['VB'].values
        for i in range(self.seq_len, len(df)):
            window_vb = vb_series[i-self.seq_len:i]
            scal = self.cwt.scalogram(window_vb)
            scalograms.append(scal.numpy())
            sequences.append(window_vb)
            targets.append(vb_series[i])
        return np.array(scalograms), np.array(sequences), np.array(targets)
