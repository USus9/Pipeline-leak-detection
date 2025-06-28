import torch
import xgboost as xgb  # retained for feature importance if needed
from core.adaptive_threshold import AdaptiveThresholdOptimizer

class LeakDetector:
    def __init__(self, model, cfg_path: str, feature_names, cwt_processor):
        self.model = model.eval()
        self.threshold_opt = AdaptiveThresholdOptimizer(cfg_path)
        self.feature_names = feature_names
        self.cwt = cwt_processor
        self.device = next(model.parameters()).device
        self.tp = self.fp = self.tn = self.fn = 0

    def process(self, feature_vector, actual_vb):
        # feature_vector : 1D numpy length sequence_len (latest window) ; uses VB series separate param
        seq = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        scal = self.cwt.scalogram(feature_vector).unsqueeze(0).to(self.device)  # (1,1,H,W)
        with torch.no_grad():
            pred_vb = self.model(scal, seq).item()
        residual = actual_vb - pred_vb
        self.threshold_opt.update(residual)
        gap = self.threshold_opt.threshold()
        leak = residual > gap
        # Update confusion (needs truth; we assume leak if residual > gap)
        if leak:
            self.fp +=1  # placeholder (can't know truth here)
        else:
            self.tn +=1
        if (self.tp + self.fp + self.tn + self.fn) % 100 == 0:
            self.threshold_opt.adapt(self.tp, self.fp, self.tn, self.fn)
        return leak, pred_vb, gap, residual
