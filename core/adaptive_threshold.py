import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from collections import deque
import yaml

class AdaptiveThresholdOptimizer:
    """Adaptive thresholding using ML or statistical rules."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg_root = yaml.safe_load(f)
        self.cfg = cfg_root["threshold_optimization"]
        self.method = self.cfg["method"]
        self.window = self.cfg["window_size"]
        self.target_fpr = self.cfg["target_false_positive_rate"]
        self.history = deque(maxlen=self.window)
        self.adaptive_factor = 1.0
        self.current_threshold = self.cfg["initial_threshold"]

        # Pick model
        if self.method == "isolation_forest":
            iso_cfg = self.cfg["isolation_forest"]
            self.model = IsolationForest(
                contamination=iso_cfg["contamination"],
                n_estimators=iso_cfg["n_estimators"],
                random_state=42,
            )
        elif self.method == "one_class_svm":
            svm_cfg = self.cfg["one_class_svm"]
            self.model = OneClassSVM(nu=svm_cfg["nu"], kernel=svm_cfg["kernel"])
        else:
            self.model = None  # statistical

    # ---------------------------------------------------------
    def update(self, residual: float):
        """Add residual and retrain periodically."""
        self.history.append(abs(residual))
        if len(self.history) < self.cfg["min_samples"]:
            return
        if len(self.history) % self.cfg["retrain_interval"] == 0:
            self._retrain()

    # ---------------------------------------------------------
    def _retrain(self):
        X = np.array(self.history).reshape(-1, 1)
        if self.method == "statistical" or self.model is None:
            mean, std = np.mean(X), np.std(X)
            self.current_threshold = mean + self.cfg["statistical"]["n_sigma"] * std
            return
        # ML based
        self.model.fit(X)
        scores = self.model.decision_function(X)
        # Higher score => more normal. Convert to anomaly distance.
        thresh_percentile = 100 * (1 - self.target_fpr)
        self.current_threshold = np.percentile(scores, thresh_percentile)

    # ---------------------------------------------------------
    def threshold(self):
        return self.current_threshold * self.adaptive_factor

    # ---------------------------------------------------------
    def adapt(self, tp, fp, tn, fn):
        """Adjust factor based on confusion counts."""
        if (fp + tn) == 0:
            return
        fpr = fp / (fp + tn)
        if fpr > self.target_fpr * 1.2:
            self.adaptive_factor *= 1.05
        elif fpr < self.target_fpr * 0.8:
            self.adaptive_factor *= 0.95
        self.adaptive_factor = np.clip(self.adaptive_factor, 0.5, 2.0)
