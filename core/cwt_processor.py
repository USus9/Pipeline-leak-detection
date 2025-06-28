import numpy as np
import pywt
import torch

class CWTProcessor:
    """Continuous Wavelet Transform helper producing scalograms."""
    def __init__(self, wavelet: str = 'morl', scales=None):
        self.wavelet = wavelet
        self.scales = scales if scales is not None else np.arange(1, 65)

    def scalogram(self, signal_1d: np.ndarray) -> torch.Tensor:
        coef, _ = pywt.cwt(signal_1d, self.scales, self.wavelet)
        # power scalogram
        scal = np.abs(coef)**2
        # normalise 0-1
        scal = (scal - scal.min()) / (scal.max() - scal.min() + 1e-8)
        return torch.tensor(scal, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
