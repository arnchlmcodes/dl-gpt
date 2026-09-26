import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        
        n=len(y_true)
        s=0
        for i in range(n):
            p = min(max(y_pred[i], 1e-7), 1 - 1e-7)
            s += y_true[i] * math.log(p) + (1 - y_true[i]) * math.log(1 - p)
        return round(-s/n,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        n=len(y_true)
        s=0
        for i in range(n):
            for j in range(len(y_true[i])):
                p = min(max(y_pred[i][j], 1e-7), 1 - 1e-7)
                s += y_true[i][j] * math.log(p)
        return round(-s/n,4)

