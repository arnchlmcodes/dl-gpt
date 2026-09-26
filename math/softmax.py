import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        res=[]
        z=z-np.max(z)
        s=0
        s=sum(np.exp(z))
        for i in z:
            res.append(np.exp(i)/s)
        return np.round(np.array(res),4)
        