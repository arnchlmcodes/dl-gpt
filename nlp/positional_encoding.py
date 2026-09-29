import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        pos=np.arange(seq_len).reshape(seq_len,1)
        pair = np.arange(d_model) // 2
        angle = pos / (10000 ** (2 * pair / d_model))
        op = np.empty((seq_len, d_model))   
        for i in range(seq_len):
            for j in range(d_model):
                if j % 2 == 0:
                    op[i][j] = np.sin(angle[i][j])
                else:
                    op[i][j] = np.cos(angle[i][j])
        return np.round(op,5)