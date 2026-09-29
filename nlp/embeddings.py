import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        x=[]
        for i in positive:
            for j in i.split():
                x.append(j)
        for i in negative:
            for j in i.split():
                x.append(j)
        gg=sorted(set(x))

        num=1.0
        dic={}
        for a in gg:
            dic[a]=num
            num+=1

        encoded=[]
        for s in positive+negative:
            ids=[]
            for word in s.split():
                ids.append(dic[word])
            encoded.append(torch.tensor(ids))

        return nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=0)