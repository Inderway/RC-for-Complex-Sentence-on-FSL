import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from math import log

class FSMRELoss(Module):
    def __init__(self, label_num):
        super(FSMRELoss, self).__init__()
        self.label_num=label_num

    def forward(self, input, target):
        return get_loss(input, target, self.label_num)

def get_loss(input, target, label_num):
    loss = 0
    for id in range(len(input)):
        _loss = 0
        num = float(len(input[id]) * (len(input[id]) - 1) * label_num)
        for i in range(len(input[id])):
            for j in range(len(input[id][i])):
                if i == j:
                    continue
                for k in range(label_num):
                    _loss += cross_entropy(input[id][i][j][k], target[id][i][j][k])
        loss += _loss / num
    loss = loss / float(len(input))
    return torch.tensor(loss)

def cross_entropy(predict, ground):
    probability = (predict ** ground) * ((1 - predict) ** (1 - ground))
    return 0 - log(probability)
