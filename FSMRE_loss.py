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
    loss = torch.tensor(0)
    single_acc=0.0
    multi_acc = 0.0
    multi_count=0.0
    for id in range(len(input)):
        _loss = torch.tensor(0)
        num = float(len(input[id]) * (len(input[id]) - 1) * label_num)
        multi_count+=len(input[id]) * (len(input[id]) - 1)
        for i in range(len(input[id])):
            for j in range(len(input[id][i])):
                if i == j:
                    continue
                flag=False
                for k in range(label_num):
                    _loss += cross_entropy(input[id][i][j][k][k], target[id][i][j][k])
                    if (input[id][i][j][k][k]==max(input[id][i][j][k]) and target[id][i][j][k]==1) or \
                            (input[id][i][j][k][k]!=max(input[id][i][j][k]) and target[id][i][j][k]==0):
                        single_acc+=1
                    else:
                        flag=True
                if not flag:
                    multi_acc+=1
        loss += _loss / num
    loss = loss / float(len(input))
    single_count = multi_count * label_num
    single_acc=single_acc/single_count
    multi_acc=multi_acc/multi_count
    return loss, single_acc, multi_acc

def cross_entropy(predict, ground):
    probability = (predict ** ground) * ((1 - predict) ** (1 - ground))
    return 0 - log(probability)
