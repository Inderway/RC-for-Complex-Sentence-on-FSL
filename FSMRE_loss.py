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
    m_loss = []
    single_acc=0.0
    multi_acc = 0.0
    multi_count=0.0
    single_count=0.0
    for id in range(len(input)):
        __loss = []
        num = float(len(input[id]) * (len(input[id]) - 1) * label_num)
        multi_count+=len(input[id]) * (len(input[id]) - 1)
        for i in range(len(input[id])):
            for j in range(len(input[id][i])):
                if i == j:
                    continue
                flag=False
                for k in range(label_num):
                    __loss.append(cross_entropy(input[id][i][j][k][k], target[id][i][j][k]))
                    if target[id][i][j][k]==1:
                        single_count+=1
                        if input[id][i][j][k][k]==max(input[id][i][j][k]):
                            single_acc+=1
                    if (input[id][i][j][k][k]==max(input[id][i][j][k]) and target[id][i][j][k]==1) or \
                            (input[id][i][j][k][k]!=max(input[id][i][j][k]) and target[id][i][j][k]==0):
                        pass
                    else:
                        flag=True
                if not flag:
                    multi_acc+=1
        for idx in range(len(__loss)):
            __loss[idx] = torch.unsqueeze(__loss[idx], 0)
        _loss = torch.cat(__loss, 0).sum()
        m_loss.append(_loss/num)
    for idx in range(len(m_loss)):
        m_loss[idx] = torch.unsqueeze(m_loss[idx], 0)
    m_loss = torch.cat(m_loss, 0).sum()
    loss = m_loss / float(len(input))
    # single_count = multi_count * label_num
    single_acc=single_acc/single_count
    multi_acc=multi_acc/multi_count
    return loss, single_acc, multi_acc

def cross_entropy(predict, ground):
    probability = torch.pow(predict, ground) * torch.pow((1 - predict), (1 - ground))
    return 0 - torch.log(probability)
