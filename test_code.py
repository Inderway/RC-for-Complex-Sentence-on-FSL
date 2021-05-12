# coding=utf-8
# created at 2021/4/4
# used for learning and test
# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

# device=torch.device('cuda')

'''
x=torch.tensor([[-1], [0], [1]], dtype=torch.float)
# 定义两次边以表示无向
edge_index=torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)

# 另一种写法，每个元组表示两个节点，但该写法要转成上面的形式
edge_index2=torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
# data为dictionary
data=Data(x=x, edge_index=edge_index2.t().contiguous())
#data=data.to(device)

print(data)

from torch_geometric.datasets import TUDataset
dataset=TUDataset(root='data/tmp/ENZYMES', name='ENZYMES')
print(len(dataset)) # 共600个图
print(dataset.num_classes)
print(dataset.num_node_features)
data=dataset[0]
# graph=有37个点，每个顶点有3个特征，共168条边，分类为1
print(data)


# 洗牌
dataset=dataset.shuffle()
# 等同于上述操作
perm=torch.randperm(len(dataset))
dataset=dataset[perm]


#分割
train_dataset=dataset[:540]
test_dataset=dataset[540:]

'''
'''
# 使用半监督数据集Cora
from torch_geometric.datasets import Planetoid
# 该数据集仅包含一个无向图
# 共5278条边, 2708个1433个特征的点，每个结点一个label
# 并分出140个结点用于train, 500用于验证, 1000用于测试
dataset=Planetoid(root='data/tmp/Cora',name='Cora')
'''

# # GCN on Cora
# dataset=Planetoid(root='data/tmp/Cora', name='Cora')
#
# # two-layer GCN
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1=GCNConv(dataset.num_node_features, 16)
#         self.conv2=GCNConv(16, dataset.num_classes)
#
#     def forward(self, data):
#         x, edge_index=data.x, data.edge_index
#         x=self.conv1(x, edge_index)
#         x=F.relu(x)
#         x=F.dropout(x, training=self.training)
#         x=self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)
#
# model=Net().to(device)
# data=dataset[0].to(device)
# optimizer=torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#
# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out=model(data)
#     loss=F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#
# model.eval()
# _, pred=model(data).max(dim=1)
# correct=int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc=correct/int(data.test_mask.sum())
# print('Accuracy: {:.4f}'.format(acc))

d={1:1, 2:2, 3:3}
l=[1, 2, 3, 4, 1, 2]
r=1
l=torch.tensor(l)
r=torch.tensor(r)
print(r.item())


