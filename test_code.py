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



device=torch.device('cuda')

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

# GCN on Cora
# dataset=Planetoid(root='data/tmp/Cora', name='Cora')

# two-layer GCN
# class MyDataset:
#     def __init__(self):
#         super(MyDataset, self).__init__()
#         self.num_classes=3
#         self.num_edges=3
#         self.num_node_features=3
#         self.x=torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [0.4, 0.5, 0.6]]).to(device)
#         self.edge_index=torch.tensor([[0, 1, 0, 2, 1, 2],[1, 0, 2, 0, 2, 1]]).to(device)
#         self.y=torch.tensor([0,1,2]).to(device)
#
#
# dataset=MyDataset()
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1=GCNConv(dataset.num_node_features, dataset.num_classes)
#         # self.conv2=GCNConv(16, dataset.num_classes)
#
#         self.edge_weight=torch.tensor([1, 1, 100, 100, 1, 1], dtype=torch.float).to(device)
#         # print("---------------------------egde's shape")
#         # print(self.edge_weight.shape)
#
#     def forward(self, data):
#         x, edge_index=data.x, data.edge_index
#
#
#         x=self.conv1(x, edge_index, edge_weight=self.edge_weight)
#         # x=F.relu(x)
#         # x=F.dropout(x, training=self.training)
#         # x=self.conv2(x, edge_index,self.edge_weight)
#         return x
#
# data=dataset
# model=Net().to(device)
# optimizer=torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#
# model.train()
# for epoch in range(10):
#     print("epoch:{}===========================".format(epoch))
#     optimizer.zero_grad()
#     out=model(data)
#     # print("------------------------x")
#     # print(data.x)
#     print("-----------------------out")
#     print(out)
#     loss=F.nll_loss(out, data.y)
#     loss.backward()
#     optimizer.step()

# model.eval()
# _, pred=model(data).max(dim=1)
# correct=int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc=correct/int(data.test_mask.sum())
# print('Accuracy: {:.4f}'.format(acc))


#
d=[2,2,1,2,2]
l=[1, 1, 0, 1, 1]
r=[[4,5], [5,6]]
prediction=[[[[0 for m in range(5)] for k in range(4)] for j in range(4)] for i in range(2)]
prediction=torch.tensor(prediction)
print(prediction.shape)




