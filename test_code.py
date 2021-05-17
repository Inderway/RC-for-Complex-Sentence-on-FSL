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
import xml.etree.cElementTree as ET
import os


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




# s="The School of Business and Social Sciences at the Aarhus University (Universitas Aarhusiensis, in Latin), was established in 1928 in Aarhus. The school has 737 academic staff and 1600 students, and is affiliated with the European University Association. Its dean is Thomas Pallesen."
# print(s.find("Thomas Pallesen"))




# relations={}
# dir='D:/Dataset/en'
# for root_, dirs, files in os.walk(dir):
#     for file in files:
#         file_dir=os.path.join(root_,file)
#         # print(file_dir)
#         tree = ET.ElementTree(file=file_dir)
#         root=tree.getroot()
#         root=root[0]
#         for entry in root:
#         # print((entry.tag, entry.attrib))
#             instance_num = 0
#             triples = []
#             for child in entry:
#                 if child.tag=='modifiedtripleset':
#                     for triple in child:
#                         tmp=triple.text.split('|')[1].strip()
#                         if tmp not in triples:
#                             triples.append(tmp)
#                 if child.tag=='lex':
#                     instance_num+=1
#             for triple in triples:
#                 if triple not in relations:
#                     relations[triple]=instance_num
#                 else:
#                     relations[triple]+=instance_num
# print(len(relations))
# cnt=0
# count=0
# for relation in relations:
#     count+=relations[relation]
#     if relations[relation]>50:
#         print("relation: {:50}, count: {}".format(relation, relations[relation]))
#         cnt+=1
# print(count)


dir='D:/Dataset/en/train/1triples'
cnt=0
for root_, dirs, files in os.walk(dir):
    for file in files:
        file_dir=os.path.join(root_,file)
        # print(file_dir)
        tree = ET.ElementTree(file=file_dir)
        root=tree.getroot()
        root=root[0]
        for entry in root:

        # print((entry.tag, entry.attrib))
            entities = []
            for child in entry:
                if child.tag=='modifiedtripleset':
                    for triple in child:
                        tmp_1=triple.text.split('|')[0].strip().replace('_', ' ')
                        tmp_2=triple.text.split('|')[2].strip().replace('_', ' ')
                        if tmp_1 not in entities:
                            entities.append(tmp_1)
                        if tmp_2 not in entities:
                            entities.append(tmp_2)
                if child.tag=='lex':
                    for entity in entities:
                        text=child.text
                        if text.find(entity)==-1:
                            print(text)
                            print(entity)
                            cnt += 1
                            break

print(cnt)


