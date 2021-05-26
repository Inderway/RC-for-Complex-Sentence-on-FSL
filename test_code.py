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
import json
from transformers import BertTokenizer
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

#
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# ids=[[ 1448,  1104,  1103,  1211,  2712,   117,  1499,  1268,   117,  1110,
#           1121,  9344,  1107,  2123,   117, 25839,  1103,  2761,   112,   188,
#           8250,  2963,   118, 13559, 15402,  1115,  1127,  5624,  1118,  2490,
#            117,  1259,  6907,  1116,   119,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0],
#         [ 1960,  1168,  5432,  6683,   117,  3215,  4978, 20847,  1116,  1104,
#           3640,  4342,  1105,  7319,  1287,   139,   119, 23360,  1104,  1537,
#          13709,   117,  1138,  1145,  4151,  1103,  1933,   117,  1112,  1138,
#           1242,  1469,  6670,  1105, 12296,  1105,  4801,  2114,  1107,  1103,
#           1160,  2231,   119,     0,     0,     0,     0,     0,     0,     0,
#              0]]
#
# tokens=[[],[]]
# for i, id in enumerate(ids):
#     for j in id:
#         tokens[i].append(tokenizer.convert_ids_to_tokens(j))
#         print(tokens[i])
# sentencses=[[],[]]
# for i, token in enumerate(tokens):
#     sentencses[i]=tokenizer.decode(token)
# print(sentencses)

#

data=json.load(open('data/webnlg.json'))
# count1=0
# count2=0
# for r in data:
#     count1+=len(data[r])
# for r in data2:
#     count2+=len(data2[r])
# print(count1)
# print(count2)

res=[]
# for class_ in data:
#     for id, sentence_dic in enumerate(data[class_]):
#         entities=[]
#         for entity in sentence_dic['entityMentions']:
#             if entity['text'] not in entities:
#                 entities.append(entity['text'])
#         for relation in sentence_dic["relationMentions"]:
#             if relation['em1Text'] not in entities or relation['em2Text'] not in entities:
#                 res.append((class_, sentence_dic))
#                 break
#
#
# # tokenzier = BertTokenizer.from_pretrained('bert-base-cased')
# # for class_ in data:
# #     for id, sentence_dic in enumerate(data[class_]):
# #         entities=[]
# #         sentence_token = tokenzier.convert_tokens_to_ids(tokenzier.tokenize(sentence_dic['sentText']))
# #         for entity in sentence_dic['entityMentions']:
# #             if entity['text'] not in entities:
# #                 entities.append(entity['text'])
# #         for entity in entities:
# #             token = tokenzier.convert_tokens_to_ids(tokenzier.tokenize(entity))
# #             flag=False
# #             for j in range(len(sentence_token) + 1 - len(token)):
# #                 if sentence_token[j: j + len(token)] == token:
# #                     flag=True
# #             if not flag:
# #                 res.append((class_, sentence_dic))
# #                 break
#
#
#
#
#
#print(len(res))
#
print(len(data))

# #
# for ele in res:
#     rel=ele[0]
#     sent=ele[1]
#     data[rel].remove(sent)
# #
# with open('data/webnlg_4.json', 'w') as json_file:
#     json.dump(data, json_file)







