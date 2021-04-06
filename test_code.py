# coding=utf-8
# created at 2021/4/4
# used for learning and test
import torch
from torch_geometric.data import Data

device=torch.device('cuda')
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