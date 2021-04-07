# created at 2021/4/7
# GCN

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

device=torch.device('cuda')
dataset={
    'num_node_features':1,
    'num_classes':3,
    'x': [[2], [1], [1.5]],
    'edge_index':[[0,1,0,2,1,2],[1,0,2,0,2,1]],
    'edge_weight':[1,1,2,2,3,3],
    'y':[0, 1, 2],
}

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1=GCNConv(dataset['num_node_features'], dataset['num_classes'])

    def forward(self, data):
        x, edge_index=torch.tensor(data['x'],dtype=torch.float), torch.tensor(data['edge_index'],dtype=torch.long)
        x=self.conv1(x, edge_index,edge_weight=torch.tensor(data['edge_weight'],dtype=torch.float))
        return F.log_softmax(x,dim=0)

model=GCN()
data=dataset
optimizer=torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out=model(data)
    print('---epoch {:d}---\nout:'.format(epoch))
    print(out)
    loss=F.nll_loss(out, torch.tensor(data['y'],dtype=torch.long))
    loss.backward()
    optimizer.step()


