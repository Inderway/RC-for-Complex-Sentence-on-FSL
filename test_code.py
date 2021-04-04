# created at 2021/4/4
# used for learning and test
import torch
from torch_geometric.data import Data

x=torch.tensor([[-1], [0], [1]], dtype=torch.float)
# 定义两次边以表示无向
edge_index=torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)

# 另一种写法，每个元组表示两个节点，但该写法要转成上面的形式
edge_index2=torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
data=Data(x=x, edge_index2=edge_index2.t().contiguous())
