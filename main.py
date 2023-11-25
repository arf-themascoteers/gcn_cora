import torch
from torch_geometric.datasets import Planetoid
from cora_gcn import CoraGCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Planetoid(root=".", name='Cora')
data = dataset[0]

model = CoraGCN().to(device)
model.train_me(data)
model.test_me(data)
