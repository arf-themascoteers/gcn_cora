import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CoraGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """ GCNConv layers """
        self.conv1 = GCNConv(1433, 16)
        self.conv2 = GCNConv(16, 7)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    @staticmethod
    def compute_accuracy(pred_y, y):
        return (pred_y == y).sum()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def train_me(self, data):
        print("Start training")
        self.train()
        losses = []
        accuracies = []
        for epoch in range(200):
            self.optimizer.zero_grad()
            out = self(data)

            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            correct = CoraGCN.compute_accuracy(out.argmax(dim=1)[data.train_mask], data.y[data.train_mask])
            acc = int(correct) / int(data.train_mask.sum())
            losses.append(loss.item())
            accuracies.append(acc)

            loss.backward()
            self.optimizer.step()

            print('Epoch: {}, Loss: {:.4f}, Training Accuuracy: {:.4f}'.format(epoch + 1, loss.item(), acc))

    def test_me(self, data):
        print("Start Testing")
        self.eval()
        out = self(data)
        correct = CoraGCN.compute_accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        acc = int(correct) / int(data.test_mask.sum())
        print("Test Accuracy",acc)
