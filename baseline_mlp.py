import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

# Load Cora dataset
dataset = Planetoid(root="data/Cora", name="Cora")
data = dataset[0]

print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Features per node:", dataset.num_node_features)
print("Classes:", dataset.num_classes)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = (pred[mask] == data.y[mask]).sum() / mask.sum()
        accs.append(acc.item())
    return accs

for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Test Acc {test_acc:.4f}"
        )

