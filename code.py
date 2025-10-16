 import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphSAGE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Example simulated data (replace this with your real dataset)
# Structure: Each row = learner interaction
np.random.seed(42)
data = pd.DataFrame({
    'source': np.random.randint(0, 200, 1000),  # learner IDs
    'target': np.random.randint(0, 200, 1000),
    'interaction_type': np.random.choice(['reply', 'endorsement', 'question'], 1000),
    'weight': np.random.uniform(0.1, 1.0, 1000)
})
data = data[data['source'] != data['target']]
G = nx.from_pandas_edgelist(
    data, 'source', 'target', ['interaction_type', 'weight'], create_using=nx.DiGraph()
)
# Example node features (can be replaced with real learner metrics)
for node in G.nodes():
    G.nodes[node]['activity_level'] = np.random.rand()
    G.nodes[node]['engagement'] = np.random.rand()
    G.nodes[node]['response_rate'] = np.random.rand()
# Feature matrix
node_features = []
for node in G.nodes():
    node_features.append([
        G.nodes[node]['activity_level'],
        G.nodes[node]['engagement'],
        G.nodes[node]['response_rate']
    ])
X = torch.tensor(node_features, dtype=torch.float)
# Edge index (for PyTorch Geometric)
node_map = {node: idx for idx, node in enumerate(G.nodes())}
edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
# Influence labels (synthetic ground truth for supervised learning)
# Example: based on in-degree + random noise
influence_labels = np.array([G.in_degree(n) + np.random.rand() for n in G.nodes()])
y = torch.tensor(influence_labels, dtype=torch.float)
indices = np.arange(len(G.nodes()))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_mask = torch.zeros(len(G.nodes()), dtype=torch.bool)
test_mask = torch.zeros(len(G.nodes()), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True
graph_data = Data(x=X, edge_index=edge_index, y=y,
                  train_mask=train_mask, test_mask=test_mask)
class InfluenceGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(InfluenceGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lin = torch.nn.Linear(out_channels, 1)
      def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x.squeeze()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InfluenceGCN(in_channels=3, hidden_channels=64, out_channels=32).to(device)
data = graph_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        model.eval()
        test_loss = F.mse_loss(out[data.test_mask], data.y[data.test_mask])
        print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")
model.eval()
with torch.no_grad():
    influence_pred = model(data).cpu().numpy()
# Attach influence scores to the graph
for i, node in enumerate(G.nodes()):
    G.nodes[node]['pred_influence'] = float(influence_pred[i])
# Show top influencers
top_nodes = sorted(G.nodes(data=True),
                   key=lambda x: x[1]['pred_influence'], reverse=True)[:10]
print("\nTop 10 Predicted Influential Learners:")
for n, d in top_nodes:
    print(f"Learner {n}: Influence Score = {d['pred_influence']:.3f}")
degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G)
betweenness = nx.betweenness_centrality(G)
# Normalize baselines
deg_values = np.array(list(degree_centrality.values()))
pr_values = np.array(list(pagerank.values()))
bt_values = np.array(list(betweenness.values()))
base_mean = (deg_values + pr_values + bt_values) / 3
# Evaluate improvement (synthetic metric: correlation with ground truth)
from scipy.stats import pearsonr
corr_gnn, _ = pearsonr(influence_pred, influence_labels)
corr_base, _ = pearsonr(base_mean, influence_labels)
improvement = ((corr_gnn - corr_base) / corr_base) * 100
print(f"\nBaseline correlation: {corr_base:.3f}")
print(f"GNN correlation: {corr_gnn:.3f}")
print(f"Improvement: {improvement:.2f}%")
def simulate_diffusion(G, seeds, steps=5, prob=0.2):
    """Independent Cascade diffusion simulation."""
    active = set(seeds)
    newly_active = set(seeds)
    for _ in range(steps):
        next_active = set()
        for node in newly_active:
            for nbr in G.successors(node):
                if nbr not in active and np.random.rand() < prob:
                    next_active.add(nbr)
        if not next_active:
            break
        active |= next_active
        newly_active = next_active
    return active
seed_nodes = [n for n, _ in top_nodes[:3]]
diffused = simulate_diffusion(G, seed_nodes, steps=5, prob=0.3)
print(f"\nKnowledge diffusion from seeds {seed_nodes}: {len(diffused)} learners reached.")
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
node_color = [d['pred_influence'] for _, d in G.nodes(data=True)]
nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.plasma, node_size=60)
nx.draw_networkx_edges(G, pos, alpha=0.2)
plt.title("Educational Influence Network (Node color = Predicted Influence)")
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma), label="Influence Score")
plt.axis("off")
plt.show()