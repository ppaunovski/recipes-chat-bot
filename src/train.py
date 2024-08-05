import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from torch_geometric.nn import to_hetero
from torch_geometric.nn import Linear, SAGEConv
from torch.optim import SGD
from torch_geometric.datasets import AmazonBook
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import classification_report, roc_auc_score
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader
from torch_geometric.nn import LightGCN
from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, start, to):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        self.start = start
        self.to = to

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict[self.start][row], z_dict[self.to][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
  def __init__(self, hidden_channels, data, start, to):
      super().__init__()
      self.encoder = GNNEncoder(hidden_channels, hidden_channels)
      self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
      self.decoder = EdgeDecoder(hidden_channels=hidden_channels, start=start, to=to)

  def forward(self, x_dict, edge_index_dict, edge_label_index):
      z_dict = self.encoder(x_dict, edge_index_dict)
      return self.decoder(z_dict, edge_label_index)
  

def train_link_prediction(model, train_data, val_data, optimizer, start, to, epochs=5):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict,
                     train_data[start,'has_sub', to].edge_label_index)

        target = train_data[start,'has_sub', to].edge_label
        loss = mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
          pred = model(val_data.x_dict, val_data.edge_index_dict,
                      val_data[start,'has_sub', to].edge_label_index)
        pred = pred.clamp(min=0, max=1)
        target = val_data[start,'has_sub', to].edge_label.float()
        val_loss = mse_loss(pred, target).sqrt()

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

def test_link_prediction(model, test_data, start, to):
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        pred = model(test_data.x_dict, test_data.edge_index_dict,
                     test_data[start, 'has_sub', to].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = test_data[start, 'has_sub', to].edge_label.float()

    y_true = target.cpu().numpy()
    y_pred = pred.round().detach().cpu().numpy()

    print(classification_report(y_true=y_true, y_pred=y_pred, digits=4))

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f'ROC AUC: {roc_auc:.4f}')

    val_loss = mse_loss(pred, target).sqrt()
    print(f'Loss: {val_loss:.4f}')
    

def train_lightgcn(dataset, train_loader, model, optimizer, num_ingrs, epochs=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Print dataset attributes for debugging
    print("Dataset attributes:", dataset.__dict__)

    for epoch in range(epochs):
        total_loss, total_examples = 0, 0

        for node_ids in train_loader:
            # Use edge_index instead of edge_label_index if it's not present
            pos_edge_label_index = dataset.edge_index[:, node_ids]
            generated = torch.randint(0, num_ingrs, (node_ids.numel(),)).to(device)

            # Ensure generated indices are within bounds
            generated = torch.clamp(generated, 0, num_ingrs - 1)

            neg_edge_label_index = torch.stack([pos_edge_label_index[0],
                                                generated],
                                               dim=0)

            edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)

            # Check if any index in edge_label_index exceeds the bounds
            if edge_label_index.max() >= num_ingrs:
                print(f"Warning: Index out of bounds detected in edge_label_index with max value {edge_label_index.max()}")

            optimizer.zero_grad()

            pos_rank, neg_rank = model(dataset.edge_index.to(device), edge_label_index.to(device)).chunk(2)

            loss = model.recommendation_loss(pos_rank, neg_rank, node_id=edge_label_index.unique())
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pos_rank.numel()
            total_examples += pos_rank.numel()

            print(f'Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}')