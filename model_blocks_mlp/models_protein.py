import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINEConv, global_add_pool, GINConv
from torch.nn import BatchNorm1d, Linear, ReLU, Dropout, Sequential
import torch.nn as nn

class GCN_GAT_GINE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_GAT_GINE, self).__init__()
        self.num_classes = num_classes
        # GCN-representation
        self.conv1 = GCNConv(num_node_features, 128, cached=False)
        self.bn01 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn02 = BatchNorm1d(64)
        self.conv3 = GCNConv(64, 32, cached=False)
        self.bn03 = BatchNorm1d(32)
        # GAT-representation
        self.gat1 = GATConv(num_node_features, 128, heads=8)
        self.bn11 = BatchNorm1d(128 * 8)
        self.gat2 = GATConv(128 * 8, 64, heads=8)
        self.bn12 = BatchNorm1d(64 * 8)
        self.gat3 = GATConv(64 * 8, 32, heads=8)
        self.bn13 = BatchNorm1d(32 * 8)
        # GIN-representation
        fc_gin1 = Sequential(Linear(num_node_features, 128), ReLU(), Linear(128, 128))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(128)
        fc_gin2 = Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(64)
        fc_gin3 = Sequential(Linear(64, 32), ReLU(), Linear(32, 32))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(32)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32 + 32 * 8 + 32, 128)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.fc2 = Linear(128, 64)
        self.dropout2 = Dropout(p=0.2)
        self.fc3 = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_add_pool(x, batch)
        # GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_add_pool(y, batch)
        # GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_add_pool(z, batch)
        # Concatenating representations
        cr = torch.cat((x, y, z), 1)
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        return cr.view(-1, self.num_classes)
    
class GCN_GAT_GINE_FIXED_OUT_DIM(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_GAT_GINE_FIXED_OUT_DIM, self).__init__()
        self.num_classes = num_classes
        # GCN-representation
        self.conv1 = GCNConv(num_node_features, 128, cached=False)
        self.bn01 = BatchNorm1d(128, track_running_stats=False)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn02 = BatchNorm1d(64, track_running_stats=False)
        self.conv3 = GCNConv(64, 32, cached=False)
        self.bn03 = BatchNorm1d(32, track_running_stats=False)
        # GAT-representation
        self.gat1 = GATConv(num_node_features, 128, heads=8)
        self.bn11 = BatchNorm1d(128 * 8, track_running_stats=False)
        self.gat2 = GATConv(128 * 8, 64, heads=8)
        self.bn12 = BatchNorm1d(64 * 8, track_running_stats=False)
        self.gat3 = GATConv(64 * 8, 32, heads=8)
        self.bn13 = BatchNorm1d(32 * 8, track_running_stats=False)
        # GIN-representation
        fc_gin1 = Sequential(Linear(num_node_features, 128), ReLU(), Linear(128, 128))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(128, track_running_stats=False)
        fc_gin2 = Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(64, track_running_stats=False)
        fc_gin3 = Sequential(Linear(64, 32), ReLU(), Linear(32, 32))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(32, track_running_stats=False)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32 + 32 * 8 + 32, 128)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.fc2 = Linear(128, 64)
        self.dropout2 = Dropout(p=0.2)
        self.fc3 = Linear(64, 32)

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_add_pool(x, batch)
        # GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_add_pool(y, batch)
        # GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_add_pool(z, batch)
        # Concatenating representations
        cr = torch.cat((x, y, z), 1)
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        return cr.view(-1, 32)
            
class GCN_GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_GAT, self).__init__()
        self.num_classes = num_classes
        # GCN-representation
        self.conv1 = GCNConv(num_node_features, 128, cached=False)
        self.bn01 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn02 = BatchNorm1d(64)
        self.conv3 = GCNConv(64, 32, cached=False)
        self.bn03 = BatchNorm1d(32)
        # GAT-representation
        self.gat1 = GATConv(num_node_features, 128, heads=8)
        self.bn11 = BatchNorm1d(128 * 8)
        self.gat2 = GATConv(128 * 8, 64, heads=8)
        self.bn12 = BatchNorm1d(64 * 8)
        self.gat3 = GATConv(64 * 8, 32, heads=8)
        self.bn13 = BatchNorm1d(32 * 8)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32 + 32 * 8, 128)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.fc2 = Linear(128, 64)
        self.dropout2 = Dropout(p=0.2)
        self.fc3 = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_add_pool(x, batch)
        # GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_add_pool(y, batch)
        # Concatenating representations
        cr = torch.cat((x, y), 1)
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        return cr.view(-1, self.num_classes)

class GCN_GINE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_GINE, self).__init__()
        self.num_classes = num_classes
        # GCN-representation
        self.conv1 = GCNConv(num_node_features, 128, cached=False)
        self.bn01 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn02 = BatchNorm1d(64)
        self.conv3 = GCNConv(64, 32, cached=False)
        self.bn03 = BatchNorm1d(32)
        # GIN-representation
        fc_gin1 = Sequential(Linear(num_node_features, 128), ReLU(), Linear(128, 128))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(128)
        fc_gin2 = Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(64)
        fc_gin3 = Sequential(Linear(64, 32), ReLU(), Linear(32, 32))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(32)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32 + 32, 128)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.fc2 = Linear(128, 64)
        self.dropout2 = Dropout(p=0.2)
        self.fc3 = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_add_pool(x, batch)
        # GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_add_pool(z, batch)
        # Concatenating representations
        cr = torch.cat((x, z), 1)
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        return cr.view(-1, self.num_classes)


class GAT_GINE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT_GINE, self).__init__()
        self.num_classes = num_classes
        # GAT-representation
        self.gat1 = GATConv(num_node_features, 128, heads=8)
        self.bn11 = BatchNorm1d(128 * 8)
        self.gat2 = GATConv(128 * 8, 64, heads=8)
        self.bn12 = BatchNorm1d(64 * 8)
        self.gat3 = GATConv(64 * 8, 32, heads=8)
        self.bn13 = BatchNorm1d(32 * 8)
        # GIN-representation
        fc_gin1 = Sequential(Linear(num_node_features, 128), ReLU(), Linear(128, 128))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(128)
        fc_gin2 = Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(64)
        fc_gin3 = Sequential(Linear(64, 32), ReLU(), Linear(32, 32))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(32)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32 * 8 + 32, 128)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.fc2 = Linear(128, 64)
        self.dropout2 = Dropout(p=0.2)
        self.fc3 = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_add_pool(y, batch)
        # GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_add_pool(z, batch)
        # Concatenating representations
        cr = torch.cat((y, z), 1)
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        return cr.view(-1, self.num_classes)

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.num_classes = num_classes
        # GCN-representation
        self.conv1 = GCNConv(num_node_features, 128, cached=False)
        self.bn01 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn02 = BatchNorm1d(64)
        self.conv3 = GCNConv(64, 32, cached=False)
        self.bn03 = BatchNorm1d(32)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32, 128)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.fc2 = Linear(128, 64)
        self.dropout2 = Dropout(p=0.2)
        self.fc3 = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GCN-representation
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn01(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn02(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn03(x)
        x = global_add_pool(x, batch)
        # Concatenating representations
        cr = x
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        return cr.view(-1, self.num_classes)

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.num_classes = num_classes
        # GAT-representation
        self.gat1 = GATConv(num_node_features, 128, heads=8)
        self.bn11 = BatchNorm1d(128 * 8)
        self.gat2 = GATConv(128 * 8, 64, heads=8)
        self.bn12 = BatchNorm1d(64 * 8)
        self.gat3 = GATConv(64 * 8, 32, heads=8)
        self.bn13 = BatchNorm1d(32 * 8)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32 * 8, 128)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.fc2 = Linear(128, 64)
        self.dropout2 = Dropout(p=0.2)
        self.fc3 = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GAT-representation
        y = F.relu(self.gat1(y, edge_index))
        y = self.bn11(y)
        y = F.relu(self.gat2(y, edge_index))
        y = self.bn12(y)
        y = F.relu(self.gat3(y, edge_index))
        y = self.bn13(y)
        y = global_add_pool(y, batch)
        # Concatenating representations
        cr = y
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        return cr.view(-1, self.num_classes)

class GINE(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GINE, self).__init__()
        self.num_classes = num_classes
        # GIN-representation
        fc_gin1 = Sequential(Linear(num_node_features, 128), ReLU(), Linear(128, 128))
        self.gin1 = GINConv(fc_gin1)
        self.bn21 = BatchNorm1d(128)
        fc_gin2 = Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.gin2 = GINConv(fc_gin2)
        self.bn22 = BatchNorm1d(64)
        fc_gin3 = Sequential(Linear(64, 32), ReLU(), Linear(32, 32))
        self.gin3 = GINConv(fc_gin3)
        self.bn23 = BatchNorm1d(32)
        # Fully connected layers for concatenating outputs
        self.fc1 = Linear(32, 128)  # Adjusted input size to match concatenated size
        self.dropout1 = Dropout(p=0.2)
        self.fc2 = Linear(128, 64)
        self.dropout2 = Dropout(p=0.2)
        self.fc3 = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_w, batch = data.x, data.edge_index, data.edge_attr, data.batch
        y = x
        z = x
        # GIN-representation
        z = F.relu(self.gin1(z, edge_index))
        z = self.bn21(z)
        z = F.relu(self.gin2(z, edge_index))
        z = self.bn22(z)
        z = F.relu(self.gin3(z, edge_index))
        z = self.bn23(z)
        z = global_add_pool(z, batch)
        # Concatenating representations
        cr = z
        cr = F.relu(self.fc1(cr))
        cr = self.dropout1(cr)
        cr = F.relu(self.fc2(cr))
        cr = self.dropout2(cr)
        cr = self.fc3(cr)
        return cr.view(-1, self.num_classes)
    
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)  # Output same size as input for residual
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        
        # Pre-activation design
        out = self.relu(x)
        out = self.dropout(out)
        out = self.fc1(out)
        
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out + residual