'''
this file contains the FineTuneGNN class which is used to fine-tune a GNN model
it goes into the MolCLR or GraphLoG repo and is implemented based on their GNN architecture
and depends on the model and dataset contained in the MolCLR or GraphLoG repo
it is designed to be used with the molclr_finetune.py or graphlog_finetune.py scripts
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# Atom and edge constants
num_atom_type = 119
num_chirality_tag = 3
num_bond_type = 5
num_bond_direction = 3

class FineTuneGNN(nn.Module):
    def __init__(self, 
                 conv_layer,  # GINEConv (from molclr) or GINConv (from graphlog)
                 num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0,
                 pool='mean', projection_hidden_dim=512, projection_output_dim=2): 
        super(FineTuneGNN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        
        # Atom embeddings
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # GNN layers
        self.gnns = nn.ModuleList([conv_layer(emb_dim) for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

        # Pooling
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pool type: {pool}")

        # Feature linear layer
        self.feat_lin = nn.Linear(emb_dim, feat_dim)

        # Projection head (standard MLP head)
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for i in range(self.num_layer):
            h = self.gnns[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            if i == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        pooled = self.pool(h, data.batch)
        feat = self.feat_lin(pooled)

        output = self.projection_head(feat)
        return feat, output

    def load_my_state_dict(self, state_dict, strict=False):
        """
        Load pretrained weights from MolCLR or GraphLoG checkpoint.
        Use strict=False to ignore missing keys like projection head.
        """
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        skipped = [k for k in state_dict if k not in pretrained_dict]
        if skipped:
            print(f"Skipping keys not loaded: {skipped}")
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
    