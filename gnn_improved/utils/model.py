import torch
from torch import nn

class GNN(nn.Module):
    def __init__(self, other_features_dim, num_passes=3, num_embed=100, embed_dim=64):
        super().__init__()

        self.num_passes = num_passes
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(num_embed, embed_dim)
        self.message_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                         nn.ReLU(),
                                         nn.Linear(embed_dim, embed_dim))
        self.update_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(embed_dim, embed_dim))
        self.readout_mlp = nn.Sequential(nn.Linear(embed_dim + other_features_dim, embed_dim + other_features_dim),
                                         nn.ReLU(),
                                         nn.Linear(embed_dim + other_features_dim, 1),
                                         )
        
    def forward(self, encoded_atoms, edges, natoms, other_features):
        h = self.embed(encoded_atoms)
        
        for _ in range(self.num_passes):
            pairwise_products = h[edges[0]] * h[edges[1]]
            pairwise_messages = self.message_mlp(pairwise_products)
            
            index = edges[0].unsqueeze(-1).expand_as(pairwise_messages)
            node_messages = torch.zeros((sum(natoms), self.embed_dim), dtype=pairwise_messages.dtype, device=pairwise_messages.device)
            node_messages = node_messages.scatter_add_(dim=0, index=index, src=pairwise_messages)
            
            h = h + self.update_mlp(node_messages)
            
        h = torch.cat((h, other_features), dim=1)
        nodes_out = self.readout_mlp(h)
        out = torch.stack([mol.sum() for mol in nodes_out.split(natoms)])
        return out