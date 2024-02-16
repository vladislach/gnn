import torch
from torch.utils.data import Dataset
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, encoded_atoms_list, edges_list, natoms_list, other_features, dGsolv_list):
        '''
        encoded_atoms_list: list of torch.LongTensor of atomic numbers
        edges_list: list of torch.LongTensor of edges
        natoms_list: list of int of number of atoms
        dGsolv_list: list of torch.Tensor of solvation energy
        '''
        self.encoded_atoms_list = encoded_atoms_list
        self.edges_list = edges_list
        self.natoms_list = natoms_list
        self.other_features = other_features
        self.dGsolv_list = dGsolv_list

    def __len__(self):
        return len(self.natoms_list)
    
    def __getitem__(self, idx):
        encoded_atoms = torch.LongTensor(self.encoded_atoms_list[idx])
        edges = torch.LongTensor(self.edges_list[idx])
        natoms = self.natoms_list[idx]
        other_features = self.other_features[idx]
        dGsolv = torch.Tensor(self.dGsolv_list[idx])

        return encoded_atoms, edges, natoms, other_features, dGsolv

def collate_graphs(batch):
    '''
    Batch multiple graphs into one batched graph.
    Args:
        batch: a tuple of tuples of (encoded_atoms, edges, natoms, dGsolv) obtained from GraphDataset.__getitem__()
    Return:
        (tuple): batched encoded_atoms, edges, natoms, dGsolv
    Example:
    >>> batch = [(torch.LongTensor([6, 6, 7]), torch.LongTensor([[0, 2, 2, 1], [2, 0, 1, 2]]), 3, torch.Tensor([0.1])),
                 (torch.LongTensor([6, 6, 8]), torch.LongTensor([[0, 2, 2, 1], [2, 0, 1, 2]]), 3, torch.Tensor([0.2]))]
    >>> collate_graphs(batch)
    (torch.Tensor([6, 6, 7, 6, 6, 8]), torch.Tensor([[0, 2, 2, 1, 3, 5, 5, 4], [2, 0, 1, 2, 5, 3, 4, 5]]), [3, 3], torch.Tensor([0.1, 0.2]))
    '''
    encoded_atoms_batch = []
    edges_batch = []
    natoms_batch = []
    other_features_batch = []
    dGsolv_batch = []

    index_shifts = np.cumsum([0] + [b[2] for b in batch])[:-1]

    for i in range(len(batch)):
        encoded_atoms, edges, natoms, other_features, dGsolv = batch[i]
        edges = edges + index_shifts[i]
        encoded_atoms_batch.append(encoded_atoms)
        edges_batch.append(edges)
        natoms_batch.append(natoms)
        other_features_batch.append(other_features)
        dGsolv_batch.append(dGsolv)

    encoded_atoms_batch = torch.cat(encoded_atoms_batch)
    edges_batch = torch.cat(edges_batch, dim=1)
    natoms_batch = natoms_batch
    other_features_batch = torch.cat(other_features_batch)
    dGsolv_batch = torch.cat(dGsolv_batch)

    return encoded_atoms_batch, edges_batch, natoms_batch, other_features_batch, dGsolv_batch
