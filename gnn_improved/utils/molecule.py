import numpy as np

class Molecule:
    def __init__(self, smiles, atoms, bonds, charges=None, dGsolv=None):
        self.smiles= smiles
        self.atoms = atoms
        self.bonds = bonds
        self.charges = charges
        self.dGsolv = dGsolv
        self.natoms = len(atoms)

        self.adj_matrix = self.get_adjacency_matrix()
        self.adj_list = self.get_adj_list()
        self.edges = self.get_edges()
        self.atoms2enc = None
        self.enc2atoms = None

    def __repr__(self):
        return f"Molecule('{self.smiles}')"

    def get_adjacency_matrix(self):
        adj_matrix = np.zeros((self.natoms, self.natoms), dtype=int)
        for bond in self.bonds:
            adj_matrix[bond[0]][bond[1]] = 1
            adj_matrix[bond[1]][bond[0]] = 1
        return adj_matrix
    
    def get_adj_list(self):
        return {i: [j for j, val in enumerate(row) if val == 1] for i, row in enumerate(self.adj_matrix)}

    def get_edges(self):
        return np.stack(self.adj_matrix).nonzero()
    
    def set_encoder(self, atoms2enc):
        self.atoms2enc = atoms2enc

    def set_decoder(self, enc2atoms):
        self.enc2atoms = enc2atoms

    def encode_atoms(self):
        if self.atoms2enc is None:
            raise ValueError("atoms2enc is not set. Please set it using set_encoder method.")
        return [self.atoms2enc[atom] for atom in self.atoms]

    def decode_atoms(self):
        if self.enc2atoms is None:
            raise ValueError("enc2atoms is not set. Please set it using set_decoder method.")
        return [self.enc2atoms[enc] for enc in self.encode_atoms()]

    def find_mol_paths(self, length, decode=False):
        paths = []

        def dfs(node, current_path):
            if len(current_path) == length:
                if current_path[::-1] not in paths:
                    paths.append(current_path)
                return
            for neighbor in self.adj_list[node]:
                if neighbor not in current_path:
                    dfs(neighbor, current_path + [neighbor])

        for node in self.adj_list:
            dfs(node, [node])

        if decode:
            paths = [[self.atoms[ix] for ix in path] for path in paths]

        return paths