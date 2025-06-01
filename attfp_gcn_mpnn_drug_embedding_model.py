from dgllife.utils.featurizers import BaseAtomFeaturizer, BaseBondFeaturizer
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import (
    atom_type_one_hot, atom_degree_one_hot, atom_is_aromatic,
    atom_formal_charge, atom_total_num_H_one_hot, atom_hybridization_one_hot,
    atom_implicit_valence_one_hot, atom_num_radical_electrons,
    bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
    bond_stereo_one_hot
)
from dgllife.model.gnn import GCN
from dgllife.model.gnn.mpnn import MPNNGNN
from dgllife.model.gnn.attentivefp import AttentiveFPGNN

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import dgl
from rdkit import Chem
from tqdm import tqdm

class SafeManualAtomFeaturizer(BaseAtomFeaturizer):
    ATOM_TYPES = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B", "V",
        "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au",
        "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb"
    ]

    def __init__(self):
        super().__init__({
            "type": lambda atom: atom_type_one_hot(atom, allowable_set=self.ATOM_TYPES),
            "degree": lambda atom: atom_degree_one_hot(atom, allowable_set=list(range(6))),
            "implicit_valence": lambda atom: atom_implicit_valence_one_hot(atom, allowable_set=list(range(7))),
            "formal_charge": atom_formal_charge,
            "num_radical_electrons": atom_num_radical_electrons,
            "hybridization": lambda atom: atom_hybridization_one_hot(atom, ['SP', 'SP2', 'SP3']),
            "is_aromatic": atom_is_aromatic,
            "num_hs": lambda atom: atom_total_num_H_one_hot(atom, allowable_set=list(range(5)))
        })

    def __call__(self, mol):
        features = {key: [] for key in self.featurizer_funcs}
        for atom in mol.GetAtoms():
            for name, func in self.featurizer_funcs.items():
                val = func(atom)
                if isinstance(val, (int, float, bool)):
                    val = [float(val)]
                elif isinstance(val, list):
                    val = [float(v) for v in val]
                features[name].append(val)

        out = {}
        for name, values in features.items():
            arr = np.array(values, dtype=np.float32)
            out[name] = torch.tensor(arr)
        out["h"] = torch.cat(list(out.values()), dim=1)
        return out


class SafeManualBondFeaturizer(BaseBondFeaturizer):
    def __init__(self):
        super().__init__({
            "bond_type": lambda bond: bond_type_one_hot(bond, ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']),
            "conjugated": bond_is_conjugated,
            "in_ring": bond_is_in_ring,
            "stereo": lambda bond: bond_stereo_one_hot(bond, ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOANY'])
        })

    def __call__(self, mol):
        features = {key: [] for key in self.featurizer_funcs}
        for bond in mol.GetBonds():
            for name, func in self.featurizer_funcs.items():
                val = func(bond)
                for _ in range(2):  # bidirectional
                    if isinstance(val, (int, float, bool)):
                        features[name].append([float(val)])
                    elif isinstance(val, list):
                        features[name].append([float(v) for v in val])

        out = {}
        for name, values in features.items():
            arr = np.array(values, dtype=np.float32)
            out[name] = torch.tensor(arr)
        out["e"] = torch.cat(list(out.values()), dim=1)
        return out


def safe_construct_bigraph_from_mol(mol, add_self_loop=True):
    src, dst = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([u, v])
        dst.extend([v, u])
    if add_self_loop:
        for i in range(mol.GetNumAtoms()):
            src.append(i)
            dst.append(i)
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=mol.GetNumAtoms())
    return g


def safe_mol_to_bigraph(mol, atom_feats, bond_feats):
    g = safe_construct_bigraph_from_mol(mol, add_self_loop=True)
    g.ndata.update(atom_feats)

    num_edges = g.num_edges()
    original_edges = list(bond_feats.values())[0].shape[0]
    num_self_loops = num_edges - original_edges
    for key in bond_feats:
        feat = bond_feats[key]
        pad = torch.zeros((num_self_loops, feat.shape[1]), dtype=feat.dtype)
        bond_feats[key] = torch.cat([feat, pad], dim=0)

    g.edata.update(bond_feats)
    return g


class DGLRepresentationExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.atom_featurizer = SafeManualAtomFeaturizer()
        self.bond_featurizer = SafeManualBondFeaturizer()

        # dummy mol to get dimensions
        dummy_mol = Chem.MolFromSmiles("CC")
        atom_feats = self.atom_featurizer(dummy_mol)
        bond_feats = self.bond_featurizer(dummy_mol)
        self.node_dim = atom_feats["h"].shape[1]
        self.edge_dim = bond_feats["e"].shape[1]

        self.models = {
            "GCN": GCN(
                in_feats=self.node_dim,
                hidden_feats=[256, 256],
                gnn_norm=['none', 'none'],
                activation=[torch.nn.ReLU(), None],
                residual=[False, False],
                batchnorm=[False, False],
                dropout=[0.0, 0.0]
            ).to(self.device),
            "MPNN": MPNNGNN(
                node_in_feats=self.node_dim,
                edge_in_feats=self.edge_dim,
                node_out_feats=256,
                edge_hidden_feats=128,
                num_step_message_passing=3
            ).to(self.device),
            "attfp": AttentiveFPGNN(
                node_feat_size=self.node_dim,
                edge_feat_size=self.edge_dim,
                num_layers=2,
                graph_feat_size=256,
                dropout=0.1
            ).to(self.device)
        }

    def featurize_and_encode(self, smiles_list: list[str]) -> torch.Tensor:
        graphs = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                atom_feats = self.atom_featurizer(mol)
                bond_feats = self.bond_featurizer(mol)
                graph = safe_mol_to_bigraph(mol, atom_feats, bond_feats)
                graphs.append(graph)
            except Exception as e:
                print(f"[Error] Failed to process SMILES {smiles} → {e}")
                graphs.append(dgl.graph(([0], [0])))  # dummy graph

        batch_reps = []
        for graph in graphs:
            try:
                graph = graph.to(self.device)
                h = graph.ndata["h"]
                num_nodes = graph.num_nodes()
                padded_reps = []
                with torch.no_grad():
                    for model in self.models.values():
                        node_rep = model(graph, h, return_node_embeddings=True)
                        padded = torch.zeros((256, 256), dtype=torch.float32, device=self.device)
                        padded[:min(num_nodes, 256), :] = node_rep[:256, :]
                        padded_reps.append(padded)
                stacked = torch.stack(padded_reps, dim=0)  # shape: [3, 256, 256]
                batch_reps.append(stacked)
            except Exception as e:
                print(f"[Error] Failed to encode graph → {e}")
                batch_reps.append(torch.zeros((3, 256, 256), dtype=torch.float32, device=self.device))

        return torch.stack(batch_reps, dim=0)  # shape: [B, 3, 256, 256]