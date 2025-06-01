import logging
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytoda
import torch
import torch.nn as nn
from pytoda.smiles.transforms import AugmentTensor

from utils.hyperparams import LOSS_FN_FACTORY, ACTIVATION_FN_FACTORY
from utils.interpret import monte_carlo_dropout, test_time_augmentation
from utils.layers import convolutional_layer, ContextAttentionLayer, dense_layer
from utils.utils import get_device, get_log_molar
from utils.drug_embedding import DGLRepresentationExtractor

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PASO_GEP_CNV_MUT(nn.Module):
    """Based on the MCA model in Molecular Pharmaceutics:
        https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520.
    """

    def __init__(self, params, *args, **kwargs):

        super(PASO_GEP_CNV_MUT, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]
        self.min_max_scaling = True if params.get('drug_sensitivity_processing_parameters', {}) != {} else False
        if self.min_max_scaling:
            self.IC50_max = params['drug_sensitivity_processing_parameters']['parameters']['max']
            self.IC50_min = params['drug_sensitivity_processing_parameters']['parameters']['min']

        # Model inputs
        self.smiles_padding_length = 256
        self.smiles_embedding_size = 256  # fixed to match drug representation
        self.number_of_pathways = params.get('number_of_pathways', 619)
        self.smiles_attention_size = params.get('smiles_attention_size', 64)
        self.gene_attention_size = params.get('gene_attention_size', 1)
        self.molecule_temperature = params.get('molecule_temperature', 1.)
        self.gene_temperature = params.get('gene_temperature', 1.)

        # Model architecture (hyperparameter)
        # Attention head configs
        self.molecule_gep_heads = params.get('molecule_gep_heads', [2, 2, 2])
        self.molecule_cnv_heads = params.get('molecule_cnv_heads', [2, 2, 2])
        self.molecule_mut_heads = params.get('molecule_mut_heads', [2, 2, 2])
        self.gene_heads = params.get('gene_heads', [1, 1, 1])
        self.cnv_heads = params.get('cnv_heads', [1, 1, 1])
        self.mut_heads = params.get('mut_heads', [1, 1, 1])
        self.n_heads = params.get('n_heads', 1)
        self.num_layers = params.get('num_layers', 2)
        self.omics_dense_size = params.get('omics_dense_size', 128)
        self.hidden_sizes = (
            [
                # Only use DrugEmbeddingModel output
                self.molecule_gep_heads[0] * params['smiles_embedding_size'] + # 6x256
                self.molecule_cnv_heads[0] * params['smiles_embedding_size'] +
                self.molecule_mut_heads[0] * params['smiles_embedding_size'] +
                sum(self.gene_heads) * self.omics_dense_size +
                sum(self.cnv_heads) * self.omics_dense_size +
                sum(self.mut_heads) * self.omics_dense_size
            ] + params.get('stacked_dense_hidden_sizes', [1024, 512])
        )

        self.dropout = params.get('dropout', 0.5)
        self.temperature = params.get('temperature', 1.)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get('activation_fn', 'relu')]

        # Build the model
        # Drug Embedding Model
        self.representation_extractor = DGLRepresentationExtractor()

        smiles_hidden_sizes = ([params['smiles_embedding_size']] +
                               [params['smiles_embedding_size']] + 
                               [params['smiles_embedding_size']])
        
        # Drug attention stream
        self.molecule_attention_layers_gep = nn.Sequential(OrderedDict([
            (
                f'molecule_gep_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[layer],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_pathways,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.molecule_temperature
                )
            ) for layer in range(len(self.molecule_gep_heads))
            for head in range(self.molecule_gep_heads[layer])
        # (bs x ref_seq_length x ref_hidden_size) ContextAttention layer의 출력 형태. 이거 플러스 어텐션 값 bs x ref_seq_length
        ]))  # yapf: disable

        self.molecule_attention_layers_cnv = nn.Sequential(OrderedDict([
            (
                f'molecule_cnv_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[layer],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_pathways,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.molecule_temperature
                )
            ) for layer in range(len(self.molecule_cnv_heads))
            for head in range(self.molecule_cnv_heads[layer])
        ]))

        self.molecule_attention_layers_mut = nn.Sequential(OrderedDict([
            (
                f'molecule_mut_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[layer],
                    reference_sequence_length=self.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=self.number_of_pathways,
                    attention_size=self.smiles_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.molecule_temperature
                )
            ) for layer in range(len(self.molecule_mut_heads))
            for head in range(self.molecule_mut_heads[layer])
        ]))

        # Gene attention stream
        self.gene_attention_layers = nn.Sequential(OrderedDict([
            (
                f'gene_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_pathways,
                    context_hidden_size=smiles_hidden_sizes[layer],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.gene_temperature
                )
            ) for layer in range(len(self.molecule_gep_heads))
            for head in range(self.gene_heads[layer])
        ]))  # yapf: disable

        # CNV attention stream
        self.cnv_attention_layers = nn.Sequential(OrderedDict([
            (
                f'cnv_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_pathways,
                    context_hidden_size=smiles_hidden_sizes[layer],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.gene_temperature
                )
            ) for layer in range(len(self.molecule_cnv_heads))
            for head in range(self.cnv_heads[layer])
        ]))

        # MUT attention stream
        self.mut_attention_layers = nn.Sequential(OrderedDict([
            (
                f'mut_attention_{layer}_head_{head}',
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=self.number_of_pathways,
                    context_hidden_size=smiles_hidden_sizes[layer],
                    context_sequence_length=self.smiles_padding_length,
                    attention_size=self.gene_attention_size,
                    individual_nonlinearity=params.get(
                        'context_nonlinearity', nn.Sequential()
                    ),
                    temperature=self.gene_temperature
                )
            ) for layer in range(len(self.molecule_mut_heads))
            for head in range(self.mut_heads[layer])
        ]))

        # Omics dense stream
        # GEP
        self.gep_dense_layers = nn.Sequential(OrderedDict([
            (
                f'gep_dense_{layer}_head_{head}',
                dense_layer(
                    self.number_of_pathways, # 619
                    self.omics_dense_size, #hidden
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=params.get('batch_norm', True)
                ).to(self.device)
            ) for layer in range(len(self.molecule_gep_heads))
            for head in range(self.gene_heads[layer])
            ]))  # yapf: disable
        # CNV
        self.cnv_dense_layers = nn.Sequential(OrderedDict([
            (
                f'cnv_dense_{layer}_head_{head}',
                dense_layer(
                    self.number_of_pathways,
                    self.omics_dense_size,
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=params.get('batch_norm', True)
                ).to(self.device)
            ) for layer in range(len(self.molecule_cnv_heads))
            for head in range(self.cnv_heads[layer])
            ]))  # yapf: disable
        # MUT
        self.mut_dense_layers = nn.Sequential(OrderedDict([
            (
                f'mut_dense_{layer}_head_{head}',
                dense_layer(
                    self.number_of_pathways,
                    self.omics_dense_size,
                    act_fn=self.act_fn,
                    dropout=self.dropout,
                    batch_norm=params.get('batch_norm', True)
                ).to(self.device)
            ) for layer in range(len(self.molecule_mut_heads))
            for head in range(self.mut_heads[layer])
            ]))  # yapf: disable

        # Only applied if params['batch_norm'] = True
        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get('batch_norm', True)
                        ).to(self.device)
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = (
            nn.Linear(self.hidden_sizes[-1], 1)
            if not params.get('final_activation', False) else nn.Sequential(
                OrderedDict(
                    [
                        ('projection', nn.Linear(self.hidden_sizes[-1], 1)),
                        ('sigmoidal', ACTIVATION_FN_FACTORY['sigmoid'])
                    ]
                )
            )
        )

    def forward(self, smiles, gep, cnv, mut):
        """
        Args:
            smiles (list of str): SMILES strings of shape [bs]
            gep (torch.Tensor): Gene expression profile of shape [bs, number_of_pathways]
            cnv (torch.Tensor): Copy number variation data of shape [bs, number_of_pathways]
            mut (torch.Tensor): Mutation data of shape [bs, number_of_pathways]

        Returns:
            predictions (torch.Tensor): IC50 predictions of shape [bs, 1]
            prediction_dict (dict): dictionary containing prediction tensor under key 'IC50'
        """
        gep = torch.unsqueeze(gep, dim=-1) # [bs, number_of_genes , 1]
        cnv = torch.unsqueeze(cnv, dim=-1) # [bs, number_of_genes , 1]
        mut = torch.unsqueeze(mut, dim=-1) # [bs, number_of_genes , 1]

        drug_repr = self.representation_extractor.featurize_and_encode(smiles)
        encoded_smiles = [drug_repr[i] for i in range(3)]

        # Molecule context attention
        (encodings, smiles_alphas_gep, smiles_alphas_cnv,
         smiles_alphas_mut, gene_alphas, cnv_alphas, mut_alphas) = [], [], [], [], [], [], []
        for layer in range(len(self.molecule_gep_heads)):
            for head in range(self.molecule_gep_heads[layer]):
                ind = self.molecule_gep_heads[0] * layer + head
                e, a = self.molecule_attention_layers_gep[ind](
                    encoded_smiles[layer], gep
                )
                encodings.append(e)
                smiles_alphas_gep.append(a)
        # encodings 는  (bs x ref_seq_length x ref_hidden_size) 모양 10개 

        for layer in range(len(self.molecule_cnv_heads)): # 5번 반복
            for head in range(self.molecule_cnv_heads[layer]): # 2번 반복 
                ind = self.molecule_cnv_heads[0] * layer + head
                e, a = self.molecule_attention_layers_cnv[ind](
                    encoded_smiles[layer], cnv
                )
                encodings.append(e)
                smiles_alphas_cnv.append(a)

        for layer in range(len(self.molecule_mut_heads)):
            for head in range(self.molecule_mut_heads[layer]):
                ind = self.molecule_mut_heads[0] * layer + head
                e, a = self.molecule_attention_layers_mut[ind](
                    encoded_smiles[layer], mut
                )
                encodings.append(e)
                smiles_alphas_mut.append(a)

        # Gene context attention
        for layer in range(len(self.gene_heads)): # 5번 반복
            for head in range(self.gene_heads[layer]): # 1번 반복
                ind = self.gene_heads[0] * layer + head

                e, a = self.gene_attention_layers[ind](
                    gep, encoded_smiles[layer], average_seq=False
                )

                e = self.gep_dense_layers[ind](e)
                encodings.append(e)
                gene_alphas.append(a)

        for layer in range(len(self.cnv_heads)):
            for head in range(self.cnv_heads[layer]):
                ind = self.cnv_heads[0] * layer + head

                e, a = self.cnv_attention_layers[ind](
                    cnv, encoded_smiles[layer], average_seq=False
                )

                e = self.cnv_dense_layers[ind](e)
                encodings.append(e)
                cnv_alphas.append(a)

        for layer in range(len(self.mut_heads)):
            for head in range(self.mut_heads[layer]):
                ind = self.mut_heads[0] * layer + head

                e, a = self.mut_attention_layers[ind](
                    mut, encoded_smiles[layer], average_seq=False
                )

                e = self.mut_dense_layers[ind](e)
                encodings.append(e)
                mut_alphas.append(a)
        
        # encodings는 (45, (bs x 256 x smile hidden))
        encodings = torch.cat(encodings, dim=1)

        # Apply batch normalization if specified
        inputs = self.batch_norm(encodings) if self.params.get(
            'batch_norm', False
        ) else encodings
        # NOTE: stacking dense layers as a bottleneck
        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
        prediction_dict = {}

        if not self.training:
            # The below is to ease postprocessing
            smiles_attention_gep = torch.cat([torch.unsqueeze(p, -1) for p in smiles_alphas_gep], dim=-1)
            smiles_attention_cnv = torch.cat([torch.unsqueeze(p, -1) for p in smiles_alphas_cnv], dim=-1)
            smiles_attention_mut = torch.cat([torch.unsqueeze(p, -1) for p in smiles_alphas_mut], dim=-1)
            gene_attention = torch.cat([torch.unsqueeze(p, -1) for p in gene_alphas], dim=-1)
            cnv_attention = torch.cat([torch.unsqueeze(p, -1) for p in cnv_alphas], dim=-1)
            mut_attention = torch.cat([torch.unsqueeze(p, -1) for p in mut_alphas], dim=-1)
            prediction_dict.update({
                'gene_attention': gene_attention,
                'cnv_attention': cnv_attention,
                'mut_attention': mut_attention,
                'smiles_attention_gep': smiles_attention_gep,
                'smiles_attention_cnv': smiles_attention_cnv,
                'smiles_attention_mut': smiles_attention_mut,
                'IC50': predictions,
                'log_micromolar_IC50':
                    get_log_molar(
                        predictions,
                        ic50_max=self.IC50_max,
                        ic50_min=self.IC50_min
                    ) if self.min_max_scaling else predictions
            })  # yapf: disable
        return predictions, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def _associate_language(self, smiles_language):
        """
        Bind a SMILES language object to the model. Is only used inside the
        confidence estimation.

        Arguments:
            smiles_language {[pytoda.smiles.smiles_language.SMILESLanguage]}
            -- [A SMILES language object]

        Raises:
            TypeError:
        """
        if not isinstance(
            smiles_language, pytoda.smiles.smiles_language.SMILESLanguage
        ):
            raise TypeError(
                'Please insert a smiles language (object of type '
                'pytoda.smiles.smiles_language.SMILESLanguage). Given was '
                f'{type(smiles_language)}'
            )
        self.smiles_language = smiles_language

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)

# Example Usage
if __name__ == "__main__":
    from data.TripleOmics_Drug_Dataset import TripleOmics_Drug_dataset, custom_collate_fn
    from torch.utils.data import DataLoader

    # File paths
    drug_sensitivity_filepath = '/home/bgd/MOUM/yaicon/data/10_fold_data/mixed/MixedSet_train_Fold0.csv'
    smiles_filepath = '/home/bgd/MOUM/yaicon/data/CCLE-GDSC-SMILES.csv'
    gep_filepath = '/home/bgd/MOUM/yaicon/data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = '/home/bgd/MOUM/yaicon/data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = '/home/bgd/MOUM/yaicon/data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'

    # Dataset
    dataset = TripleOmics_Drug_dataset(
        drug_sensitivity_filepath=drug_sensitivity_filepath,
        smiles_filepath=smiles_filepath,
        gep_filepath=gep_filepath,
        cnv_filepath=cnv_filepath,
        mut_filepath=mut_filepath,
        gep_standardize=True,
        cnv_standardize=True,
        mut_standardize=True,
        drug_sensitivity_min_max=True,
        column_names=('drug', 'cell_line', 'IC50')
    )

    # DataLoader
    batch_size = 4
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Model parameters
    params = {
        'smiles_padding_length': 256,
        'smiles_embedding_size': 256,
        'number_of_pathways': 619,
        'smiles_attention_size': 64,
        'gene_attention_size': 1,
        'molecule_temperature': 1.0,
        'gene_temperature': 1.0,
        'molecule_gep_heads': [2, 2, 2],
        'molecule_cnv_heads': [2, 2, 2],
        'molecule_mut_heads': [2, 2, 2],
        'gene_heads': [1, 1, 1],
        'cnv_heads': [1, 1, 1],
        'mut_heads': [1, 1, 1],
        'n_heads': 2,
        'num_layers': 4,
        'omics_dense_size': 256,
        'stacked_dense_hidden_sizes': [1024, 512],
        'dropout': 0.5,
        'temperature': 1.0,
        'activation_fn': 'relu',
        'batch_norm': True,
        'drug_sensitivity_processing_parameters': {
            'parameters': {'max': 100, 'min': 0}
        },
        'loss_fn': 'mse'
    }

    # Instantiate model
    model = PASO_GEP_CNV_MUT(params).to(get_device())
    model.eval()

    # Test with one batch
    for batch in trainloader:
        drug_data, gep_data, cnv_data, mut_data, ic50 = batch
        with torch.no_grad():
            predictions, prediction_dict = model(drug_data, gep_data, cnv_data, mut_data)
        print("=== Output Check ===")
        print(f"Drug data shapes: x={drug_data[0].shape}, adj_matrix={drug_data[1].shape}")
        print(f"GEP shape: {gep_data.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Prediction dict keys: {prediction_dict.keys()}")
        break