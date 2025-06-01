import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from tqdm import tqdm  # Import tqdm for progress bars

# DeepChem 로거 가져오기
dc_logger = logging.getLogger("deepchem")
dc_logger.setLevel(logging.ERROR)  # ERROR 이상만 출력

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import json
import pickle
from time import time
import torch

from data.TripleOmics_Drug_Dataset import TripleOmics_Drug_dataset, custom_collate_fn
from models.model import PASO_GEP_CNV_MUT
from utils.hyperparams import OPTIMIZER_FACTORY
from utils.loss_functions import pearsonr, r2_score
from utils.utils import get_device, get_log_molar

def main(
    drug_sensitivity_filepath,
    gep_filepath,
    cnv_filepath,
    mut_filepath,
    smiles_filepath,
    gene_filepath,
    model_path,
    params,
    training_name
):
    # Process parameter file:
    torch.backends.cudnn.benchmark = True
    params = params
    params.update(
        {
            "batch_size": 512,
            "epochs": 200,
            "num_workers": 4,
            "stacked_dense_hidden_sizes": [
                1024,
                512
            ],
        }
    )
    logger.info("Parameters: %s", params)

    # Prepare the dataset
    logger.info("Starting data preprocessing...")

    # Load the gene list
    with open(gene_filepath, "rb") as f:
        pathway_list = pickle.load(f)

    n_folds = params.get("fold", 11)
    logger.info("Starting %d-fold cross-validation", n_folds)

    for fold in range(n_folds):
        logger.info("============== Fold [%d/%d] ==============", fold+1, params['fold'])
        # Create model directory and dump files
        model_dir = os.path.join(model_path, training_name, 'Fold' + str(fold+1))
        os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
        with open(os.path.join(model_dir, "TCGA_classifier_best_aucpr_GEP.json"), "w") as fp:
            json.dump(params, fp, indent=4)

        # Load the drug sensitivity data
        drug_sensitivity_train = drug_sensitivity_filepath + 'train_Fold' + str(fold) + '.csv'
        train_dataset = TripleOmics_Drug_dataset(
            drug_sensitivity_filepath=drug_sensitivity_train,
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
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=4,
            collate_fn=custom_collate_fn
        )
        drug_sensitivity_test = drug_sensitivity_filepath + 'test_Fold' + str(fold) + '.csv'
        min_value = params["drug_sensitivity_processing_parameters"]["parameters"]["min"]
        max_value = params["drug_sensitivity_processing_parameters"]["parameters"]["max"]
        test_dataset = TripleOmics_Drug_dataset(
            drug_sensitivity_filepath=drug_sensitivity_test,
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
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,  # Fixed: Use test_dataset instead of train_dataset
            batch_size=params["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=4,
            collate_fn=custom_collate_fn
        )
        logger.info(
            "FOLD [%d/%d] Training dataset has %d samples, test set has %d.",
            fold+1, params['fold'], len(train_dataset), len(test_dataset)
        )
        device = get_device()
        print(device)
        save_top_model = os.path.join(model_dir, "weights/{}_{}_{}.pt")
        params.update(
            {
                "number_of_genes": len(pathway_list),
            }
        )
        model = PASO_GEP_CNV_MUT(params).to(get_device())
        model.train()

        min_loss, min_rmse, max_pearson, max_r2 = 100, 1000, 0, 0

        # Define optimizer
        optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")](
            model.parameters(), lr=params.get("lr", 0.001)
        )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params.update({"number_of_parameters": num_params})
        logger.info("Number of parameters: %d", num_params)

        # Overwrite params.json file with updated parameters.
        with open(os.path.join(model_dir, "TCGA_classifier_best_aucpr_GEP.json"), "w") as fp:
            json.dump(params, fp)

        # Start training
        logger.info("Training started for Fold %d...", fold+1)
        t = time()
        # Start training
        for epoch in range(params["epochs"]):
            logger.info("== Fold [%d/%d] Epoch [%d/%d] ==", fold+1, params['fold'], epoch+1, params['epochs'])
            training(model, device, epoch, fold, train_loader, optimizer, params, t)
            t = time()

            test_loss_a, test_pearson_a, test_rmse_a, test_r2_a, predictions, labels = (
                evaluation(model, device, test_loader, params, epoch, fold, max_value, min_value))

            def save(path, metric, typ, val=None):
                fold_info = "Fold_" + str(fold+1)
                model.save(path.format(fold_info + typ, metric, "bgd-test"))
                with open(os.path.join(model_dir, "results", fold_info + metric + ".json"), "w") as f:
                    json.dump(info, f)
                if typ == "best":
                    logger.info(
                        '\tNew best performance in "%s" with value: %.7f in epoch: %d',
                        metric, val, epoch
                    )

            def update_info():
                return {
                    "best_rmse": str(float(min_rmse)),
                    "best_pearson": str(float(max_pearson)),
                    "test_loss": str(min_loss),
                    "best_r2": str(float(max_r2)),
                    "predictions": [float(p) for p in predictions],
                }

            if test_loss_a < min_loss:
                min_rmse = test_rmse_a
                min_loss = test_loss_a
                min_loss_pearson = test_pearson_a
                min_loss_r2 = test_r2_a
                info = update_info()
                save(save_top_model, "mse", "best", min_loss)
                ep_loss = epoch
            if test_pearson_a > max_pearson:
                max_pearson = test_pearson_a
                max_pearson_loss = test_loss_a
                max_pearson_r2 = test_r2_a
                info = update_info()
                save(save_top_model, "pearson", "best", max_pearson)
                ep_pearson = epoch
            if test_r2_a > max_r2:
                max_r2 = test_r2_a
                max_r2_loss = test_loss_a
                max_r2_pearson = test_pearson_a
                info = update_info()
                save(save_top_model, "r2", "best", max_r2)
                ep_r2 = epoch
        logger.info(
            "Overall Fold %d best performances are: \n \t"
            "Loss = %.4f in epoch %d "
            "\t (Pearson was %.4f; R2 was %.4f) \n \t"
            "Pearson = %.4f in epoch %d "
            "\t (Loss was %.2f; R2 was %.4f) \n \t"
            "R2 = %.4f in epoch %d "
            "\t (Loss was %.4f; Pearson was %.4f) \n",
            fold+1, min_loss, ep_loss, min_loss_pearson, min_loss_r2,
            max_pearson, ep_pearson, max_pearson_loss, max_pearson_r2,
            max_r2, ep_r2, max_r2_loss, max_r2_pearson
        )
        save(save_top_model, "training", "done")

    logger.info("Training completed, models saved, shutting down.")

def training(model, device, epoch, fold, train_loader, optimizer, params, t):
    model.train()
    train_loss = 0
    # Add tqdm progress bar
    progress_bar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Training", leave=False)
    for ind, (drug_data, omic_1, omic_2, omic_3, y) in enumerate(progress_bar):
        y_hat, pred_dict = model(
            drug_data, omic_1.to(device), omic_2.to(device), omic_3.to(device))
        loss = model.loss(y_hat, y.to(device))
        optimizer.zero_grad()
        loss.backward()
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2e-6)
        optimizer.step()
        train_loss += loss.item()
        # Update progress bar with current loss
        progress_bar.set_postfix({'loss': f"{train_loss / (ind + 1):.5f}"})
        print(next(model.parameters()).device)
    progress_bar.close()
    logger.info(
        "**** TRAINING **** Fold[%d] Epoch [%d/%d], loss: %.5f. This took %.1f secs.",
        fold+1, epoch + 1, params['epochs'], train_loss / len(train_loader), time() - t
    )

def evaluation(model, device, test_loader, params, epoch, fold, max_value, min_value):
    model.eval()
    test_loss = 0
    log_pres = []
    log_labels = []
    # Add tqdm progress bar
    progress_bar = tqdm(test_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Testing", leave=False)
    with torch.no_grad():
        for ind, (drug_data, omic_1, omic_2, omic_3, y) in enumerate(progress_bar):
            y_hat, pred_dict = model(
                drug_data, omic_1.to(device), omic_2.to(device), omic_3.to(device)
            )
            log_pre = pred_dict.get("log_micromolar_IC50")
            log_pres.append(log_pre)
            log_y = get_log_molar(y, ic50_max=max_value, ic50_min=min_value)
            log_labels.append(log_y)
            loss = model.loss(log_pre, log_y.to(device))
            test_loss += loss.item()
            # Update progress bar with current loss
            progress_bar.set_postfix({'loss': f"{test_loss / (ind + 1):.5f}"})
    progress_bar.close()

    predictions = torch.cat([p.cpu() for preds in log_pres for p in preds])
    labels = torch.cat([l.cpu() for label in log_labels for l in label])
    test_pearson_a = pearsonr(torch.Tensor(predictions), torch.Tensor(labels))
    test_rmse_a = torch.sqrt(torch.mean((predictions - labels) ** 2))
    test_loss_a = test_loss / len(test_loader)
    test_r2_a = r2_score(torch.Tensor(predictions), torch.Tensor(labels))
    logger.info(
        "**** TEST **** Fold[%d] Epoch [%d/%d], loss: %.5f, Pearson: %.4f, RMSE: %.4f, R2: %.4f.",
        fold+1, epoch + 1, params['epochs'], test_loss_a, test_pearson_a, test_rmse_a, test_r2_a
    )
    return test_loss_a, test_pearson_a, test_rmse_a, test_r2_a, predictions, labels

if __name__ == "__main__":
    drug_sensitivity_filepath = 'data/10_fold_data/mixed/MixedSet_'
    smiles_filepath = 'data/CCLE-GDSC-SMILES.csv'
    gep_filepath = 'data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    cnv_filepath = 'data/CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    mut_filepath = 'data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv'
    gene_filepath = 'data/MUDICUS_Omic_619_pathways.pkl'

    model_path = 'result/model'
    params = {
        'fold': 10,
        'optimizer': "adam",
        'smiles_padding_length': 256,
        'smiles_embedding_size': 256,
        'number_of_pathways': 619,
        'smiles_attention_size': 256,
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
            'parameters': {"min": -8.658382, "max": 13.107465}
        },
        'loss_fn': 'mse'
    }
    training_name = 'maxlen_128'
    # Run the training
    main(
        drug_sensitivity_filepath,
        gep_filepath,
        cnv_filepath,
        mut_filepath,
        smiles_filepath,
        gene_filepath,
        model_path,
        params,
        training_name
    )