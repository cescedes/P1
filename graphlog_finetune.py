'''
this file goes into the GraphLoG repo/directory with the 'finetune_model.py' file 
it is used to finetune and run the finetune_model on the bace and bbbp datasets:
it experiments with freezing different layers of the GNN and evaluates the performance
and saves the results and embeddings for further analysis
'''
'''
uses the same environment as the original GraphLoG repo
to run this file, you need to have the GraphLoG repo cloned and the finetune_model.py file in the same directory

for running random initialization:
python graphlog_finetune.py --dataset bace 
python graphlog_finetune.py --dataset bbbp

for running with pretrained model and experimenting freezing different layers:
python graphlog_finetune.py --dataset bace --input_model_file models\graphlog.pth
python graphlog_finetune.py --dataset bbbp --input_model_file models\graphlog.pth

for other arguments, please refer to the argparse section in the code below
'''

import argparse
from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
import os
import numpy as np
import random

from splitters import scaffold_split
import pandas as pd
import itertools

from model import GINConv
from finetune_model import FineTuneGNN

criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(args, model, device, loader, optimizer, scheduler):
    model.train()
    scheduler.step()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        feat, pred = model(batch)
        y = batch.y.view_as(pred).float()

        is_valid = y ** 2 > 0
        loss_mat = criterion(pred, (y + 1) / 2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            feat, pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list)


def extract_embeddings(model, loader, device, save_dir):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feat, _ = model(batch)
            embeddings.append(feat.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "embeddings.npy"), np.concatenate(embeddings))
    np.save(os.path.join(save_dir, "labels.npy"), np.concatenate(labels))
    print(f"Saved embeddings and labels to {save_dir}")


def freeze_gnn_layers(model, freeze_indices):
    for i, conv in enumerate(model.gnns):
        if i in freeze_indices:
            for param in conv.parameters():
                param.requires_grad = False
            for param in model.batch_norms[i].parameters():
                param.requires_grad = False
            print(f"Froze GNN layer {i} and its BatchNorm")
        else:
            for param in conv.parameters():
                param.requires_grad = True
            for param in model.batch_norms[i].parameters():
                param.requires_grad = True


def generate_freeze_combinations(num_layers=5):
    all_combos = []
    for r in range(num_layers + 1):
        all_combos.extend(itertools.combinations(range(num_layers), r))
    return all_combos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--graph_pooling", type=str, default="mean")
    parser.add_argument("--JK", type=str, default="last")
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--dataset", type=str, default="", help="bace or bbbp only")
    parser.add_argument("--input_model_file", type=str, default="", help="pretrained model path or empty for random init")
    parser.add_argument("--split", type=str, default="scaffold")
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=5, help="early stopping patience in epochs")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    assert args.dataset in ["bace", "bbbp"], "Only BACE and BBBP datasets are supported."

    # Load dataset
    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)
    smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0,
                                                                frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Prepare freeze combinations
    freeze_combinations = generate_freeze_combinations(num_layers=args.num_layer)
    results_list = []

    if args.input_model_file == "":
        print("Running randomly initialized model with no frozen layers.")
        
        model = FineTuneGNN(conv_layer=GINConv, num_layer=5, emb_dim=300, projection_output_dim=1)

        model.to(device)

        model_param_group = [
            {"params": model.x_embedding1.parameters()},
            {"params": model.x_embedding2.parameters()},
            {"params": model.gnns.parameters()},
            {"params": model.batch_norms.parameters()},
            {"params": model.feat_lin.parameters()},
            {"params": model.projection_head.parameters()}
        ]

        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.3)

        best_valid = 0
        best_test_acc = 0
        epochs_since_improvement = 0

        for epoch in range(1, args.epochs + 1):
            print(f"==== Epoch {epoch} | Freeze []")
            train(args, model, device, train_loader, optimizer, scheduler)
            val_acc = eval(args, model, device, val_loader)
            test_acc = eval(args, model, device, test_loader)
            print(f"Validation ROC AUC: {val_acc:.4f}, Test ROC AUC: {test_acc:.4f}")

            if val_acc > best_valid:
                best_valid = val_acc
                best_test_acc = test_acc
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Early stopping
            if epochs_since_improvement >= args.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print(f"Best Test ROC AUC: {best_test_acc:.4f}")
        results_list.append([[], best_test_acc])

        save_dir = f"experiments/{args.dataset}/random_init"
        extract_embeddings(model, val_loader, device, save_dir)

    else:
        print(f"Running pretrained model: {args.input_model_file}")
        state_dict = torch.load(args.input_model_file, map_location='cpu')

        for freeze_layers in freeze_combinations:
            print(f"Running with frozen layers: {freeze_layers}")

            model = FineTuneGNN(conv_layer=GINConv, num_layer=5, emb_dim=300, projection_output_dim=1)
            model.load_my_state_dict(state_dict)
            model.to(device)

            freeze_gnn_layers(model, freeze_layers)

            model_param_group = [
                {"params": model.x_embedding1.parameters()},
                {"params": model.x_embedding2.parameters()},
                {"params": model.gnns.parameters()},
                {"params": model.batch_norms.parameters()},
                {"params": model.feat_lin.parameters()},
                {"params": model.projection_head.parameters()}
            ]

            optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
            scheduler = StepLR(optimizer, step_size=30, gamma=0.3)

            best_valid = 0
            best_test_acc = 0
            epochs_since_improvement = 0

            for epoch in range(1, args.epochs + 1):
                print(f"==== Epoch {epoch} | Freeze {freeze_layers}")
                train(args, model, device, train_loader, optimizer, scheduler)
                val_acc = eval(args, model, device, val_loader)
                test_acc = eval(args, model, device, test_loader)
                print(f"Validation ROC AUC: {val_acc:.4f}, Test ROC AUC: {test_acc:.4f}")

                if val_acc > best_valid:
                    best_valid = val_acc
                    best_test_acc = test_acc
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= args.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch} for frozen layers {freeze_layers}")
                    break

            print(f"Best Test ROC AUC with frozen layers {freeze_layers}: {best_test_acc:.4f}")
            results_list.append([freeze_layers, best_test_acc])

            save_dir = f"experiments/{args.dataset}/freeze_{'-'.join(map(str, freeze_layers))}"
            extract_embeddings(model, val_loader, device, save_dir)

    # Save summary
    df = pd.DataFrame(results_list, columns=["frozen_layers", "roc_auc"])
    os.makedirs(f"experiments/{args.dataset}", exist_ok=True)
    df.to_csv(f"experiments/{args.dataset}/finetune_with_freezing.csv", index=False)
    print("Saved final results.")


if __name__ == "__main__":
    main()
