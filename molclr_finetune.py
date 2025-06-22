'''
this file goes into the MolCLR repo/directory with the 'finetune_model.py' file 
it is used to finetune and run the finetune_model on the bace and bbbp datasets:
it experiments with freezing different layers of the GNN and evaluates the performance
and saves the results and embeddings for further analysis

uses the same environment as the original MolCLR repo
to run this file, you need to have the MolCLR repo cloned and the finetune_model.py file in the same directory

run this file with the command:
python molclr_finetune.py 

to configure the finetuning process, you can modify the 'config_finetune.yaml' file
please use the version of the 'config_finetune.yaml' file from this repo
it has added features for "initialization" (random | pretrained) and "early stopping"
'''


import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test import MolTestDatasetWrapper

from finetune_model import FineTuneGNN
from models.ginet_finetune import GINEConv


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))

class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        # Determine base experiment dir 
        self.exp_base_dir = os.path.join('experiments', config['task_name'].lower())
        os.makedirs(self.exp_base_dir, exist_ok=True)


        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        log_dir = os.path.join('finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset

        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        _, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        if self.config['model_type'] == 'gin':
            
            model = FineTuneGNN(conv_layer=GINEConv, num_layer=5, emb_dim=300).to(self.device)
            
        freeze_indices = config.get('freeze_layers', [])  # passed in config
        if config['init_from'] == 'pretrained':
            model = self._load_pre_trained_weights(model)
            
            if freeze_indices:  # only apply if any layers are specified
                freeze_gnn_layers(model, freeze_indices)
                print(f"Frozen layers: {freeze_indices}")
            else:
                print("No layers frozen. Full fine-tuning.")

        elif config['init_from'] == 'random':
            print("Initializing model with random weights. No layers will be frozen.")
            # Ensure all layers are trainable
            for param in model.parameters():
                param.requires_grad = True



        layer_list = []
        for name, param in model.named_parameters():
            if 'projection_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)


        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        patience = self.config.get('early_stopping', {}).get('patience', 10)
        early_stop_enabled = self.config.get('early_stopping', {}).get('enabled', False)
        patience_counter = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        best_valid_cls = valid_cls
                        patience_counter = 0
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                        print(f"Validation ROC AUC improved to {best_valid_cls:.4f}, saving model.")
                    else:
                        patience_counter += 1
                        print(f"No improvement in ROC AUC for {patience_counter} eval steps.")

                    if early_stop_enabled and patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch_counter}.")
                        break

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        
        # Save embeddings and labels for ROGI-XD
        if config['init_from'] == 'random':
            save_dir = os.path.join(self.exp_base_dir, 'random_init')
        else:
            save_dir = os.path.join(self.exp_base_dir, f"freeze_{'-'.join(map(str, freeze_indices))}")

        self._save_embeddings_for_rogi(model, valid_loader, save_dir)


        self._test(model, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                _, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _save_embeddings_for_rogi(self, model, loader, save_dir):
        model.eval()
        embeddings, labels = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                feat, _ = model(data)  # assumes model returns (embedding, logits)
                embeddings.append(feat.cpu().numpy())
                labels.append(data.y.cpu().numpy())
        
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'embeddings.npy'), np.concatenate(embeddings))
        np.save(os.path.join(save_dir, 'labels.npy'), np.concatenate(labels))
        print(f"Saved embeddings and labels to {save_dir}")


    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                _, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)


def freeze_gnn_layers(model, freeze_indices):
    for i, gnn_layer in enumerate(model.gnns):
        if i in freeze_indices:
            for param in gnn_layer.parameters():
                param.requires_grad = False
            print(f"Froze GNN layer {i}")

def main(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

    fine_tune = FineTune(dataset, config)
    fine_tune.train()
    
    if config['dataset']['task'] == 'classification':
        return fine_tune.roc_auc


if __name__ == "__main__":
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]


    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    else:
        raise ValueError('Undefined downstream task!')

    print(config)

    from itertools import combinations

    def generate_freeze_combinations(num_layers=5):
        all_combos = []
        for r in range(num_layers + 1):
            all_combos.extend(combinations(range(num_layers), r))
        return all_combos

    results_list = []
    freeze_combinations = generate_freeze_combinations()

    for target in target_list:
        config['dataset']['target'] = target

        if config['init_from'] == 'random':
            # Run only once, no freezing
            config['freeze_layers'] = []
            print("Running randomly initialized model with no frozen layers.")
            result = main(config)
            results_list.append([target, [], result])

        elif config['init_from'] == 'pretrained':
            # Run full sweep over freezing combinations
            for freeze_layers in freeze_combinations:
                config['freeze_layers'] = list(freeze_layers)
                print(f"Running with frozen layers: {freeze_layers}")
                result = main(config)
                results_list.append([target, freeze_layers, result])

    best_result = max(results_list, key=lambda x: x[2])  # highest ROC AUC
    print("Best config:", best_result)

    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list, columns=["target", "frozen_layers", "roc_auc"])
    df.to_csv(
        'experiments/{}_{}_finetune_with_freezing.csv'.format(config['fine_tune_from'], config['task_name']),
        index=False
    )
