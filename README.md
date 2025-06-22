# 🧬 Investigating Factors for Effective Transfer-Learning with Chemical Graphs

This project investigates the effects of **layer freezing strategies** in pretrained **graph neural networks (GNNs)** on molecular property prediction tasks. It evaluates how partial or full freezing of GNN layers impacts downstream performance across several models and datasets.

## 🚀 Overview

- Uses **pretrained GNN encoders** from:
  - [MolCLR](https://github.com/yuyangw/MolCLR)
  - [GraphLoG](https://github.com/DeepGraphLearning/GraphLoG)
- Includes **random initialization** baselines for comparison.
- Standardizes projection heads (MLPs) across all models.
- Evaluates using the **ROGI-XD** roughness index:
  - [ROGI-XD](https://github.com/coleygroup/rogi-xd)

## 🧪 Tasks

Experiments are conducted on the following [MoleculeNet](https://moleculenet.org/) datasets:
- BACE
- BBBP

Model is fine-tuned with different layer freezing combinations (e.g., freeze bottom 1, 2, 3 layers, etc.).

## 🏗 Architecture

- Backbone: GIN/GINE-based GNNs from MolCLR and GraphLoG
- Pooling: Mean (default) / Max / Add
- Projection Head: MLP with 2 hidden layers (ReLU activations)
- Compatible with MolCLR and GraphLoG pretrained weights

## 📦 Dependencies

- This project is heavily inspired by and adapted from:
  - [MolCLR](https://github.com/yuyangw/MolCLR)
  - [GraphLoG](https://github.com/DeepGraphLearning/GraphLoG)
  - [ROGI-XD](https://github.com/coleygroup/rogi-xd)
- All environments from original repositories are needed

