
# ResNet50 - Fundus Disease Classification

This folder contains the implementation and evaluation of the ResNet-50 model applied to the ODIR-5K dataset.

## Contents
- `resnet50model.ipynb`: Jupyter notebook containing the full training pipeline, evaluation, and visualizations.
- `resnet50_accuracy_plot.png`: Line plot showing training and validation accuracy over 20 epochs.
- `resnet50_loss_plot.png`: Line plot showing training and validation loss over 20 epochs.

## Summary
The model uses transfer learning with ResNet-50 pretrained on ImageNet. It was trained in two phases:
1. Top layers trained with base frozen (10 epochs)
2. Final 40 layers unfrozen and fine-tuned (10 epochs)

Final validation accuracy plateaued at approximately **45–48%**, indicating limited generalization capacity under current conditions.

Model saved as: `resnet50_fundus_finetuned.keras` (not included for size).

---
For more details, refer to the final report or explore the notebook directly.
