# Language-Model-Based Detection of Genetic Editing in Bacteria

Code accompanying the preprint **"Language-Model-Based Detection of Genetic Editing in Bacteria"** by Edan Gabay and David Burstein.

This project uses a natural language processing approach to detect artificial gene insertions in bacterial genomes. Instead of treating nucleotides or amino acids as tokens, we represent genes as words and short genomic regions as sentences. The model learns to identify inserted genes based on the unnatural genomic context they create.

## Model

Each sample consists of a sequence of five consecutive gene-family identifiers.

Gene identifiers are represented using pretrained 300-dimensional gene embeddings. An additional feature is used to represent genes without an available embedding.

The repository includes two classification models:

- A fully connected neural network used as the baseline model.
- A transformer-encoder classifier used as the final model.

The transformer model was first trained on simulated random bacterial gene insertions and then fine-tuned on simulated insertions of potentially harmful genes.

## Repository files

- `dataset_genetic_editing_detection.py` – dataset handling and conversion of gene identifiers to embeddings.
- `transformer_dataset_w2v_embeddings.py` – data loading for the transformer model.
- `classification_nn_genetic_editing_detection.py` – baseline fully connected neural network and general evaluation functions.
- `genomic_transformer_genetic_editing_detection.py` – transformer architecture and training pipeline.
- `genomic_transformer_evaluations_genetic_editing_detection.py` – transformer evaluation and extraction of high-scoring gene sequences.

## Requirements

The code was implemented in Python using PyTorch.

Main dependencies include:

- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Weights & Biases (`wandb`)

The code also requires the pretrained gene embeddings used in the study. Update `EMBEDDINGS_MODEL_PATH` and the relevant input/output paths before running the models.

## Training

The final transformer model uses three transformer-encoder layers with two attention heads.

Training was performed in two stages:

1. Pre-training on the random gene insertion simulation.
2. Fine-tuning on the malicious gene insertion simulation, while freezing the transformer-encoder layers.

Training and test datasets are expected as tab-separated files containing five gene identifiers per sample, together with the sample label and contig information.

## Evaluation

The evaluation code reports:

- Accuracy
- Precision
- Recall
- F1 score
- AUROC
- AUPRC

It also generates ROC curves, precision-recall curves, prediction histograms, and confusion matrices.

The final model achieved an AUROC and AUPRC of up to **0.95** on the simulated malicious gene insertion task.

## Citation

If you use this code, please cite:

**Gabay E, Burstein D. Language-Model-Based Detection of Genetic Editing in Bacteria.**
