import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import auc , roc_curve , accuracy_score , precision_score, recall_score, f1_score , \
    confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
from dataset_genetic_editing_detection import GenomicSentenceDataset, custom_collate_fn, get_embeddings_for_all_words
import seaborn as sns
import os
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

TRAIN_FP = "<INSERT_HERE>"
TEST_FP = "<INSERT_HERE>"

class GenomicFCNN(nn.Module):
    def __init__(self, input_size, hidden_size=1024, num_hidden_layers=5, output_size=1):
        super(GenomicFCNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))

        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        We assume that the input is of shape (batch_size, seq_length, embeddings_size)
        In FCNN, we flatten the sequence to a single vector that looks like:
        concat(first_vector_embeddings, second_vector_embeddings, ...)
        """
        x = x.reshape(x.shape[0], -1)
        x = self.layers(x)
        x = self.sigmoid(x)
        return x

class EarlyStopper:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            print(f"{self.counter} epochs getting worse, will early stop in {self.patience - self.counter} epochs.")
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def load_data(simulation_data_filepath , train_contigs_fp = TRAIN_FP, test_contigs_fp = TEST_FP,
              batch_size=32, num_workers=4):
    # Get genomic sentences
    toy_data = pd.read_csv(simulation_data_filepath, sep='\t',
                           index_col=0)
    # Turn each word to an embedding
    toy_data = get_embeddings_for_all_words(toy_data)

    # Get names of contigs for the train set and names of contigs for the test set
    with open(train_contigs_fp) as f:
        train_contigs = f.read().split('\n')
    with open(test_contigs_fp) as f:
        test_contigs = f.read().split('\n')

    # Split the data to train and test
    train_data = toy_data[toy_data['contig'].isin(train_contigs)]
    test_data = toy_data[toy_data['contig'].isin(test_contigs)]

    train_dataset = GenomicSentenceDataset(train_data)
    test_dataset = GenomicSentenceDataset(test_data)

    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size,
                                  collate_fn=custom_collate_fn, sampler=RandomSampler(train_dataset))
    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size,
                                 collate_fn=custom_collate_fn)

    return train_dataloader, test_dataloader

def print_status(msg, do_print):
    if do_print:
        print(msg)

def load_data_for_seperate_train_and_test_files(train_data_filepath , test_data_filepath,  batch_size=32, num_workers=4 , do_print=True, shuffle_train_data = True):
    train_data = pd.read_csv(train_data_filepath , sep='\t' , index_col=0)
    test_data = pd.read_csv(test_data_filepath , sep='\t' , index_col=0)

    if shuffle_train_data:
        print_status(" \n Shuffling train dataframe:" , do_print)
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        print_status(" \n Done shuffling data." , do_print)

    print_status(" \nFinished loading dataframe. Create GenomicSentenceDataset: \n" , do_print)

    train_dataset = GenomicSentenceDataset(train_data)
    test_dataset = GenomicSentenceDataset(test_data)

    print_status("Finished creating GenomicSentenceDataset, Create Dataloader: \n" , do_print)

    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size,
                                  collate_fn=custom_collate_fn, sampler=RandomSampler(train_dataset))
    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size,
                                 collate_fn=custom_collate_fn)
    return train_dataloader, test_dataloader

def load_data_for_test_only(test_data_filepath,  batch_size=32, num_workers=4 , do_print=True):
    test_data = pd.read_csv(test_data_filepath, sep='\t', index_col=0)

    print_status(" \nFinished reading CSV. Create GenomicSentenceDataset: \n" , do_print)

    test_dataset = GenomicSentenceDataset(test_data)

    print_status("Finished creating GenomicSentenceDataset, Create Dataloader: \n" , do_print)

    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size,
                                 collate_fn=custom_collate_fn)

    return test_dataloader

def init_fcnn(input_size, hidden_size, num_hidden_layers, output_size):
    return GenomicFCNN(input_size, hidden_size, num_hidden_layers, output_size)

def initialize_and_train_network(train_dataloader, test_dataloader, num_epochs, learning_rate , device, sentence_size, model_filepath):
    log_interval = 2000
    test_interval = 20000
    save_interval = 100000

    model = init_fcnn(input_size=sentence_size * 301, hidden_size=1024, num_hidden_layers=5, output_size=1)

    # Move to GPU
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer)

    # Early Stopper
    early_stopper = EarlyStopper(patience=2, min_delta=0.1)
    step = 0

    for curr_epoch in range(num_epochs):
        iter = 0
        model.train()

        for genes, labels, contigs, bacteria_names in train_dataloader:
            genes = genes.to(device)
            labels = labels.to(device)

            output = model(genes)
            loss = criterion(output.flatten(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % log_interval == 0:
                wandb.log({"Training Loss": loss.item()}, step=step)
                print(f"Epoch: {curr_epoch}, Iter: {iter}, Training loss: {loss.item():.4f}")

            if step % test_interval == 0 and step != 0:
                # Evaluate Model
                test_loss = print_evaluations(model, test_dataloader, device, step)

                # Adjust learning rate
                scheduler.step(test_loss)

            if step % save_interval == 0 and step != 0:
                torch.save(model, model_filepath)
                print(f"Saved model at {model_filepath}, Epoch: {curr_epoch}, Iter: {iter}")

            iter += 1
            step += 1

        # Early stop if necessary
        if early_stopper.early_stop(test_loss):
            print(f"Early stopping at {curr_epoch} epochs")
            break

    # Evaluate model
    print("EVALUATING FINAL MODEL")
    print_evaluations(model, test_dataloader, device , step , do_log=True)

    return model

def print_evaluations(model, test_dataloader, device , step , do_log=True):
    print("EVALUATING MODEL")
    test_acc, test_avg_loss, test_prec, test_rec, test_f1, test_ROC_AUC, test_PR_AUC = evaluate_model(
        model, test_dataloader, device, step)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_avg_loss:.4f}")
    print(f"Test Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")
    print(f"Test ROC AUC: {test_ROC_AUC:.4f}, PR AUC: {test_PR_AUC:.4f}")

    if do_log:
        wandb.log({"Test loss": test_avg_loss,"Test Accuracy": test_acc, "Test Recall": test_rec,
                   "Test Precision": test_prec,"Test F1": test_f1, "Test ROC AUC": test_ROC_AUC,
                   "Test PR AUC": test_PR_AUC}, step)

    return test_avg_loss
def evaluate_model(model, test_dataloader, device, step, thresholds_for_confusion_mat=[0.3 , 0.4 , 0.5 , 0.6 , 0.7]):
    model.eval()  # Set the model to evaluation mode

    all_labels = []
    all_probabilities = []

    criterion = nn.BCELoss()

    with torch.no_grad():
        losses = []

        for genes, labels, contigs, bacteria_names in test_dataloader:
            genes = genes.to(device)
            labels = labels.to(device)

            output = model(genes)
            loss = criterion(output.flatten(), labels)
            losses.append(loss.item())

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(output.cpu().numpy())

    all_probabilities = np.array(all_probabilities)

    # Best threshold
    all_predictions = (all_probabilities >= 0.5).astype(int)

    # Acccuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    # AVg loss
    average_loss = np.mean(np.array(losses))

    # Precision + recall
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)

    # F1
    f1 = f1_score(all_labels, all_predictions)

    #Plotting evaluation metrics
    roc_auc , pr_auc = plot_evaluation(np.array(all_labels), all_probabilities, thresholds_for_confusion_mat, step)

    return accuracy, average_loss, precision, recall, f1 , roc_auc , pr_auc


def plot_evaluation(y_true, y_prob, thresholds, step, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    # Plot Precision-Recall curve
    plot_curve(recall, precision, 'blue', f'PR curve (area = {pr_auc:.2f})', 'Recall', 'Precision',
               'Precision-Recall Curve', os.path.join(output_dir, 'precision_recall_curve.svg'), step)

    # Plot ROC curve
    plot_curve(fpr, tpr, 'red', f'ROC curve (area = {roc_auc:.2f})', 'False Positive Rate', 'True Positive Rate',
               'ROC Curve', os.path.join(output_dir, 'roc_curve.svg'), step)

    # Plot and log histogram of y_prob
    plot_histogram(y_prob, os.path.join(output_dir, 'probability_histogram'), step)

    # Calculate and plot Confusion Matrix
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, threshold, output_dir, step)


    print("Plots saved in the directory:", output_dir)
    return roc_auc , pr_auc

def plot_curve(x, y, color, label, xlabel, ylabel, title, file_path, step):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color=color, lw=2, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='lower left')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(file_path, format='svg', dpi=2400)
    wandb.log({title: plt}, step=step)
    plt.close()


def plot_confusion_matrix(cm, threshold, output_dir, step):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for threshold {threshold}")
    plt.xticks([0.5, 1.5], ["Negative" , "Positive"])
    plt.yticks([0.5, 1.5], ["Negative" , "Positive"])

    cm_path = os.path.join(output_dir, f'confusion_matrix_{threshold}.svg')
    plt.savefig(cm_path, format='svg', dpi=2400)

    png_path = os.path.join(output_dir, f'confusion_matrix_{threshold}.png')
    plt.savefig(png_path, format='png')

    wandb.log({f"Confusion Matrix {threshold}": wandb.Image(png_path)}, step=step)
    plt.close()


def plot_histogram(probas, output_file_path_without_file_type, step):
    plt.figure(figsize=(8, 6))
    plt.hist(probas, bins=50)
    plt.xlabel("Probabilities")
    plt.ylabel("Frequencies")
    plt.title("Predicated Probabilities Histogram")

    output_file_path_svg =output_file_path_without_file_type + ".svg"
    output_file_path_png = output_file_path_without_file_type + ".png"

    plt.savefig(output_file_path_svg, format='svg', dpi=2400)

    plt.savefig(output_file_path_png, format='png')
    wandb.log({"Predicated Probabilities Histogram" : wandb.Image(output_file_path_png)}, step=step)
    plt.close()


def return_gpu_device():
    if torch.cuda.is_available()== False:
        print("ERROR! NO GPU CONNECTED")
        return "cpu"
    else:
        print("Found GPU")
        return torch.device("cuda:0")