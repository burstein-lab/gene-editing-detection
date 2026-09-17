import torch
import wandb
import numpy as np
import pandas as pd
from datetime import datetime
import os
from sklearn.metrics import  accuracy_score , precision_score, recall_score, f1_score
from classification_nn_genetic_editing_detection import plot_evaluation , return_gpu_device

EMBEDDING_SIZE = 302
NUM_MA_HEADS = 2
DROP_PROB = 0.1
BLOCK_SIZE = 5

device = return_gpu_device()

def log_evaluations(model, criterion, test_dataloader, device , step , lr="not_specified"):
	test_acc, test_avg_loss, test_prec, test_rec, test_f1, test_ROC_AUC, test_PR_AUC = evaluate_model(model,
																									  criterion,
																									  test_dataloader,
																									  device, step)

	wandb.log({"Test loss": test_avg_loss,"Test Accuracy": test_acc, "Test Recall": test_rec,
			   "Test Precision": test_prec,"Test F1": test_f1, "Test ROC AUC": test_ROC_AUC,
			   "Test PR AUC": test_PR_AUC, "Learning rate": lr},  step)

	return test_avg_loss
def evaluate_model(model, criterion, test_dataloader, device, step, thresholds_for_confusion_mat=[0.2,0.3, 0.4 ,0.45, 0.5 ,0.55, 0.6]):
	model.eval()  # Set the model to evaluation mode

	all_labels = []
	all_probabilities = []


	with torch.no_grad():
		losses = []

		for genes, labels, contigs, bacteria_names , gene_anots in test_dataloader:
			genes = genes.to(device)
			labels = labels.to(device)

			output = model(genes)
			loss = criterion(output.flatten(), labels)
			losses.append(loss.item())

			all_labels.extend(labels.cpu().numpy())
			all_probabilities.extend(output.cpu().numpy())

	all_probabilities = np.array(all_probabilities)

	# calc prediction using threshold
	all_predictions = (all_probabilities >= 0.5).astype(int)

	# Acccuracy
	accuracy = accuracy_score(all_labels, all_predictions)

	# Avg loss
	average_loss = np.mean(np.array(losses))

	# Precision + recall
	precision = precision_score(all_labels, all_predictions)
	recall = recall_score(all_labels, all_predictions)

	# F1
	f1 = f1_score(all_labels, all_predictions)

	now = datetime.now()
	plot_dir = os.path.join("plots", now.strftime("%d.%m.%Y.%H:%M"))

	#Plotting evaluation metrics
	roc_auc , pr_auc = plot_evaluation(np.array(all_labels), all_probabilities, thresholds_for_confusion_mat, step, plot_dir)

	return accuracy, average_loss, precision, recall, f1 , roc_auc , pr_auc


def gene_list_to_dataframe(data):
	# Split each string in the sublists by comma and strip whitespace
	processed_data = [
		[item.strip() for item in sublist[0].split(',')]
		for sublist in data
	]

	# Create the DataFrame
	df = pd.DataFrame(
		processed_data,
		columns=['gene_1', 'gene_2', 'gene_3', 'gene_4', 'gene_5']
	)

	return df
def get_probs_for_gene_seqs(model, criterion, test_dataloader, device, threshold=0.9):
	model.eval()  # Set the model to evaluation mode

	all_probabilities = []
	all_contigs = []
	all_genes = []

	with torch.no_grad():
		losses = []

		for genes, labels, contigs, bacteria_names , gene_anots in test_dataloader:
			all_genes.extend(gene_anots)
			genes = genes.to(device)
			labels = labels.to(device)

			output = model(genes)
			loss = criterion(output.flatten(), labels)
			losses.append(loss.item())

			all_probabilities.extend(output.cpu().numpy())
			all_contigs.extend(contigs)

	all_probabilities = np.array(all_probabilities)

	if all_probabilities.ndim == 2:
		all_probabilities = all_probabilities.flatten()

	all_contigs = np.array(all_contigs)
	all_genes = np.array(all_genes)

	df = gene_list_to_dataframe(all_genes)
	df["prob"] = all_probabilities

	high_prob_gene_sequences = df[df['prob'] > threshold]

	print(f"\nHigh probability gene sequences with score over {threshold}: \n")
	print(high_prob_gene_sequences)
	return high_prob_gene_sequences

def check_sequence(words, genes):
	return all(word == gene for word, gene in zip(words, genes))

def get_gene_seq_info(gene_seqs_df , gene_features_with_locations_as_words_df):
	gene_columns = ['gene_1', 'gene_2', 'gene_3', 'gene_4', 'gene_5']
	word_series = gene_features_with_locations_as_words_df['word']
	results = []

	# Iterate through each row in gene_seqs_df
	for idx, row in gene_seqs_df.iterrows():
		genes = row[gene_columns].tolist()

		# Use rolling window to check for consecutive matches

		for i in range(len(word_series) - 4):
			if check_sequence(word_series.iloc[i:i + 5], genes):
				results.append({
					'gene_seq_index': idx,
					'start': gene_features_with_locations_as_words_df.loc[i , 'start'],
					'end': gene_features_with_locations_as_words_df.loc[i+4, 'end'],
					'prob': row["prob"],
					'feature_1': gene_features_with_locations_as_words_df.loc[i , 'feature'],
					'feature_2': gene_features_with_locations_as_words_df.loc[i+1 , 'feature'],
					'feature_3': gene_features_with_locations_as_words_df.loc[i + 2 , 'feature'],
					'feature_4': gene_features_with_locations_as_words_df.loc[i + 3 , 'feature'],
					'feature_5': gene_features_with_locations_as_words_df.loc[i + 4 , 'feature'],
					'word_1': gene_features_with_locations_as_words_df.loc[i, 'word'],
					'word_2': gene_features_with_locations_as_words_df.loc[i + 1, 'word'],
					'word_3': gene_features_with_locations_as_words_df.loc[i + 2, 'word'],
					'word_4': gene_features_with_locations_as_words_df.loc[i + 3, 'word'],
					'word_5': gene_features_with_locations_as_words_df.loc[i + 4, 'word'],
				})
				break  # Move to the next row in gene_seqs_df

	# Convert results to a DataFrame
	results_df = pd.DataFrame(results)

	return results_df


