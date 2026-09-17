import numpy as np
import torch
from torch.utils.data import Dataset
from genomic_embeddings import Embeddings
import pandas as pd

EMBEDDINGS_MODEL_PATH = "<INSERT_PATH_HERE>"


def get_embeddings(word, gene_embeddings):
	if word == "UNKNOWN":
		return np.append(np.zeros(300), 1)
	else:
		return np.append(gene_embeddings.wv[word], 0)

def get_embeddings_for_all_words(df, model_path=EMBEDDINGS_MODEL_PATH):
	gene_embeddings = Embeddings.load_embeddings(model_path)
	df_result = df.copy()

	if "bacteria_name" in df.columns:
		df.drop(columns="bacteria_name" , inplace=True)

	for col in df.columns:
		if col not in ['contig', 'label']:
			df_result[col] = df_result[col].apply(get_embeddings, args=(gene_embeddings,))

	return df_result

def get_embeddings_for_dataloader(df , gene_embeddings):
	df_result = df.copy()

	if "bacteria_name" in df.columns:
		df.drop(columns="bacteria_name" , inplace=True)

	for col in df.columns:
		if col not in ['contig', 'label']:
			df_result[col] = df_result[col].apply(get_embeddings, args=(gene_embeddings,))

	return df_result

def sum_col_strings(row):
	return " , ".join(row.astype(str))


class GenomicSentenceDataset(Dataset):
	def __init__(self, genomic_df):
		print("Get contigs and labels:")
		self.contigs = genomic_df['contig']
		self.labels = genomic_df['label']

		print("Get bacteria names:")

		if not 'bacteria_name' in genomic_df.columns:
			genomic_df["bacteria_name"] = genomic_df["contig"]

		self.bacteria_names = genomic_df["bacteria_name"]

		print("Get Gene Annotations:")
		self.genes = genomic_df.drop(['contig', 'label', 'bacteria_name'], axis=1)
		self.gene_annotations = pd.DataFrame({"gene_annotations": self.genes.apply(sum_col_strings, axis=1) })

		print("Get embeddings:")
		self.embeddings = Embeddings.load_embeddings(EMBEDDINGS_MODEL_PATH)



	def __len__(self):
		return len(self.genes)

	def __getitem__(self, idx):
		# Get the batch
		sample_df = self.genes.loc[[idx]]

		# Get embeddings for it
		sample_df = get_embeddings_for_dataloader(sample_df , self.embeddings)

		# Turn the gene embeddings into a numpy array
		sample_numpy_arr = np.array(sample_df.to_numpy().tolist())

		# Turn the numpy array to a tensor
		sample = torch.tensor(sample_numpy_arr).type(torch.float32)

		contig = self.contigs.iloc[idx]
		label = int(self.labels.iloc[idx])
		bacteria_name = self.bacteria_names.iloc[idx]
		gene_anots = self.gene_annotations.iloc[idx]
		return sample, label, contig, bacteria_name , gene_anots



def custom_collate_fn(batch):
	samples, labels, contigs, bacteria_names , gene_anots = zip(*batch)
	samples = torch.stack(samples)
	labels = torch.tensor(labels, dtype=torch.float32)
	return samples, labels, contigs, bacteria_names , gene_anots

