import torch
from dataset_genetic_editing_detection import GenomicSentenceDataset
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler

EMBEDDINGS_MODEL_PATH = "<INSERT_PATH_HERE>"

def custom_collate_fn(batch):
	samples, labels, contigs, bacteria_names , gene_anots = zip(*batch)
	samples = torch.cat(samples, dim=0)
	batch_size , block_size , embedding_len = samples.shape
	extra_dimension = torch.zeros(batch_size , block_size , 1)
	# Concatenate the extra dimension to the original tensor along the third dimension (dimension 2)
	samples = torch.cat([samples, extra_dimension], dim=2)
	labels = torch.tensor(labels, dtype=torch.float32)
	return samples, labels, contigs, bacteria_names , gene_anots

def load_transformer_data(train_data_filepath , test_data_filepath,  batch_size=30, num_workers=4 , do_print=True, shuffle_train_data = True, just_test = False):
	if not just_test:
		print("Loading train dataframe")
		train_data = pd.read_csv(train_data_filepath , sep='\t' , index_col=0)
		train_data_len = len(train_data)
		num_iters = train_data_len / batch_size

	print("Loading test dataframe")
	test_data = pd.read_csv(test_data_filepath , sep='\t' , index_col=0)

	if shuffle_train_data and not just_test:
		print_status(" \n Shuffling train dataframe:" , do_print)
		train_data = train_data.sample(frac=1).reset_index(drop=True)
		print_status(" \n Done shuffling data." , do_print)

	print_status("\nFinished loading dataframe.\nCreate GenomicSentenceDataset:\n" , do_print)

	if not just_test:
		train_dataset = GenomicSentenceDataset(train_data)
		print_status("\nFinished creating GenomicSentenceDataset for train set, Create Dataloader:\n", do_print)

		train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size,
									  collate_fn=custom_collate_fn, sampler=RandomSampler(train_dataset))

	test_dataset = GenomicSentenceDataset(test_data)

	test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size,
								 collate_fn=custom_collate_fn)

	if just_test:
		return test_dataloader

	return train_dataloader, test_dataloader , num_iters

def print_status(msg, do_print):
    if do_print:
        print(msg)