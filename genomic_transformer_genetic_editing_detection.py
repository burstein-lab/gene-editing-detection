import math
import wandb
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from classification_nn_genetic_editing_detection import return_gpu_device , EarlyStopper
from transformer_dataset_w2v_embeddings import load_transformer_data
from genomic_transformer_evaluations_genetic_editing_detection import log_evaluations

sen_len = 5

class ClassificationTransformerGene2VecEmbeddings(nn.Module): #Using https://pytorch.org/tutorials/beginner/transformer_tutorial.html
	def __init__(self, nhead=2, d_hid=302, nlayers=1, d_embeddings=302, dropout=0.1, do_positional_embeddings=False):
		super().__init__()
		self.d_embeddings = d_embeddings
		self.do_positional_embeddings = do_positional_embeddings

		if do_positional_embeddings:
			self.pos_encoder = PositionalEncoding(d_embeddings, dropout)

		encoder_layers = TransformerEncoderLayer(d_model = d_embeddings,
												 nhead = nhead,
												 dim_feedforward = d_hid,
												 dropout = dropout,
												 batch_first =True)

		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

		half_d_embeddings = int(0.5 * d_embeddings)

		# Concat Transformer output
		concated_sen_length = sen_len*d_embeddings

		self.classifier_layers = nn.Sequential(
			nn.Linear(in_features= concated_sen_length, out_features=d_embeddings),
			nn.ReLU(),
			nn.Linear(in_features=d_embeddings, out_features=half_d_embeddings),
			nn.ReLU(),
			nn.Linear(in_features=half_d_embeddings, out_features=1)
		)

		self.sigmoid = nn.Sigmoid()

	def forward(self , src):
		# Input is Danielle's embeddings
		embedded = src

		if self.do_positional_embeddings:
			embedded = self.pos_encoder(embedded)

		# Pass through transformer
		transformer_output = self.transformer_encoder(embedded)

		# Concat the transformer output for each sample
		batch_size, block_size, embedding_size = src.shape
		concatenated_transformer_output = transformer_output.view(batch_size, -1)

		# Pass through classification head
		classifier_layers_output = self.classifier_layers(concatenated_transformer_output)
		final_output = self.sigmoid(classifier_layers_output)

		return final_output

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=20):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(1, max_len, d_model)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:, :x.size(1)] # x: Tensor, shape ``[batch_size, seq_len, embedding_dim]
		return self.dropout(x)

def train_transformer(device, n_encoder_layers, num_ma_heads , num_epochs, train_dataloader, num_iters_per_epoch,
					  test_dataloader , output_filepath , learning_rate, pretrained_filepath =None,
					  num_of_layers_to_freeze=2):

	if pretrained_filepath != None:
		model = torch.load(pretrained_filepath)


		if num_of_layers_to_freeze > 0:
			print("Freeze paramaters for first two transformer layers:")
			for i in range(num_of_layers_to_freeze):
				print(f"Freezing params for layer {i}")
				for param in model.transformer_encoder.layers[i].parameters():
					param.requires_grad = False

	else:
		model = ClassificationTransformerGene2VecEmbeddings(nhead = num_ma_heads, nlayers=n_encoder_layers)

	model.to(device)
	criterion = nn.BCELoss()
	model_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
	print(f"{num_iters_per_epoch} iterations per epoch, {num_iters_per_epoch * num_epochs} iterations for cosine decay")
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=num_epochs * num_iters_per_epoch, eta_min=5*1e-7)

	log_interval = 10000
	test_interval = 100000
	save_interval = 100000
	step = 0
	model.train()

	# Early Stopper
	early_stopper = EarlyStopper(patience=1, min_delta=0.05)

	for epoch in range(num_epochs):

		iter = 0

		for genes, labels, contigs, bacteria_names , gene_anots in train_dataloader:
			genes = genes.to(device)
			labels = labels.to(device)

			output = model(genes)
			loss = criterion(output.flatten(), labels)

			model_optimizer.zero_grad()
			loss.backward()
			model_optimizer.step()
			scheduler.step()

			if iter % log_interval == 0:
				wandb.log({"Training Loss": loss.item()}, step=step)
				print(f"Epoch: {epoch}, Iter: {iter}, Training loss: {loss.item():.4f}")

			if step % test_interval == 0 and step != 0:
				# Evaluate Model
				test_loss = log_evaluations(model, criterion, test_dataloader, device, step, get_last_learning_rate(model_optimizer, scheduler))
				print(f"Average Test Loss = {test_loss}")

			if step % save_interval == 0  and step != 0:
				torch.save(model, output_filepath)
				print(f"Saved model at {output_filepath}, Epoch: {epoch}, Iter: {iter}")

			iter += 1
			step += 1

		if early_stopper.early_stop(test_loss):
			print(f"Early stopping at {epoch} epochs")
			break

		# Evaluate model
		log_evaluations(model, criterion, test_dataloader, device, step, get_last_learning_rate(model_optimizer, scheduler))

	torch.save(model , output_filepath)

def get_last_learning_rate(optimizer, scheduler):
	if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
		return optimizer.param_groups[0]["lr"]
	return scheduler.get_last_lr()[0]

def transformer_pipeline(train_data_filepath , test_data_filepath , model_filepath, num_encoder_layers , num_ma_heads ,
						 epochs , lr, just_test=False, pretrained_filepath=None, num_layers_to_freeze=0):

	device = return_gpu_device()

	print("LOADING DATA")
	if just_test:
		print("JUST TEST")
		test_dataloader = load_transformer_data(train_data_filepath, test_data_filepath, just_test=just_test)
		print("FINISHED LOADING DATA, TESTING MODEL")

		print("LOADING MODEL FROM FILE")
		model = torch.load(model_filepath).to(device)
		criterion = nn.BCELoss()
		log_evaluations(model, criterion, test_dataloader, device, step=1, lr=None)

	else:
		train_dataloader, test_dataloader , num_iters = load_transformer_data(train_data_filepath, test_data_filepath)
		print("FINISHED LOADING DATA, TRAINING MODEL")
		train_transformer(device=device, n_encoder_layers=num_encoder_layers, num_ma_heads=num_ma_heads,
						  num_epochs=epochs, train_dataloader=train_dataloader, num_iters_per_epoch=num_iters,
						  test_dataloader=test_dataloader, output_filepath=model_filepath, learning_rate=lr,
						  pretrained_filepath=pretrained_filepath, num_of_layers_to_freeze=num_layers_to_freeze)


