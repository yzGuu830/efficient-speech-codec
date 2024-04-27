import torch
import torch.nn as nn
import warnings


@torch.no_grad()
def codebook_init_forward_hook_pvq(self, input, output):
	""" initializes codebook from data """

	if (not self.training) or (self.codebook_initialized.item() == 1):
		return # no initialization during inference

	if self.verbose_init is True:
		if self.kmeans_init is None:
			print("Initializing Product VQs with KaimingNormal")
		elif self.kmeans_init is True:
			print('Initializing Product VQs with k-means++')
		elif self.kmeans_init is False:
			print('Initializing Product VQs by randomly choosing from z_e')

	outputs, _ = output
	_, z_e_downs, _ = outputs
	# z_e_downs [B, group_size, T, codebook_dim]
	for i in range(self.num_vqs):
		if self.kmeans_init is not None:
			z_e_i = z_e_downs[:,i] 		# [B, T, codebook_dim]
			init_codebook = sample_centroids(z_e_i, self.codebook_size, self.kmeans_init)
			self.vqs[i].embedding.weight.data = init_codebook
		else:
			nn.init.kaiming_normal_(self.vqs[i].embedding.weight) 		
	
	self.codebook_initialized.fill_(1) # set boolean flag
	return

@torch.no_grad()
def sample_centroids(z_e, codebook_size, use_kmeans=False):
	""" create an initialize codebook one-time from z_e
	Args: 
		z_e: encoded embedding Tensor of size [bs,T,d]
		codebook_size: number of codewords

		returns: 
			new_codebook: Tensor of size [codebook_size, d]
	"""

	z_e = z_e.reshape(-1, z_e.size(-1)) # bs*T, d
	if codebook_size >= z_e.size(0):
		e_msg = f'\ncodebook size > warmup samples: {codebook_size} vs {z_e.size(0)}. ' + \
					'recommended to decrease the codebook size or increase batch size.'
		warnings.warn(e_msg)
		# repeat until it fits and add noise
		repeat = 1 + codebook_size // z_e.shape[0]
		new_codes = z_e.data.tile([repeat, 1])[:codebook_size]
		new_codes += 1e-3 * torch.randn_like(new_codes.data)
	else:
		# you have more warmup samples than codebook. subsample data
		if use_kmeans:
			from torchpq.clustering import KMeans
			kmeans = KMeans(n_clusters=codebook_size, distance='euclidean', init_mode="kmeans++")
			kmeans.fit(z_e.data.T.contiguous())
			new_codes = kmeans.centroids.T
		else:
			indices = torch.randint(low=0, high=codebook_size, size=(codebook_size,))
			indices = indices.to(z_e.device)
			new_codes = torch.index_select(z_e, 0, indices).to(z_e.device).data

	return new_codes