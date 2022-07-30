from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import Quantizer
from models.decoder import Decoder



class VQVAE(nn.Module):
    '''
    General VQVAE framework combining Encoder, Quantizer, Decoder together
    '''

    def __init__(self, h_dim, res_h_dim, n_res_layers, 
                n_embeddings, embedding_dim, beta, save_img_embedding_map=False) -> None:
        super(VQVAE,self).__init__()
        

        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        self.vector_quantization = Quantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)


    
    def forward(self, x, verbose=False):


        z_e = self.encoder(x) # encoder output
        codebook_loss, z_q, _, _, perplexity = self.vector_quantization(z_e) # quantizer output
        x_prime = self.decoder(z_q) # decoder output

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_prime.shape)
            assert False
        

        return codebook_loss, x_prime, perplexity




