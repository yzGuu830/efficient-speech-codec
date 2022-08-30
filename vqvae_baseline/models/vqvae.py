from tabnanny import verbose
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import Quantizer
from models.decoder import Decoder
from config import cfg
from utils import pad_result
from torch_dct import isdct_torch

class VQVAE(nn.Module):
    '''
    General VQVAE framework combining Encoder, Quantizer, Decoder together
    '''

    def __init__(self, h_dim, res_h_dim, n_res_layers, 
                n_embeddings, embedding_dim, beta) -> None:
        super(VQVAE,self).__init__()
        

        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        self.vector_quantization = Quantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(h_dim, h_dim, n_res_layers, res_h_dim)
        

    
    def forward(self, x, verbose=False):

        z_e = self.encoder(x) # encoder output
        codebook_loss, z_q, _, _, perplexity = self.vector_quantization(z_e) # quantizer output
        x_prime = self.decoder(z_q) # decoder output
        
        x_prime = pad_result(x_prime,x)
        recon_loss = torch.mean((x_prime - x)**2) # reconstruction loss
        # recon_audio = [isdct_torch(x_prime[i],frame_step=256) for i in range(len(x_prime))] # reconstructed audio

        output = {}
        output['loss'] = codebook_loss + recon_loss
        output['x_prime'] = x_prime
        # output['recon_audio'] = recon_audio
        output['perplexity'] = perplexity

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_prime.shape)
            assert False
        

        return output


def vqvae():
    h_dim = cfg['vqvae']['h_dim']
    res_h_dim = cfg['vqvae']['res_h_dim']
    n_res_layers = cfg['vqvae']['n_res_layers']
    n_embeddings = cfg['vqvae']['n_embeddings']
    embedding_dim = cfg['vqvae']['embedding_dim']
    beta = cfg['vqvae']['beta']
    model = VQVAE(h_dim,res_h_dim,n_res_layers,n_embeddings,embedding_dim,beta)
    return model
