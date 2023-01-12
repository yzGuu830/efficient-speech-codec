import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

from models.quantizer import Quantizer

from models.vqgan_model import Encoder,Decoder
from .utils import init_param, reconstruct_audio
from utils import pad_img
import numpy as np
from math import nan
from models.convolutional_encoder import ConvEncoder
from models.convolutional_decoder import ConvDecoder
    

class VQVAE(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=128, depth=1, 
                    num_res_block=2, res_size=32, embedding_size=64, num_embedding=512, vq_commit=0.25):
        super().__init__()
        # self.encoder = ConvEncoder(in_channels=input_size, out_channels=embedding_size, num_hiddens=hidden_size, 
        #                     num_residual_layers=num_res_block, num_residual_hiddens=res_size, 
        #                     hierarchy_layer=depth, verbose=False)
        # self.quantizer = Quantizer(embedding_size, num_embedding, vq_commit)
        # self.decoder = ConvDecoder(in_channels=embedding_size, out_channels=output_size, num_hiddens=hidden_size, 
        #                     num_residual_layers=num_res_block, num_residual_hiddens=res_size, 
        #                     hierarchy_layer=depth, verbose=False)
        ddconfig={
                "double_z": False,
                "z_channels": 128,
                "resolution": 128,
                "in_channels": input_size,
                "out_ch": output_size,
                "ch": hidden_size,
                "ch_mult": [1,2,4],  # num_down = len(ch_mult)-1
                "num_res_blocks": num_res_block,
                "attn_resolutions": [16],
                "dropout": 0.0
                }
        self.encoder, self.decoder = Encoder(**ddconfig), Decoder(**ddconfig)
        self.quantizer = Quantizer(embedding_size, num_embedding, vq_commit)

        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embedding_size, 1)
        self.post_quant_conv = torch.nn.Conv2d(embedding_size, ddconfig["z_channels"], 1)

    def encode(self, input):
        x = input
        encoded = self.encoder(x)
        encoded = self.quant_conv(encoded)
        quantized, diff, code = self.quantizer(encoded)
        return quantized, diff, code

    def decode(self, quantized):
        quantized = self.post_quant_conv(quantized)
        decoded = self.decoder(quantized)
        return decoded

    def decode_code(self, code):
        quantized = self.quantizer.embedding_code(code).transpose(1, -1).contiguous()
        decoded = self.decode(quantized)
        return decoded

    def forward(self, input, Epoch=None):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        if 'feature' in input.keys():
            x = input['feature']
        else:
            x = input['data']

        quantized, diff, output['code'] = self.encode(x)
        decoded = self.decode(quantized)
        decoded = pad_img(input['feature'],decoded)
        output['recon_feature'] = decoded
        if 'feature' in input.keys():
            output['recon_audio'] = reconstruct_audio(X=output['recon_feature'],input=input)
        
        output['loss'] = F.mse_loss(output['recon_feature'], input['feature']) + diff
        # output['loss'] = F.mse_loss(output['recon_feature'], input['data']) + diff
        
        return output


def vqvae():
    data_shape = cfg['data_shape']
    hidden_size = cfg['vqvae']['hidden_size']
    depth = cfg['vqvae']['depth']
    num_res_block = cfg['vqvae']['num_res_block']
    res_size = cfg['vqvae']['res_size']
    embedding_size = cfg['vqvae']['embedding_size']
    num_embedding = cfg['vqvae']['num_embedding']
    vq_commit = cfg['vqvae']['vq_commit']
    model = VQVAE(input_size=data_shape[0], hidden_size=hidden_size, depth=depth, num_res_block=num_res_block,
                  res_size=res_size, embedding_size=embedding_size, num_embedding=num_embedding, vq_commit=vq_commit)
    model.apply(init_param)
    return model
