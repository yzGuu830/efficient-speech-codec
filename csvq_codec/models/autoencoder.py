from modules.model import CSVQ_Encoder, CSVQ_Decoder
from losses.csvq import Compress_Loss, Mel_Loss
from models.utils import reconstruct_audio
from config import cfg
import torch
import torch.nn as nn
import math


class CSVQ_Codec(nn.Module):
    def __init__(self, in_channels=2, ch_mult=(4,4,8,8,16,16), F=201, Groups=4, codebook_size=1024, vq_commit=0.25, 
                C_0=128, num_res_block=6, num_GRU_layers=2, C_down_rate=0.3, mel_factor=1.0, vq_factor=1.0):
        super().__init__()
        self.B = len(ch_mult)
        self.encoder = CSVQ_Encoder(in_channels, ch_mult, num_res_block, num_GRU_layers)
        self.decoder = CSVQ_Decoder(in_channels=in_channels, ch_mult=ch_mult, Groups=Groups, F=F, 
                                num_res_block=num_res_block, num_GRU_layers=num_GRU_layers,
                                codebook_size=codebook_size, vq_commit=vq_commit, C_0 = C_0, C_down_rate=C_down_rate)

        self.recon_loss = Compress_Loss()
        self.mel_loss = Mel_Loss()

        self.mel_factor, self.vq_factor = mel_factor, vq_factor

        self.bitrate = (self.B + 1) * Groups * math.log2(codebook_size) * 50

        print(f"Initialized CSVQ_Model: MAX_Bitrate is {0.001*self.bitrate} kbps")

    def encode(self, input):
        '''
        input: N,2,F,T
        '''
        encoded_hs = self.encoder(input)
        return encoded_hs
        
    def decode(self, encoded_hs, Bs):
        recon_feat, vq_loss = self.decoder(encoded_hs, Bs)
        return recon_feat, vq_loss

    def forward(self, input, Bs=None):
        '''
        input:       {'audio': (N, L), 'stft_feat': (N, F, T, 2), 'speaker_id':(N,1)}
        Bs: target bitstream to achieve (1 <= Bs <= len(ch_mult)+1)
        recon_feat:  (N, 2, F, T) 
        recon_audio: (N, L)
        '''
        raw_feat = input['stft_feat'].permute(0,3,1,2)

        encoded_hs = self.encode(raw_feat)
        recon_feat, vq_loss = self.decode(encoded_hs, Bs)                # (N, 2, F, T)

        # recon_feat = torch.nn.functional.pad(recon_feat, (0,0,0,1))  # (N, 2, F, T)

        recon_audio = reconstruct_audio(recon_feat)

        recon_loss = self.recon_loss(raw_feat, recon_feat)
        mel_loss = self.mel_loss(input['audio'], recon_audio)

        loss = recon_loss + self.vq_factor * vq_loss + self.mel_factor * mel_loss

        output = {
                  'loss': loss,
                  'recon_loss': recon_loss, 'vq_loss': vq_loss, 'mel_loss': mel_loss,
                  'recon_feat': recon_feat, 'recon_audio': recon_audio
                 }

        return output

def init_model():
    ch_mult = cfg['csvq_codec']['ch_mult']
    F = cfg['raw_F']
    Groups = cfg['csvq_codec']['Groups']
    C_0 = cfg['csvq_codec']['C_down0']
    C_down_rate = cfg['csvq_codec']['C_down_rate']
    codebook_size = cfg['csvq_codec']['codebook_size']
    vq_commit = cfg['csvq_codec']['vq_commit']
    mel_factor = cfg['csvq_codec']['mel_factor']
    vq_factor = cfg['csvq_codec']['vq_factor']
    num_res_block = cfg['csvq_codec']['num_res_block']
    num_GRU_layers = cfg['csvq_codec']['num_GRU_layers']
    
    
    model = CSVQ_Codec(in_channels=2, ch_mult=ch_mult, F=F, Groups=Groups, codebook_size=codebook_size, vq_commit=vq_commit, 
                        C_0=C_0, num_res_block=num_res_block, num_GRU_layers=num_GRU_layers,C_down_rate=C_down_rate, 
                        mel_factor=mel_factor, vq_factor=vq_factor)

    return model





