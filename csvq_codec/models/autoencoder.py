from modules.tf_net import F_downsample, F_upsample, TCM
from losses.csvq import MSE_Loss, Mel_Loss
from models.fuse import Fuse_VQ
from models.quantizer import Group_Quantize
from models.utils import feat2spec, power_law_compress, find_nearest, fold, unfold

from config import cfg
import torch.nn as nn
import torchaudio
import math
import torch


class CSVQ_TFNet(nn.Module):
    def __init__(self, in_channels=2, ch_mult=(8,8,12,12,16,32), num_groups=6, TCM_Dilation=(1,2,4,8), num_RNN_layers=4,
                    codebook_size=1024, vq_commit=0.25, down_rate=0.4, mel_factor=0.25, vq_factor=0.25, 
                    power_law_ratio=0.3, n_fft=2048, n_mels=48):
        super().__init__()

        self.Bitstreams = len(ch_mult)

        self.encoder = CSVQ_Encoder(in_channels, ch_mult, TCM_Dilation, num_RNN_layers)
        self.decoder = CSVQ_Decoder(in_channels, ch_mult, num_groups, TCM_Dilation, 
                                    num_RNN_layers, codebook_size, vq_commit, down_rate)

        self.recon_loss = MSE_Loss()
        self.mel_loss = Mel_Loss(n_fft=n_fft, n_mels=n_mels)

        self.mel_factor, self.vq_factor, self.power_law = mel_factor, vq_factor, power_law_ratio

        self.bitrate = self.Bitstreams * num_groups * math.log2(codebook_size) * 50

        self.ft = torchaudio.transforms.Spectrogram(win_length=int(cfg['win_length']*cfg['sr']*1e-3),
                                            hop_length=int(cfg['hop_length']*cfg['sr']*1e-3), power=None)
        self.ift = torchaudio.transforms.InverseSpectrogram(win_length=int(cfg['win_length']*cfg['sr']*1e-3),
                                            hop_length=int(cfg['hop_length']*cfg['sr']*1e-3))

        print(f"Initialized Cross-Scale Vector Quantized Model: MAX_Bitrate is {0.001*self.bitrate} kbps")

    def encode(self, input):
        ''' input: (N,2,F,T) '''
        encoder_hs, encoder_out = self.encoder(input)

        return encoder_hs, encoder_out
    
    def decode(self, encoder_hs, encoder_out, target_Bs):

        recon_feat, vq_loss = self.decoder(encoder_hs, encoder_out, target_Bs)

        return recon_feat, vq_loss

    def train_one_batch(self, input, target_Bs=None):
        '''
        input:       {'audio': (N, L), 'stft_feat': (N, F, T, 2), 'speaker_id':(N,1)}
        target_Bs: target bitstream to achieve (1 <= Bs <= len(ch_mult))
        recon_feat:  (N, 2, F, T) 
        recon_audio: (N, L)
        '''
        raw_feat = input['stft_feat'].permute(0,3,1,2)

        encoder_hs, encoder_out = self.encode(raw_feat)
        recon_feat, vq_loss = self.decode(encoder_hs, encoder_out, target_Bs)     # (N, 2, F, T) reconstruct feature
        
        recon_feat_complex = feat2spec(recon_feat) # (N, F, T) reconstruct complex stft feature
        recon_audio = self.ift(recon_feat_complex) # (N, L)    reconstruct audio 

        ## Calculate Loss 
        # recon_feat = self.ft(self.ift(recon_feat_complex)) # (N, F, T) # To maintain stft consistency https://arxiv.org/pdf/1811.08521.pdf
        # recon_feat = torch.view_as_real(recon_feat).permute(0,3,1,2) # (N, 2, F, T)

        if self.power_law != 1:
            c_raw, c_recon = power_law_compress(input['audio'], power=self.power_law), power_law_compress(recon_audio, power=self.power_law)
            recon_loss = self.recon_loss(torch.view_as_real(self.ft(c_raw)), torch.view_as_real(self.ft(c_recon)))
        else:
            recon_loss = self.recon_loss(raw_feat, recon_feat)

        mel_loss = self.mel_loss(input['audio'], recon_audio)

        loss = recon_loss + self.vq_factor * vq_loss + self.mel_factor * mel_loss

        output = {
                  'loss': loss,
                  'recon_loss': recon_loss, 'vq_loss': vq_loss, 'mel_loss': mel_loss,
                  'recon_feat': recon_feat, 'recon_audio': recon_audio
                 }
        
        return output

    def test_one_batch(self, input, target_Bs=None):

        raw_feat = input['stft_feat'].permute(0,3,1,2)

        encoder_hs, encoder_out = self.encode(raw_feat)
        recon_feat, _ = self.decode(encoder_hs, encoder_out, target_Bs)

        recon_audio = self.ift(feat2spec(recon_feat)) # reconstruct audio 

        output = {'recon_feat': recon_feat, 'recon_audio': recon_audio }

        return output

    def forward(self, input, train=True, target_Bs=None):
        if train:
            output = self.train_one_batch(input, target_Bs)
        else:
            output = self.test_one_batch(input, target_Bs)
        
        return output



class CSVQ_Encoder(nn.Module):
    def __init__(self, in_channels=2, ch_mult=(8,8,12,12,16,32), TCM_Dilation=(1,2,4,8), num_RNN_layers=4):
        super().__init__()
        self.Bitstreams = len(ch_mult)
        in_ch_mult = (1,)+tuple(ch_mult)
        
        self.conv_in = F_downsample(in_channels*in_ch_mult[0], in_channels*ch_mult[0], stride=(1,1))
        self.down = nn.ModuleList()

        for i in range(1, self.Bitstreams):
            block_in, block_out = in_channels*in_ch_mult[i], in_channels*ch_mult[i]
            self.down.append(F_downsample(block_in,block_out))
        
        self.temporal_filter = TCM(in_channels=cfg['f_dims'][-1]*ch_mult[-1]*in_channels,
                                    fix_channels=cfg['f_dims'][-1]*ch_mult[-1],
                                    dilations=TCM_Dilation)

        self.num_RNN_layers = num_RNN_layers
        self.RNN = nn.GRU(input_size=cfg['f_dims'][-1]*ch_mult[-1]*in_channels, 
                            hidden_size=cfg['f_dims'][-1]*ch_mult[-1]*in_channels, 
                            num_layers=num_RNN_layers, batch_first=True)
        
    def forward(self, input):
        """
            input: (N, C=2, F, T)
            hs: [E5, E4, E3, E2, E1]
            encoder_out: (N, C*F, T)
        """

        h_0 = self.conv_in(input) # (N, C=16, F, T)

        hs = [self.down[0](h_0)]
        for i in range(1, len(self.down)):
            hs.append(self.down[i](hs[-1]))
        
        encoder_out = self.temporal_filter(fold(hs[-1])) # (N, C, F, T) -> (N, C*F, T)

        h_0 = torch.randn(self.num_RNN_layers, encoder_out.size(0), encoder_out.size(1), device=cfg['device'])

        encoder_out, _ = self.RNN(encoder_out.permute(0,2,1), h_0) # (N, T, C*F)

        return hs, encoder_out.permute(0,2,1)


class CSVQ_Decoder(nn.Module):
    def __init__(self, in_channels=2, ch_mult=(8,8,12,12,16,32), num_groups=6, TCM_Dilation=(1,2,4,8), num_RNN_layers=4,
                    codebook_size=1024, vq_commit=0.25, down_rate=0.4):
        super().__init__()
        self.Bitstreams = len(ch_mult)
        out_ch_mult = (1,)+tuple(ch_mult)

        self.conv_out = F_upsample(in_channels*ch_mult[0], in_channels*out_ch_mult[0],stride=(1,1))
        self.up = nn.ModuleList()
        
        for i in reversed(range(1, self.Bitstreams)):
            block_in, block_out = in_channels*ch_mult[i], in_channels*out_ch_mult[i]
            if i == self.Bitstreams-2: self.up.append(F_upsample(block_in,block_out,output_pad=(1,0)))
            else: self.up.append(F_upsample(block_in,block_out))

        f_dims = cfg['f_dims']

        c_fix_0 = find_nearest(int(in_channels*ch_mult[-1] * f_dims[-1] * down_rate), num_groups)

        self.down_0 = nn.Conv1d(in_channels*ch_mult[-1] * f_dims[-1], c_fix_0, 1, 1)
        self.up_0 = nn.Conv1d(c_fix_0, in_channels*ch_mult[-1] * f_dims[-1], 1, 1)

        self.quantizer_0 = Group_Quantize(num_groups=num_groups, K=4*c_fix_0//num_groups, codebook_size=codebook_size, vq_commit=vq_commit)

        self.temporal_filter1 = TCM(in_channels=cfg['f_dims'][-1]*ch_mult[-1]*in_channels,
                                    fix_channels=cfg['f_dims'][-1]*ch_mult[-1],
                                    dilations=TCM_Dilation)

        self.num_RNN_layers = num_RNN_layers
        self.RNN = nn.GRU(input_size=cfg['f_dims'][-1]*ch_mult[-1]*in_channels, 
                            hidden_size=cfg['f_dims'][-1]*ch_mult[-1]*in_channels, 
                            num_layers=num_RNN_layers, batch_first=True)

        self.temporal_filter2 = TCM(in_channels=cfg['f_dims'][-1]*ch_mult[-1]*in_channels,
                                    fix_channels=cfg['f_dims'][-1]*ch_mult[-1],
                                    dilations=TCM_Dilation)

        self.fuse_vq_refine_module = nn.ModuleList([Fuse_VQ(num_groups=num_groups, input_channel=in_channels*ch_mult[self.Bitstreams-i-1], F_dim=f_dims[self.Bitstreams-i-1], 
                                    down_rate=down_rate, codebook_size=codebook_size, vq_commit=vq_commit) for i in range(self.Bitstreams-1)])
        
    def forward(self, encoder_hs, encoder_out, target_Bs):
        ''' 
          encoder_hs:  [E5,E4,E3,E2,E1]
          encoder_out: [N, C*F, T]
          target_Bs:   int N, the target bitstream to achieve (1 <= N <= Bitstreams)
        '''

        if target_Bs is None: target_Bs = self.Bitstreams    # non-scalable
        assert target_Bs <= self.Bitstreams and target_Bs >= 1

        # Group Quantization for Q_0 (no fuse)  
        z_q0, vq_loss = self.quantizer_0(self.down_0(encoder_out))  # (N, C*F, T) -> (N, c_fix_0, T)
        z_q0 = self.up_0(z_q0)                                      # (N, c_fix_0, T) -> (N, C*F, T)

        # Temporal Filtering Module
        z_q0 = self.temporal_filter1(z_q0)
        h_0 = torch.randn(self.num_RNN_layers, z_q0.size(0), z_q0.size(1), device=cfg['device'])
        z_q0, _ = self.RNN(z_q0.permute(0,2,1), h_0)             
        z_q0 = unfold(self.temporal_filter1(z_q0.permute(0,2,1)),cfg['csvq_codec']['ch_mult'][-1]*2)   # (N, C, F, T)

        # Stepwise Fuse-VQ Decoding
        F_dec = [z_q0]

        for i in range(self.Bitstreams-1): # i: 0 -> 4
            transmit = i < target_Bs-1

            F_dec_refined, vq_loss_i = self.fuse_vq_refine_module[i](encoder_hs[-i-1], F_dec[i], transmit=transmit) 
            vq_loss += vq_loss_i
    
            z_qi = self.up[i](F_dec_refined)
            F_dec.append(z_qi)
            # print(i, F_dec_refined.shape, self.up[i], z_qi.shape)

        decoder_out = self.conv_out(F_dec[-1])

        return decoder_out, vq_loss


def init_model():
    ch_mult = cfg['csvq_codec']['ch_mult']
    num_groups = cfg['csvq_codec']['num_groups']
    TCM_Dilation = cfg['csvq_codec']['TCM_Dilation']
    down_rate = cfg['csvq_codec']['down_rate']
    num_RNN_layers = cfg['csvq_codec']['num_RNN_layers']
    codebook_size = cfg['csvq_codec']['codebook_size']
    vq_commit = cfg['csvq_codec']['vq_commit']
    mel_factor = cfg['csvq_codec']['mel_factor']
    vq_factor = cfg['csvq_codec']['vq_factor']
    
    
    model = CSVQ_TFNet(in_channels=2, ch_mult=ch_mult, num_groups=num_groups, TCM_Dilation=TCM_Dilation,
                       num_RNN_layers=num_RNN_layers, codebook_size=codebook_size, vq_commit=vq_commit, down_rate=down_rate, 
                        mel_factor=mel_factor, vq_factor=vq_factor, power_law_ratio=cfg['csvq_codec']['power_law'], 
                        n_fft=2048, n_mels=cfg['csvq_codec']['n_mels'])

    return model





