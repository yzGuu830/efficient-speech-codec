import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import isdct_torch

def init_param(m):
    if isinstance(m, (nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, (nn.LayerNorm)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    return m

def denormalize(input,stats):
    broadcast_size = [1] * input.dim()
    # broadcast_size[1] = input.size(1)
    m, s = stats[0].item(), stats[1].item()
    m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
           torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
    input = input.mul(s).add(m)
    return input

def InversePowerSpectrogram(x, sign): # x: power spectrogram
    magnitude = torch.pow(torch.pow(10.0, 0.1 * x), 0.5)
    return magnitude*sign

def reconstruct_audio(X, input, frame_step=60): # x: batch of power spectrogram feature
    N,C,H,W = X.shape
    res = []
    if 'power_stats' in input: # Compose1
        for i in range(N):
            denormalize_power = denormalize(X.detach()[i],input['power_stats'][i]) # denormalize1
            normalized_sdct = InversePowerSpectrogram(denormalize_power,input['sign'][i]) # inverse powerspectrogram
            denormalized_sdct = denormalize(normalized_sdct,input['sdct_stats'][i]) # denormalize2
            res.append(isdct_torch(denormalized_sdct,frame_step=frame_step))
    elif 'sdct_stats' in input and 'power_stats' not in input: # Compose2
        for i in range(N):
            normalized_sdct = InversePowerSpectrogram(X.detach()[i],input['sign'][i]) # inverse powerspectrogram
            denormalized_sdct = denormalize(normalized_sdct,input['sdct_stats'][i]) # denormalize2
            res.append(isdct_torch(denormalized_sdct,frame_step=frame_step))
    elif 'sdct_stats' not in input and 'power_stats' not in input: # Compose3
        for i in range(N):
            denormalized_sdct = InversePowerSpectrogram(X.detach()[i],input['sign'][i]) # inverse powerspectrogram
            res.append(isdct_torch(denormalized_sdct,frame_step=frame_step))
    return res
