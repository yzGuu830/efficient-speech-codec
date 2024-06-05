import torch
import torch.nn as nn

from models.esc import BaseAudioCodec, CrossScaleRVQ, SwinTEncoder


class RVQCodecs(BaseAudioCodec):
    """
    Frequency Codec with conv/swinT backbones along with RVQs
    Experiment: RVQ + CNN / RVQ + swinT
    """
    def __init__(self, in_dim: int, in_freq: int, h_dims: list, max_streams: int, win_len: int = 20, hop_len: int = 5, sr: int = 16000) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, win_len, hop_len, sr)

        self.quantizers = self.init_residual_vqs()
        self.encoder = ConvEncoder # SwinTEncoder
        self.decoder = ConvDecoder # SwinTDecoder
        

class CSVQConvCodec(BaseAudioCodec):
    """
    Frequency Codec with conv backbones along with CSVQ
    Experiment: CSVQ + CNN 
    """
    def __init__(self, in_dim: int, in_freq: int, h_dims: list, max_streams: int, win_len: int = 20, hop_len: int = 5, sr: int = 16000) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, win_len, hop_len, sr)

        self.quantizers = self.init_product_vqs()
        self.encoder = ConvEncoder
        self.decoder = CrossScaleConvDecoder


class ConvEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class ConvDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class CrossScaleConvDecoder(CrossScaleRVQ):
    def __init__(self, backbone="conv") -> None:
        super().__init__(backbone)


class SwinTDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()