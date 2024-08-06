import torch
import torch.nn as nn
from typing import Literal

from ..modules import ProductVectorQuantize, TransformerLayer, PatchDeEmbed, ConvolutionLayer, Convolution2D
from .utils import blk_func

class CrossScaleRVQ(nn.Module):
    """Cross-Scale Residual Vector Quantization Framework"""
    def __init__(self, backbone: Literal['transformer', 'convolution']="transformer") -> None:
        super().__init__()
        if backbone == "transformer": self.dims = 3
        elif backbone == "convolution": self.dims = 4
    
    def pre_fuse(self, enc, dec):
        """Compute residuals to quantize"""
        return enc - dec
    
    def post_fuse(self, residual_q, dec):
        """Add back quantized residuals"""
        return residual_q + dec

    def csrvq(self, enc: torch.tensor, dec: torch.tensor, vq: ProductVectorQuantize, 
            transmit: bool=True, freeze_vq: bool=False):
        """ Forward Function combining encoding and decoding at a single bitstream/resolution scale
        Args:
            enc (Tensor): Tensor of encoded feature with shape (B, H*W, C) / (B, C, H, W) 
            dec (Tensor): Tensor of decoded feature with shape (B, H*W, C) / (B, C, H, W)
            vq (ProductVectorQuantize): product quantizer at this stream level
            transmit (Boolean): whether this stream is transmitted (perform quantization or not)
            freeze_vq (Boolean): whether freeze the codebook (in a pre-training stage)
        Returns: 
            Tensor of dec_refine (decoded feature conditioned on quantized encodings)
        """
        if not self.training and not transmit:
            return dec, 0., 0., None

        residual = self.pre_fuse(enc, dec)
        outputs = vq(residual, freeze_vq)
        residual_q, code = outputs["z_q"], outputs["codes"]
        cm_loss, cb_loss = outputs["cm_loss"], outputs["cb_loss"]

        if not transmit: # masking non-transmitted streams
            cm_loss, cb_loss = cm_loss * 0., cb_loss * 0.
            residual_q *= 0.

        dec_refine = self.post_fuse(residual_q, dec)
        return dec_refine, cm_loss, cb_loss, code
    
    def csrvq_encode(self, enc, dec, vq):

        residual = self.pre_fuse(enc, dec)
        code = vq.encode(residual)
        return code
    
    def csrvq_decode(self, codes, dec, vq):

        residual_q = vq.decode(codes, self.dims)
        dec_refine = self.post_fuse(residual_q, dec)
        return dec_refine
    

class CrossScaleRVQDecoder(CrossScaleRVQ):
    def __init__(self, 
                 backbone: Literal['transformer', 'convolution'],
                 in_freq: int, 
                 in_dim: int, 
                 h_dims: list,
                 patch_size: tuple,
                 kernel_size: list=[],
                 conv_depth: int=1,
                 swin_heads: list=[],
                 swin_depth: int=2,
                 window_size: int=4,
                 mlp_ratio: float=4.,) -> None:
        super().__init__(backbone)

        in_dims, out_dims = h_dims[:-1], h_dims[1:]

        blocks = nn.ModuleList()
        for i in range(len(in_dims)):
            layer = ConvolutionLayer(in_dims[i], out_dims[i], conv_depth, kernel_size, transpose=True) if backbone == "convolution" \
               else TransformerLayer(
                    in_dims[i], out_dims[i], swin_heads[i], swin_depth, window_size, mlp_ratio, 
                    activation=nn.GELU, norm_layer=nn.LayerNorm, scale="up", scale_factor=(2,1)
                )
            blocks.append(layer)
            
        self.patch_deembed = PatchDeEmbed(in_freq, in_dim, patch_size, h_dims[-1], backbone)
        self.post_nn = Convolution2D(h_dims[-1], h_dims[-1], kernel_size, scale=False) if backbone == "convolution" \
                  else TransformerLayer(
                       h_dims[-1], h_dims[-1], swin_heads[-1], swin_depth, window_size, mlp_ratio, 
                       activation=nn.GELU, norm_layer=nn.LayerNorm, scale=None
                    )
        self.blocks = blocks

    def forward(self, enc_hs: list, num_streams: int, quantizers: nn.ModuleList, feat_shape: tuple, freeze_vq: bool=False):
        """Forward Function: step-wise cross-scale decoding
        Args: 
            enc_hs (List[Tensor, ...]): a list of encoded features at all scales
            num_streams (int): number of bitstreams to use (max_streams when freeze_vq is True)
            quantizers (ModuleList): a modulelist of multi-scale quantizers
            feat_shape (Tuple): (Wh, Ww) feature shape at bottom level
            freeze_vq (Boolean): freeze vq layers during pre-training
        Returns: 
            recon_feat: reconstructed complex spectrum (Bs, 2, F, T)
            codes: discrete indices (Bs, num_streams, group_size, T//overlap) 
                    num_streams is always max_stream in training mode
            cm_loss, cb_loss: VQ losses (Bs, )
        """
        z0, cm_loss, cb_loss, code = self.csrvq(enc=enc_hs[-1], dec=0.0, vq=quantizers[0], 
                                                transmit=True, freeze_vq=freeze_vq)
        codes, dec_hs = [code], [z0]
        for i, blk in enumerate(self.blocks):
            dec_i_refine, cm_loss_i, cb_loss_i, code_i = self.csrvq(
                enc=enc_hs[-1-i], dec=dec_hs[i], vq=quantizers[i+1], transmit=(i<num_streams-1), freeze_vq=freeze_vq
                        )
            cm_loss += cm_loss_i
            cb_loss += cb_loss_i
            if code_i is not None:
                codes.append(code_i)
            
            dec_next, feat_shape = blk_func(blk, dec_i_refine, feat_shape)
            dec_hs.append(dec_next)

        dec_next, feat_shape = blk_func(self.post_nn, dec_next, feat_shape)
        recon_feat = self.patch_deembed(dec_next)
        codes = torch.stack(codes, dim=1)
        return recon_feat, codes, cm_loss, cb_loss
    
    def encode(self, enc_hs: list, num_streams: int, quantizers: nn.ModuleList, feat_shape: tuple):
        """Encode audio into indices
        Args: 
            enc_hs (List[Tensor, ...]): a list of encoded features at all scales
            num_streams (int): number of bitstreams to use
            quantizers (ModuleList): a modulelist of multi-scale quantizers
            feat_shape (Tuple): (Wh, Ww) feature shape at bottom level
        Returns: 
            multi-scale codes with shape (Bs, num_streams, group_size, T)
        """
        code0 = quantizers[0].encode(enc_hs[-1]) # [B, group_size, T]
        if num_streams == 1:
            return code0.unsqueeze(1)
        
        z0 = quantizers[0].decode(code0, dims=self.dims)
        codes, dec_hs = [code0], [z0]
        for i in range(num_streams-1):
            
            codei = self.csrvq_encode(enc=enc_hs[-1-i], dec=dec_hs[i], vq=quantizers[i+1])
            codes.append(codei)
            if len(codes) == num_streams: break

            dec_i_refine = self.csrvq_decode(codei, dec=dec_hs[i], vq=quantizers[i+1])
            dec_next, feat_shape = blk_func(self.blocks[i], dec_i_refine, feat_shape)
            dec_hs.append(dec_next)
        
        codes = torch.stack(codes, dim=1) 
        return codes    # [B, num_streams, group_size, T]
    
    def decode(self, codes: list, quantizers: nn.ModuleList, feat_shape: tuple):
        """Decode from indices
        Args: 
            codes (Tensor): multi-scale codes with shape (B, num_streams, group_size, T)
            quantizers (ModuleList): a modulelist of multi-scale quantizers
            feat_shape (Tuple): (Wh, Ww) feature shape at bottom level
        Returns: 
            decoded hidden states: List of decoded features
        """
        num_streams = codes.size(1)

        z0 = quantizers[0].decode(codes[:, 0], dims=self.dims)
        dec_hs = [z0]
        for i in range(len(self.blocks)): # using code of residuals to refine decoding
            if i < num_streams-1:
                dec_i_refine = self.csrvq_decode(codes=codes[:, i+1], dec=dec_hs[i], vq=quantizers[i+1])
            else:
                dec_i_refine = dec_hs[i]
            dec_next, feat_shape = blk_func(self.blocks[i], dec_i_refine, feat_shape)
            dec_hs.append(dec_next)

        dec_next, feat_shape = blk_func(self.post_nn, dec_next, feat_shape)
        dec_hs.append(self.patch_deembed(dec_next))
        return dec_hs