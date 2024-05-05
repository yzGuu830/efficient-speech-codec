import torch
import torch.nn as nn

from models.swin.wattn import SwinTransformerLayer
from models.swin.compress import PatchDeEmbed

class BaseCrossScaleDecoder(nn.Module):
    def __init__(self, in_H: int, in_dim: int, h_dims: list, max_streams: int, fuse_net: str,) -> None:
        super().__init__()

        self.out_h_dims = h_dims[1:] + [in_dim]
        self.h_dims = h_dims

        self.max_streams = max_streams
        
        self.pre_fuse, self.post_fuse = self.res_pre_fuse, self.res_post_fuse
    
    def res_pre_fuse(self, enc, dec, idx=None, pre_fuse_net=None):
        return pre_fuse_net[idx](enc - dec)
    
    def res_post_fuse(self, residual_q, dec, idx=None, post_fuse_net=None, transmit=True):
        if not transmit: 
            mask = torch.full((dec.shape[0],), fill_value=False, device=dec.device)
            residual_q *= mask[:, None, None]
        return post_fuse_net[idx](residual_q + dec)

    def csvq_layer(self, 
                   enc: torch.tensor, 
                   dec: torch.tensor, 
                   idx: int, 
                   vq: nn.Module, 
                   pre_fuse_net: nn.ModuleList = None, 
                   post_fuse_net: nn.ModuleList = None, 
                   transmit: bool=True,
                   alpha: float=1.0):
        # Quantization Forward that combines quantize and dequantize
        residual = self.pre_fuse(enc, dec, idx, pre_fuse_net)
        residual_q, cm_loss, cb_loss, kl_loss = vq(residual)

        # alpha: fade-in parameter for vq in progressive training ranged in [0.0, 1.0]
        residual_q *= alpha
        cm_loss, cb_loss, kl_loss = cm_loss*alpha, cb_loss*alpha, kl_loss*alpha

        dec_refine = self.post_fuse(residual_q, dec, idx, post_fuse_net, transmit)

        if not transmit:
            mask = torch.full((dec.shape[0],), fill_value=False, device=dec.device)
            cm_loss *= mask
            cb_loss *= mask
            kl_loss *= mask

        return dec_refine, cm_loss, cb_loss, kl_loss
    
    def csvq_quantize(self, enc, dec, idx, vq, pre_fuse_net):

        residual = self.pre_fuse(enc, dec, idx, pre_fuse_net)
        codes = vq.encode(residual)
        return codes
    
    def csvq_dequantize(self, codes, dec, idx, vq, post_fuse_net):

        residual_q = vq.decode(codes, dim=3) # dim=3 for transformer / dim=4 for convolution
        dec_refine = self.post_fuse(residual_q, dec, idx, post_fuse_net)
        return dec_refine


class ProgressiveCrossScaleDecoder(BaseCrossScaleDecoder):
    def __init__(self, 
                 swin_depth: int = 2,
                 swin_heads: list = [3],
                 window_size: int = 4,
                 mlp_ratio: float = 4.,
                 in_freq: int = 192, 
                 patch_size: list = [3,2], 
                 in_dim: int = 2, 
                 h_dims: list = [384,192,96,72,45,45], 
                 is_causal: bool = False,
                 max_streams: int = 6, ) -> None:
        super().__init__(in_freq//patch_size[0], in_dim, h_dims, max_streams, "None",)

        self.patch_deembed = PatchDeEmbed(in_freq, patch_size, in_dim, h_dims[-1])
        self.h_dims = self.h_dims[:-1]
        self.out_h_dims = self.out_h_dims[:-1]
        self.blocks = self.init_decoder(swin_depth, swin_heads, window_size, mlp_ratio, is_causal)

        pre_fuse_net = nn.ModuleList([nn.Identity() for _ in range(max_streams-1)])
        post_fuse_net = nn.ModuleList([nn.Identity() for _ in range(max_streams-1)])
        self.pre_fuse_net, self.post_fuse_net = pre_fuse_net, post_fuse_net

        self.post_swin = SwinTransformerLayer(
                    self.out_h_dims[-1], self.out_h_dims[-1],
                    depth=swin_depth, num_heads=swin_heads[0],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    subsample=None, is_causal=is_causal
        )# tag

    def decode(self, enc_hs: list, streams: int, vqs: nn.ModuleList, Wh: int, Ww: int, alpha: float):
        """Step-wise Fuse decoding (Combines Quantize and Dequantize for Forward Training)
        Args: 
            enc_hs: a list of encoded features at multiple scale
            streams: number of bitstreams to use <= depth + 1
            vqs: a modulelist of quantizers with size $depth$
            Wh, Ww: encoder last feature size
            alpha: fade-in hyperparameter in progressive training
        """
        assert streams <= self.max_streams and len(vqs) == self.max_streams

        z0, cm_loss, cb_loss, kl_loss = vqs[0](enc_hs[-1])
        
        # stream==0 stands for quantizer fix stages
        if streams == 0:    z0, cm_loss, cb_loss, kl_loss = enc_hs[-1] + z0*0.0, cm_loss*0.0, cb_loss*0.0, kl_loss*0.0

        dec_hs = [z0]
        for i, blk in enumerate(self.blocks):
            transmit = (i < streams-1) 
                
            if self.training: 
                # during training forward pass all quantizers and remove those not transmitted by zero mask
                # this is for multi-gpu training purpose
                dec_i_refine, cm_loss_i, cb_loss_i, kl_loss_i = self.csvq_layer(
                                                        enc=enc_hs[-1-i], dec=dec_hs[i],
                                                        idx=i, vq=vqs[i+1], 
                                                        pre_fuse_net=self.pre_fuse_net,
                                                        post_fuse_net=self.post_fuse_net,
                                                        transmit=transmit, alpha=alpha)
                cm_loss += cm_loss_i
                cb_loss += cb_loss_i
                kl_loss += kl_loss_i
            else:
                # during inference forward pass only transmitted quantizers
                if transmit:
                    dec_i_refine, cm_loss_i, cb_loss_i, kl_loss_i = self.csvq_layer(
                                                            enc=enc_hs[-1-i], dec=dec_hs[i],
                                                            idx=i, vq=vqs[i+1], 
                                                            pre_fuse_net=self.pre_fuse_net,
                                                            post_fuse_net=self.post_fuse_net, 
                                                            transmit=True, alpha=1.0)
                    cm_loss += cm_loss_i
                    cb_loss += cb_loss_i
                    kl_loss += kl_loss_i
                else:
                    dec_i_refine = dec_hs[i]
            
            # upsample [main blocks]
            dec_next, Wh, Ww = blk(dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        dec_next, Wh, Ww = self.post_swin(dec_next, Wh, Ww)
        recon_feat = self.patch_deembed(dec_next)

        return recon_feat, dec_hs, cm_loss, cb_loss, kl_loss
    
    def quantize(self, enc_hs: list, streams: int, vqs: nn.ModuleList, Wh: int, Ww: int):
        """Step-wise Compression (Quantize to code for Inference)
        Args: 
            enc_hs: a list of encoded features at multiple scale
            streams: number of bitstreams to use <= depth + 1
            vqs: a modulelist of quantizers with size $depth$
            Wh, Ww: encoder last feature size
        returns: multi-scale codes
        """
        assert streams <= self.max_streams and len(vqs) == self.max_streams

        codes0 = vqs[0].encode(enc_hs[-1])
        if streams == 1:
            return [codes0]
        
        z0 = vqs[0].decode(codes0)
        multi_codes, dec_hs = [codes0], [z0]
        for i in range(streams-1):
            
            codes_i = self.csvq_quantize(enc=enc_hs[-1-i], dec=dec_hs[i], idx=i, vq=vqs[i+1], pre_fuse_net=self.pre_fuse_net)
            multi_codes.append(codes_i)
            if len(multi_codes) == streams: 
                break
            dec_i_refine = self.csvq_dequantize(codes=codes_i, dec=dec_hs[i], idx=i, vq=vqs[i+1], post_fuse_net=self.post_fuse_net)

            dec_next, Wh, Ww = self.blocks[i](dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        return multi_codes
    
    def dequantize(self, multi_codes: list, vqs: nn.ModuleList, Wh: int, Ww: int):
        """Step-wise DeCompression (DeQuantize code for Inference)
        Args: 
            multi_codes: a list of encoded residual codes at multiple scale
            vqs: a modulelist of quantizers with size $depth$
            Wh, Ww: encoder last feature size
        returns: multi-scale codes
        """
        streams = len(multi_codes)
        assert streams <= self.max_streams and len(vqs) == self.max_streams

        z0 = vqs[0].decode(multi_codes[0])
        dec_hs = [z0]
        for i in range(streams-1): # Using code of residuals to refine decoding
            dec_i_refine = self.csvq_dequantize(codes=multi_codes[i+1], dec=dec_hs[i], idx=i, vq=vqs[i+1], post_fuse_net=self.post_fuse_net)

            dec_next, Wh, Ww = self.blocks[i](dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        dec_next, Wh, Ww = self.post_swin(dec_next, Wh, Ww)
        dec_hs.append(self.patch_deembed(dec_next))
        return dec_hs

    def init_decoder(self, depth, num_heads: list, window_size, mlp_ratio, is_causal):
        if len(num_heads) == 1:
            num_heads = num_heads[0]
            num_heads = [min(num_heads*2**(len(self.h_dims)-i-1), num_heads*2**3) for i in range(len(self.h_dims))]
        else:
            num_heads = num_heads[::-1]

        blocks = nn.ModuleList()
        for i in range(len(self.h_dims)):
            blocks.append(
                SwinTransformerLayer(
                    self.h_dims[i],
                    self.out_h_dims[i],
                    depth=depth,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    subsample="up",
                    scale_factor=[2,1],
                    is_causal=is_causal
                )
            )
        return blocks

    def vis_decoder(self):
        
        for i, blk in enumerate(self.blocks):
            print("Layer[{}]: swin_depth={} swin_hidden={} heads={} up={}".format(
                i, blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))

        blk = self.post_swin
        print("Post-swin Layer: swin_depth={} swin_hidden={} heads={} up={}".format(
                blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))