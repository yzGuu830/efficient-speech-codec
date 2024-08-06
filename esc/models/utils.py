from ..modules import TransformerLayer, ConvolutionLayer, Convolution2D

def blk_func(blk, feat, feat_shape):
    Wh, Ww = feat_shape
    if isinstance(blk, TransformerLayer): 
        feat_next, Wh, Ww = blk(feat, Wh, Ww)
    elif isinstance(blk, ConvolutionLayer):
        feat_next = blk(feat)
        Wh, Ww = Wh//2, Ww
    elif isinstance(blk, Convolution2D):
        feat_next = blk(feat)
    
    return feat_next, (Wh, Ww)