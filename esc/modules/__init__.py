from .transformer.attention import TransformerLayer
from .transformer.scale import PatchEmbed, PatchDeEmbed

from .vq.quantization import ProductVectorQuantize, ResidualVectorQuantize, ProductResidualVectorQuantize
from .loss.generator_loss import MelSpectrogramLoss, ComplexSTFTLoss
from .convolution.layers import ConvolutionLayer, Convolution2D