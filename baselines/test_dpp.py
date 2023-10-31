from models.codec import SwinCrossScaleCodec

model = SwinCrossScaleCodec(patch_size = [3,2],
                 swin_depth = 2,
                 swin_heads = 3,
                 window_size = 4,
                 mlp_ratio = 4.,
                 in_dim = 2, 
                 in_freq = 192, 
                 h_dims = [36,72,144,144,192,384], 
                 max_streams = 6, 
                 proj = 4, 
                 overlap = 2, 
                 num_vqs = 6, 
                 codebook_size = 1024, 
                 mel_nfft = 2048, 
                 mel_bins = 64, 
                 vq_commit = 1., 
                 fuse_net = False, 
                 scalable = False, )

import torch

model = torch.nn.DataParallel(model, 
                              device_ids=[i for i in range(4)])
model = model.cuda()

audio = torch.rand(12, 47920).cuda()
# x = torch.randn(12, 2, 192, 600)
outputs = model(**dict(x=audio, x_feat=None, streams=6, train=True))

print(outputs['loss'])
print(outputs["mel_loss"])
print(outputs["vq_loss"])
print(outputs['recon_loss'])