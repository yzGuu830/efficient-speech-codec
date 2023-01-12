from data import fetch_dataset, make_data_loader
from utils import process_control,process_dataset
from config import cfg, process_args
from models import vqvae
import torch
from utils import collate
import numpy as np

process_control()
cfg['seed'] = 0


# if __name__ == '__main__':
    # dataset = fetch_dataset(cfg['data_name'])
    # process_dataset(dataset)
    # data_loader = make_data_loader(dataset, cfg['model_name'])
    # print("data_loader loading finish!")
    # print(f"train_data:{len(data_loader['train'])} test_data:{len(data_loader['test'])}")

    # model = vqvae.vqvae()
    # for i,input in enumerate(data_loader['train']):
    #     input = collate(input)
        # print(np.any(np.isnan(input['feature'].cpu().numpy())),np.any(np.isnan(input['sdct_feat'].cpu().numpy())))
        # if np.any(np.isnan(input['feature'].cpu().numpy())) == True:
        #     break
        # print()
        # input = collate(input)
        # print(input.keys())
        # print(input['feature'][0])
        # print(input['sign'][0])
        # print(input['stats'][0])

        # output = model(input)

        # print(output['loss'])
        # print(output['recon_feature'][0])
        # print(output['recon_audio'][0].shape)
        # break



# if __name__ == '__main__':
#     print(torch.__version__)
#     from models.quantizer import Quantizer
#     from models.convolutional_encoder import ConvEncoder
#     from models.convolutional_decoder import ConvDecoder
#     x = torch.randn(4,1,120,132)

#     encoder = ConvEncoder(in_channels=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=64, 
#                     hierarchy_layer=1, out_channels=64, use_kaiming_normal=False, verbose=True)
#     quantizer = Quantizer(embedding_size=64, num_embedding=256, vq_commit=0.25)
#     decoder = ConvDecoder(in_channels=64, out_channels=1, num_hiddens=128, num_residual_layers=2,
#                     hierarchy_layer=1, num_residual_hiddens=64, use_kaiming_normal=False, verbose=True)

#     z = encoder(x)
#     print(z.shape)
#     q, diff, embedding_ind = quantizer(z)
#     print(q.shape)
#     x_ = decoder(q)
#     print(x_.shape)

# if __name__ == "__main__":
    # from models.vqgan_model import Encoder, Decoder
    # from models.quantizer import Quantizer
    # x = torch.randn(4,1,120,132)
    # ddconfig={
    #   "double_z": False,
    #   "z_channels": 128,
    #   "resolution": 128,
    #   "in_channels": 1,
    #   "out_ch": 1,
    #   "ch": 128,
    #   "ch_mult": [1,2,4],  # num_down = len(ch_mult)-1
    #   "num_res_blocks": 2,
    #   "attn_resolutions": [16],
    #   "dropout": 0.0
    # }
    # encoder = Encoder(**ddconfig)
    # decoder = Decoder(**ddconfig)
    # quantizer = Quantizer(embedding_size=128, num_embedding=256, vq_commit=0.25)

    # z = encoder(x)
    # print(z.shape)
    # q, diff, embedding_ind = quantizer(z)
    # print(q.shape)
    # x_ = decoder(q)
    # print(x_.shape)
    
if __name__ == "__main__":
    model = vqvae.vqvae()
    x = torch.randn(4,1,120,132)

    z = model.encoder(x)
    print(z.shape)
    q, diff, embedding_ind = model.quantizer(z)
    print(q.shape)
    x_ = model.decoder(q)
    print(x_.shape)