from models.codecs import RVQCodecs
from models.esc import ESC
import torch


model = RVQCodecs(in_dim=2, in_freq=192, h_dims=[45,72,96,144,192,384], #[64,64,96,96,192,384], # [16,16,24,24,32,64]
                max_streams=6, backbone="swinT", # swinT
                overlap=2, num_rvqs=6, group_size=3, codebook_dim=8, codebook_size=1024, l2norm=True,
                conv_depth=1)

# model = ESC(group_size=3)

# model = RVQCodecs(in_dim=2, in_freq=192, h_dims=[45,72,96,144,192,384], #[64,64,96,96,192,384], # [16,16,24,24,32,64]
#                 max_streams=6, backbone="conv", # swinT
#                 overlap=4, num_rvqs=6, group_size=3, codebook_dim=8, codebook_size=1024, l2norm=True,
#                 conv_depth=1)
# print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"   Model #Parameters: {n_params/1000000:.2f}M")

if __name__ == "__main__":

    x = torch.randn(2, 47920)
    # model.eval()
    outputs = model.forward_one_step(
        x, None, num_streams=6, freeze_codebook=True
    )

    print(outputs["cm_loss"], outputs["cb_loss"])
    print(outputs["recon_feat"].shape, outputs["recon_audio"].shape, outputs["codes"].shape)