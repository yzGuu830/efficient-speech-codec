import torch
import torch.nn as nn
from models.residual import ResBlock



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_res_block, res_size, stride):
        super().__init__()
        if stride == 8:
            blocks = [
                nn.Conv2d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm2d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size // 2, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm2d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size // 2, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif stride == 4:
            blocks = [
                nn.Conv2d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm2d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size // 2, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv2d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm2d(hidden_size // 2),

                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size // 2, hidden_size, 3, 1, 1),
            ]
        else:
            raise ValueError('Not valid stride')
        for _ in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))
        blocks.extend([
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, output_size, 1, 1, 0)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)




if __name__ == "__main__":
    # Test

    x = np.random.random_sample((3,40,40,200))
    x = torch.tensor(x).float()

    encoder = Encoder(in_dim=40,h_dim=128,n_res_layers=3,res_h_dim=64)
    Z_e = encoder(x)

    print(f'Encoder output shape: {Z_e.shape}')
