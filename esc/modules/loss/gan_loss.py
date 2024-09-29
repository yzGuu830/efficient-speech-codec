import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.

    Adapted from DAC https://github.com/descriptinc/descript-audio-codec/blob/main/
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        """
        fake/real: audio tensor of shape [batchsize, channel, len]
        """
        if fake.dim() == 2: fake = fake.unsqueeze(1)
        if real.dim() == 2: real = real.unsqueeze(1)

        d_fake = self.discriminator(**dict(x=fake))
        d_real = self.discriminator(**dict(x=real))
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2, dim=[1,2,3])
            loss_d += torch.mean((1 - x_real[-1]) ** 2, dim=[1,2,3])
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2, dim=[1,2,3])

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach(), reduction="none").mean([1,2,3])
        return loss_g, loss_feature