"""
Variational AutoEncoder framework
# Author: cain@cinnamon.is
"""


import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    classifier module: MLP network
    """

    def __init__(self, input_size=784,
                         output_size=10,
                         hidden_size=512):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),

            nn.Linear(self.hidden_size, self.output_size),
            nn.Softmax(dim=1)
        )

    def _classify(self, x):
        return self.classifier(x.view(-1, self.input_size))

    def _predict(self, x):
        alpha = self._classify(x)
        return torch.distributions.one_hot_categorical.OneHotCategorical(alpha).sample()

    def forward(self, x):
        return self._predict(x), self._classify(x)


class Encoder(nn.Module):
    """
    encoder module: MLP network
    """
    def __init__(self, input_size=784,
                        output_size=10,
                        latent_size=2,
                        hidden_size=512,
                        use_cuda=False):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )

        self.classifier = Classifier(input_size=self.input_size,
                                     output_size=self.output_size,
                                     hidden_size=512)

        self.mean = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar= nn.Linear(self.hidden_size, self.latent_size)

    def _sampling(self, latent_dist):
        mean = latent_dist["mean"]
        std = latent_dist["logvar"].mul(0.5).exp()
        if self.training:
            eps = torch.randn_like(std)
            if self.use_cuda:
                eps = eps.cuda()
            return eps.mul(std).add_(mean)
        else:
            return mean

    def forward(self, x, y=None):
        if y is None:
            y, _ = self.classifier(x)

        h = self.encoder(x.view(-1, self.input_size))
        latent_dist = {"mean": self.mean(h),
                       "logvar": self.logvar(h)}

        enc_output = torch.cat([self._sampling(latent_dist), y.view(-1, self.output_size)], dim=1)
        return enc_output, latent_dist

class Decoder(nn.Module):
    """
    decoder module: MLP network
    """
    def __init__(self, input_size=784,
                        latent_size=2,
                        hidden_size=512):
        super(Decoder,self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z.view(-1, self.latent_size))