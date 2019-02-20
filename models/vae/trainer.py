import torch
import sys
sys.path.insert(0, "../../utils")
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from utils import printr

class Trainer():
    def __init__(self, model, optimizer,
                        use_cuda=False):
        self.model = model
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        if self.use_cuda:
            for _,mdl in self.model.items():
                mdl.cuda()

        ## observation
        self.writer = SummaryWriter()

    def train(self, data_loaders, epochs=10):
        for epoch in range(epochs):
            ## STARTING EPOCH
            print("Epoch {}:\n".format(epoch+1))
            str_print = ""
            for _, mdl in self.model.items():  ## put everything in train mode
                mdl.train()
            ## training
            losses = self._train_epoch(data_loaders)

            for name, value in losses.items():
                str_print += "{}: {}\n".format(name, value)

            ## verifying accuracy
            valid_acc = self._accuracy_epoch(data_loaders["valid"])
            test_acc = self._accuracy_epoch(data_loaders["test"])
            self.writer.add_scalars('data/evol_acc',
                                    {'valid_acc': valid_acc,
                                     'test_acc': test_acc},
                                        epoch + 1)
            str_print += "valid acc: {}\n".format(valid_acc)
            str_print += "test acc: {}\n".format(test_acc)
            ## print screen
            print(str_print)
        self.writer.close()

    def _accuracy_epoch(self, data_loader):
        for _, mdl in self.model.items(): ## put everything in train mode
            mdl.eval()
        predictions, truths = [], []
        for xs, ys in data_loader:
            xs = xs.cuda()
            ys = ys.cuda()
            yp, _ = self.model["encoder"].classifier(xs)
            predictions.append(yp)
            truths.append(ys)

        # compute the number of accurate predictions
        accurate_preds = 0
        for pred, act in zip(predictions, truths):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds += (v.item() == pred[0].shape[0])

        # calculate the accuracy between 0 and 1
        accuracy = (accurate_preds * 1.0) / data_loader.dataset.__len__()

        return accuracy

    def _train_epoch(self, data_loaders):
        sup_batches = len(data_loaders["sup"])
        unsup_batches = len(data_loaders["unsup"])
        ## iterating parameter
        batches_per_epoch = sup_batches + unsup_batches
        lctrl = 0
        uctrl = 0
        epoch_losses = 0
        # loading data
        sup_iter = iter(data_loaders["sup"])
        unsup_iter = iter(data_loaders["unsup"])

        ## iterating training
        for ibatch in range(batches_per_epoch):
            if lctrl >= sup_batches:
                lctrl = 0
                sup_iter = iter(data_loaders["sup"])
            if uctrl >= unsup_batches:
                uctrl = 0
                unsup_iter = iter(data_loaders["unsup"])

            (xl, yl) = next(sup_iter)
            (xu, _) = next(unsup_iter)
            lctrl += 1
            uctrl += 1

            loss = self._train_iteration(xu, xl, yl)

            ## adding to losses
            epoch_losses += loss.item()

            ## print screen
            printr("==> [{}/{}]: {}".format(ibatch, batches_per_epoch, loss.item()))

        losses = {}
        losses["train loss"] = epoch_losses/batches_per_epoch
        return losses

    def _train_iteration(self, xu, xl, yl):
        if self.use_cuda:
            xu = xu.cuda()
            xl = xl.cuda()
            yl = yl.cuda()
        ## reconstruction loss
        self.optimizer["encoder"].zero_grad()
        self.optimizer["decoder"].zero_grad()
        loss_recon = self._loss_recon(xu) + self._loss_recon(xl, yl)

        loss_recon.backward()
        self.optimizer["encoder"].step()
        self.optimizer["decoder"].step()
        ## KL loss
        self.optimizer["comparator"].zero_grad()
        #self.optimizer["decoder"].zero_grad()
        loss_kl = self._loss_kl(xu) + self._loss_kl(xl, yl)
        #loss_kl = self._loss_kl_new(xu, xl, yl)
        loss_kl.backward()
        self.optimizer["comparator"].step()

        ## supervised loss
        self.optimizer["supervisor"].zero_grad()
        #self.optimizer["decoder"].zero_grad()
        loss_supervised = self._loss_supervised(xl, yl)
        loss_supervised.backward()
        self.optimizer["supervisor"].step()


        ## return loss
        loss = loss_recon + loss_kl + loss_supervised
        return loss

    def _loss_recon(self, xs, ys=None):
        zs,_ = self.model["encoder"](xs, ys)
        x_recon = self.model["decoder"](zs)
        return F.mse_loss(x_recon, xs)


    def _loss_kl(self, xs, ys=None):
        _, z_dist = self.model["encoder"](xs, ys)
        mu = z_dist["mean"]
        logvar = z_dist["logvar"]
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def _loss_kl_new(self,xu,xl,yl):
        _, z_u = self.model["encoder"](xu)
        _, z_l = self.model["encoder"](xl, yl)

        mu_u, logvar_u = z_u["mean"], z_u["logvar"]
        mu_l, logvar_l = z_l["mean"], z_l["logvar"]

        ## l~P, u~Q
        loss = -0.5*(1 + logvar_l - logvar_u - logvar_l.exp()/logvar_u.exp() - (mu_l-mu_u).pow(2)/logvar_u.exp())
        return torch.sum(loss)

    def _loss_supervised(self, xs, ys):
        _, yp = self.model["encoder"].classifier(xs)
        return F.binary_cross_entropy(yp, ys)

if __name__ == "__main__":
    import argparse, os
    import numpy as np
    from torch import optim
    from vae import Encoder, Decoder
    from mnist_cached import MNISTCached, setup_data_loaders
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=50)
    parser.add_argument("--total-epochs", "-e", type=int, default=20)
    parser.add_argument("--num-labeled-data", "-nl", type=int, default=100)
    parser.add_argument("--latent-size", "-ls", type=int, default=10)

    parser.add_argument("--use-cuda", "-gpu", type=int, default=1)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", "-m", type=str, default="model.hdf5")
    args = parser.parse_args()

    np.random.seed(args.seed)

    ## use cuda
    use_cuda = args.use_cuda and torch.cuda.is_available()

    if use_cuda:
        print("Using CUDA...\n")

    ## Loading model structure and current state
    model = {"encoder": Encoder(latent_size=args.latent_size),
             "decoder": Decoder(latent_size=args.latent_size + 10)}

    ## Optimizer
    optimizer = {'encoder': optim.Adam(model["encoder"].parameters(), lr=args.learning_rate),
                 'decoder': optim.Adam(model["decoder"].parameters(), lr=args.learning_rate),
                 'comparator': optim.Adam(model["encoder"].parameters(), lr=args.learning_rate),
                 'supervisor': optim.Adam(model["encoder"].parameters(), lr=args.learning_rate*0.1)}

    ## Loading MNIST dataset
    root = '~/Documents/Workspace/Cinnamon/Data/MNIST/'
    data_loaders = setup_data_loaders(MNISTCached, use_cuda=False, batch_size=args.batch_size,
                                      sup_num=args.num_labeled_data, root=root)

    ## Training
    trainer = Trainer(model, optimizer, use_cuda=use_cuda)
    trainer.train(data_loaders, epochs=args.total_epochs)




