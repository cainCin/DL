import torch
import torch.nn as nn
import os, sys

def printr(string):
    sys.stdout.write("\r\033")
    sys.stdout.write(string)
    sys.stdout.flush()

class Trainer(object):
    """
    Definition the default trainer in deep learning approach
    """
    def __init__(self, model, optimizer,
                        criterion=None,
                        checkpoint=None,
                        device=None,
                        tensorboard=False):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint= checkpoint

        ## observation
        self.tensorboard = tensorboard
        ## initilization
        if self.tensorboard: from tensorboardX import SummaryWriter
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion.to(self.device)
        for _, mdl in self.model.items():
            mdl = mdl.to(self.device)

        ## Load checkpoint if exist
        if self.checkpoint is not None:     # save checkpoint active
            if os.path.isfile(self.checkpoint):
                print("load %s ..." %self.checkpoint)
                checkpoint = torch.load(self.checkpoint)
                #print(checkpoint)
                for key, mdl in self.model.items():
                    if (key + "_state_dict") in checkpoint.keys():
                        mdl.load_state_dict(checkpoint[key + "_state_dict"])
                self.epoch = checkpoint["epoch"]
                self.cur_best_acc = checkpoint["acc"]

        else:
            self.epoch = 0
            self.cur_best_acc = 0.0

    ## definition main method
    def train(self, dataloaders, epochs=3):
        """
        train model with data in dataloaders
        :param dataloaders:
        :param epochs:
        :return:
        """
        self.writer = None if not self.tensorboard else SummaryWriter()
        ##========== CONFIGURATION ======================
        trainloaders = dict((k, v) for (k, v) in dataloaders.items() if "train" in k)
        valloaders = dict((k, v) for (k, v) in dataloaders.items() if "val" in k)
        ##========== TRAINING ===========================
        print("========> Start to train ....................")
        for epoch in range(self.epoch, epochs):
            print("--- Epoch [{}/{}]:--------------".format(epoch, epochs))
            self.epoch = epoch
            ## ==================================================================
            if len(trainloaders) > 0:   ### Training
                self._train_epoch(trainloaders)
            else:
                print("!!!!! Error: no train set...")

            ## ==================================================================
            if len(valloaders) > 0:
                self._val_epoch(valloaders)
            else:
                print("!!!!! Error: no val set...")

    def _train_epoch(self, trainloaders, mode="train"):
        num_batches = 100
        ## ======== INIT loader ===================
        num_batches = len(trainloaders[mode])
        init_loader = iter(trainloaders[mode])

        ## ======== TRAIN BATCHES =================
        average_loss = dict()
        for batch in range(num_batches):
            ## displaying
            msg = "Train [{}]".format("="*(20*batch//num_batches) + ">"*(20-20*batch//num_batches))
            ## updating model for current batch
            (xs, ys) = init_loader.next()
            losses = self._train_batch(xs, ys)

            ## updating in printing msg
            for name, value in losses.items():
                if batch == 0:
                    average_loss[name] = value
                else:
                    average_loss[name] = (average_loss[name]*batch + value)/(batch + 1)
                msg += "{0}: {1} (aver: {2})".format(name, value, average_loss[name])

            ## printing
            printr(msg)
        print("done.")

    def _train_batch(self, xs, ys=None):
        ## renitializing gradient
        for _, opt in self.optimizer.items():
            opt.zero_grad()

        ## get losses via loss function
        loss, losses = self._get_loss(xs, ys)

        ## backward
        loss.backward()
        for _, opt in self.optimizer.items():
            opt.step()

        return losses

    def _get_loss(self, xs, ys=None):
        ## feeding data into device
        xs = xs.to(self.device)
        ys = None if ys is None else ys.to(self.device)
        ## initializing
        losses = dict()
        for key, mdl in self.model.items():
            y_pred = mdl(xs)
            losses[key] = self.criterion(y_pred, ys)

        loss = 0
        losses_out = dict()
        for key, value in losses.items():
            loss = loss + value
            losses_out[key] = value.item()

        return loss, losses_out

    def _val_epoch(self, valloaders, mode="val"):
        num_batches = len(valloaders[mode])
        init_loader = iter(valloaders[mode])
        acc = 0
        for batch in range(num_batches):
            ## displaying
            msg = "Train [{}]".format("="*(20*batch//num_batches) + ">"*(20-20*batch//num_batches))
            ## load data from current batch
            (xs, ys) = init_loader.next()

            ## get acc with current batch
            acc += self._get_acc(xs, ys)

        acc = acc / valloaders[mode].dataset.__len__()
        
        if acc > self.cur_best_acc:
            self.cur_best_acc = acc
            #if self.checkpoint is not None:
            #    self._save_checkpoint()
            
        print("ACC: {:.4f}".format(acc))



    def _get_acc(self, xs, ys):
        ## feed into device
        xs = xs.to(self.device)
        ys = ys.to(self.device)

        ## get acc
        acc = 0
        for _, mdl in self.model.items():
            y_pred = mdl(xs)
            y_pred = torch.argmax(y_pred, dim=1)
            acc += y_pred.eq(ys).sum().item()

        return acc/len(self.model)
        
    def _save_checkpoint(self):
        checkpoint = dict()
        checkpoint["epoch"] = self.epoch
        checkpoint["acc"] = self.cur_best_acc
        for key, mdl in self.model.items():
            checkpoint[key + "_state_dict"] = mdl.state_dict()
        
        torch.save(checkpoint, self.checkpoint)
        
    def _export(self, name=None):
        for key, mdl in self.model.items():
            if name is None:
                torch.save(mdl, key)
            else:
                torch.save(mdl, name + key)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(64*7*7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, data, label=None):
        """
        Implementation of 1 CNN + 1 GCN
        :param data: input batch of size (B, C, H, W)
        :return:
        """
        ##- CNN layer 1
        data = F.relu(self.conv1(data))
        data = self.maxpool1(data)
        ##- CNN layer 2
        data = F.relu(self.conv2(data))
        data = self.maxpool2(data)

        ## FC
        x = F.elu(self.fc1(data.view(-1, 64*7*7)))

        return F.log_softmax(self.fc2(x), dim=1)

if __name__ == "__main__":
    import argparse
    import torchvision
    import torchvision.transforms as transforms
    from torch import optim
    import torch.nn.functional as F

    #dataloaders = {"train": []}
    #trainer = Trainer({}, {})
    #trainer.train(dataloaders=dataloaders)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--total-epochs", "-e", type=int, default=50)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
    parser.add_argument("--threshold", "-thres", type=float, default=0.8)

    parser.add_argument("--use-cuda", "-gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", "-m", type=str, default="model.pth")
    args = parser.parse_args()

    ## SETUP CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Loading MNIST dataset
    trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.,), (1.0,))])

    root = "/home/cain/Data"
    sup_dataset = torchvision.datasets.MNIST(root=root, train=True,
                                             download=True, transform=trans,
                                             target_transform=None)
    test_dataset = torchvision.datasets.MNIST(root=root, train=False,
                                              download=True, transform=trans,
                                              target_transform=None)

    data_loaders = {"train": torch.utils.data.DataLoader(dataset=sup_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True),
                    "val": torch.utils.data.DataLoader(dataset=test_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=False)}

    ## creating model
    model = {"recognition": Net().to(device)}
    ## optimizer
    optimizer = {"recognition": torch.optim.Adam(model["recognition"].parameters(), lr=0.001)}

    ## Test Trainer module
    trainer = Trainer(model=model, optimizer=optimizer, device=device)
    trainer.train(dataloaders=data_loaders)