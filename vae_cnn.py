#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from models import conv3x3


class VAE(nn.Module):
    def __init__(self, input_dim=3, h_dim=6 * 6 * 64, z_dim=512):
        super(VAE, self).__init__()
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        self.encoder = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 6x6x64
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 13x13x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 27x27x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 55x55x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 111x111x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2),  # 224x224x3
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

    def reparameterize(self, mu, logvar):
        eps = torch.randn(mu.size(0), mu.size(1)).cuda()
        z = mu + eps * torch.exp(logvar / 2)
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 64, 6, 6)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


def loss_fn(recon_x, x, mu, logvar, current_epoch, wramup_epochs):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # wram-up procedure, KLD from 0 to 1
    beta = (current_epoch / wramup_epochs) if current_epoch <= wramup_epochs else 1
    return BCE + beta * KLD, BCE, KLD * beta


def compare(x, model):
    recon_x, _, _ = model(x)
    return torch.cat([x, recon_x])


def generate_image(model, name):
    sample = torch.randn(32, 64, 6, 6).cuda()
    compare_x = model.decoder(sample)

    save_image(compare_x.detach().cpu(), name)


def main(args):
    # log_dir = './log/warmup_with_batchnorm_mse_aug'
    log_dir = args.log_dir
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load Data
    bs = 32
    dataset = datasets.ImageFolder(root='data/rl-sight/', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    len(dataset.imgs), len(dataloader)

    # Fixed input for debugging
    fixed_x, _ = next(iter(dataloader))
    save_image(fixed_x, log_dir + '/real_image.png')

    image_channels = fixed_x.size(1)
    model = VAE(input_dim=image_channels, z_dim=args.z_dim).to(device)
    model.train()

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load('./log/warmup_with_batchnorm_mse_aug/99.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['checkpoint'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.00001, amsgrad=True)
    lr_scheduler = MultiStepLR(optimizer, milestones=args.step_lr, gamma=0.1)

    epochs = 50
    for epoch in range(start_epoch, epochs + start_epoch):
        for idx, (images, _) in enumerate(dataloader):
            images = images.cuda()
            recon_images, mu, logvar = model(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar, epoch, 20)
            if idx % 100 == 0:
                print("bce loss {} kld loss {}".format(bce / bs, kld / bs))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1, epochs + start_epoch, loss.item() / bs, bce.item() / bs, kld.item() / bs)
        print(to_print)
        writer.add_scalar('total loss', loss.item() / bs, epoch)
        writer.add_scalar('bce loss', bce.item() / bs, epoch)
        writer.add_scalar('kld loss', kld.item() / bs, epoch)

        compare_x = compare(fixed_x.cuda(), model)
        save_image(compare_x.detach().cpu(), log_dir + '/' + str(epoch) + '_epoch_sample_image.png')

        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'checkpoint': model.state_dict()}, log_dir + '/' + str(epoch) + '.pth')
            generate_image(model, log_dir + '/' + str(epoch) + '_generated.png')

    torch.save(model.state_dict(), log_dir + '/final.pth')
    # decoder generates image
    generate_image(model, log_dir + '/generated.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--z_dim', default=32)
    # optimizer
    parser.add_argument('--step_lr', type=list, default=[10, 30])
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--log_dir', required=True)
    args = parser.parse_args()
    main(args)
