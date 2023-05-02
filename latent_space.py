from model import VaiationalAutoEncoder, AutoEncoder
from data import get_data_loaders
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Hyperparameters for training')
parser.add_argument('-m', '--model_name', help='Model name (model_latentSize_reconstructType_dataSet)')
args = parser.parse_args()




if __name__ == '__main__':
    freeze_support()
    
    model_name = args.model_name
    model_path = 'models/{}.pth'.format(model_name)
    x = model_name.split('_')
    assert len(x) == 4, 'model name must be in the format of model_latentSize_reconstructType_dataSet'
    model_ = x[0]
    assert model_ == 'vae' or model_ == 'ae', 'model must be vae or ae'
    latent_size = int(x[1])
    reconstruct_type = x[2]
    data_set = x[3]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_set == 'mnist' or data_set == 'fashionmnist':
        input_size = 28
        channels = 1
    elif data_set == 'cifar10':
        input_size = 32
        channels = 3
      
    # load and use the model
    model_path = 'models/{}.pth'.format(model_name)
    if model_ == 'vae':
        model = VaiationalAutoEncoder(input_size, channels, latent_size, device).to(device)
    elif model_ == 'ae':
        model = AutoEncoder(input_size, channels, latent_size, device).to(device)
    model.load_state_dict(torch.load(model_path))

    # load data using get_data_loaders
    train_loader, test_loader = get_data_loaders(50000, data_set)
    # visulize the latent space of autoencoder
    with torch.no_grad():
        for data, labels in train_loader: # take data and labels from the training set
            data = data.to(device)
            labels = labels.to(device)
            if model_ == 'vae':
                _, _, _, z = model(data)
            elif model_ == 'ae':
                z, _ = model.encoder(data)
            z = z.cpu().numpy()
            labels = labels.cpu().numpy()

            # display the data in the latent space
            plt.figure(figsize=(8, 6))
            plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='rainbow')
            plt.colorbar()
            plt.savefig('latent space/{}.png'.format(model_name))
            plt.show()
            break