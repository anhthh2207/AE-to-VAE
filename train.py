from model import VaiationalAutoEncoder, AutoEncoder
from data import get_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import freeze_support
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Hyperparameters for training')
parser.add_argument('-m', '--model', help='Model')
parser.add_argument('-i', '--input_size', type = int, help='Input size of the image (height = width)')
parser.add_argument('-c', '--channels', type = int, help='Number of channels')
parser.add_argument('-l', '--latent_size',type = int, default = 2, help='Latent size of the image')
parser.add_argument('-b', '--batch_size', type = int, default = 128, help='Batch size')
parser.add_argument('-d', '--data_set', help='Data set')
parser.add_argument('-ne', '--num_epochs', type = int, default = 10, help='Number of epochs')
parser.add_argument('-lr', '--learning_rate', type = int, default = 0.001, help='Learning rate')
parser.add_argument('-rt', '--reconstruct_type', default = 'mse', help='Loss type')
args = parser.parse_args()


if __name__ == '__main__':
    freeze_support()
    
    model_  = args.model
    input_size  = args.input_size
    latent_size = args.latent_size
    batch_size  = args.batch_size
    num_epochs  = args.num_epochs
    learning_rate = args.learning_rate
    data_set    = args.data_set
    channels    = args.channels
    reconstruct_type = args.reconstruct_type

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the data
    train_loader, test_loader = get_data_loaders(batch_size, data_set)

    # initiate the model
    assert model_ in ['vae', 'ae'], "Model name is not correct! It should be either vae or ae." 
    if model_ == "vae":
        model = VaiationalAutoEncoder(input_size, channels, latent_size, device = device).to(device)
    elif model_ == 'ae':
        model = AutoEncoder(input_size, channels, latent_size, device = device).to(device)
    print(model)

    # define the optimizer and train the model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Start training...")
    model.train(reconstruct_type, train_loader, optimizer, num_epochs, input_size, channels, latent_size)
    print("Training is done!")

    # save the model
    model_path = 'models/{}_{}_{}_{}.pth'.format(model_, latent_size, reconstruct_type, data_set)
    torch.save(model.state_dict(), model_path)

    # plot the loss
    loss_values = model.loss_values
    plt.figure(figsize=(8, 6))
    plt.plot(loss_values)
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Loss - {} {} {} {}'.format(model_, latent_size, reconstruct_type, data_set))
    plt.savefig('loss/{}_{}_{}_{}.png'.format(model_, latent_size, reconstruct_type, data_set))
    plt.show()