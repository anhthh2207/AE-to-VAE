from model import VaiationalAutoEncoder, AutoEncoder
from data import get_data_loaders
from multiprocessing import freeze_support
import torch
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Hyperparameters for training')
parser.add_argument('-m', '--model_name', help='Model name (model_latentSize_reconstructType_dataSet)')
args = parser.parse_args()



if __name__ == '__main__':
    freeze_support()

    # model name and parameters
    model_name = args.model_name
    x = model_name.split('_')
    model = x[0]
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
    if model == 'vae':
        model = VaiationalAutoEncoder(input_size, channels, latent_size, device).to(device)

    elif model == 'ae':
        model = AutoEncoder(input_size, channels, latent_size, device).to(device)
    model.load_state_dict(torch.load(model_path))
    

    with torch.no_grad():
        z = 5*torch.randn(12*12,latent_size).to(device) -5  # radom from uniform distribution ranging from -5 to 5
        sample = model.decoder(z).cpu() # uniformly sample from the latent space
        
        # display the generated images 
        fig = plt.figure()
        for i in range(12*12):
            plt.subplot(12, 12, i+1)
            # reshape and display the images
            x = sample[i].reshape(channels, input_size, input_size)
            x = x.permute(1, 2, 0)
            plt.imshow(x) 
            plt.axis('off')

        plt.savefig('result/{}.png'.format(model_name))
        plt.show()
        
    print("Done!")
