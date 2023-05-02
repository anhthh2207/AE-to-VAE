import datetime
import torch
import torch.nn.functional as F
from torch import nn


# define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_size = 28, channels = 1, latent_size = 8):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.channels = channels

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(channels * input_size * input_size, 14*14)
        self.linear2 = nn.Linear(14*14, 10*10)
        self.linear3 = nn.Linear(10*10, latent_size)
        self.linear4 = nn.Linear(10*10, latent_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        mean = self.linear3(x)
        log_var = self.linear4(x)

        return mean, log_var



# define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_size = 8, channels = 1, output_size = 28):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.output_size = output_size
        self.channels = channels

        self.linear1 = nn.Linear(latent_size, 10*10)
        self.linear2 = nn.Linear(10*10, 14*14)
        self.linear3 = nn.Linear(14*14, channels * output_size * output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        return x



# define the VAE
class VaiationalAutoEncoder(nn.Module):
    """
        Variational AutoEncoder. 
        When initialize the model, you should specify the device (cpu or cuda) and move the model to that device at the same time.
    """
    def __init__(self, input_size = 28, channels = 1, latent_size = 8, device = 'cpu'):
        super(VaiationalAutoEncoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.channels = channels
        self.device = device  
        self.loss_values = []  # loss_values attribute to save the change of loss function over training steps

        self.encoder = Encoder(input_size, channels, latent_size)
        self.decoder = Decoder(latent_size, channels, input_size)


    def forward(self, x):
        mean, log_var = self.encoder(x)
        # sample in the latent space
        sample = torch.randn(mean.size()).to(self.device)
        z = mean + torch.exp(log_var/2) * sample # latent variable
        x = self.decoder(z)

        return x, mean, log_var, z


    def loss_function(self, x_hat, x, mean, log_var, reconstruct_type = "mse"):
        """
            Loss function for VAE is sum of reconstruction loss which can be calculated by binarycrossentropy (for binary input) or MSE, l1 and KL divergence.
            Note that, binary_cross_entroy only works for binary input, so we need to use sigmoid to convert the output of decoder to probability;
                about the input, it shoulde be binary already after using ToTensor() in torchvision.transforms.

            reconstruct_type should be either "mse", "l1" or "binarycrossentropy".
            All the input should be flatten to 1D tensor.
        """
        assert reconstruct_type in ["binarycrossentropy", "mse", 'l1'], "reconstruct_type should be either binarycrossentropy or mse, l1"
        if reconstruct_type == "binarycrossentropy":
            reconstruction_Loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        elif reconstruct_type == "mse":
            reconstruction_Loss = F.mse_loss(x_hat, x, reduction='sum')
        elif reconstruct_type == "l1":
            reconstruction_Loss = F.l1_loss(x_hat, x, reduction='sum')

        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) # KL divergence for two gaussian distribution

        return reconstruction_Loss + kl_divergence

    
    def train(self, reconstruct_type, train_loader, optimizer, num_epochs, input_size = 28, channels = 1, latent_size = 8):
        """
            Train the VAE models.
            reconstruct_type should be either "mse", "l1" or "binarycrossentropy".
        """
        for epoch in range(num_epochs):
            for i, (images, label) in enumerate(train_loader):
                x = images.to(self.device)
                optimizer.zero_grad()

                x_hat, mean, log_var, _ = self.forward(x)
                x = x.view(-1, channels* input_size*input_size)
                mean = mean.view(-1, latent_size)
                log_vag = log_var.view(-1, latent_size)

                loss = self.loss_function(x_hat, x, mean, log_vag, reconstruct_type = reconstruct_type)
                loss.backward()
                optimizer.step()
                self.loss_values.append(loss.item()) # save loss value

                if (i+1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss(reconstruction {}): {:.4f}, Time: {}".format(epoch+1, num_epochs, i+1, len(train_loader), reconstruct_type, loss.item(), datetime.datetime.now()))



# define AutoEncoder
class AutoEncoder(nn.Module):
    """
        AutoEncoder. 
        When initialize the model, you should specify the device (cpu or cuda) and move the model to that device at the same time.
    """
    def __init__(self, input_size = 28, channels = 1, latent_size = 8, device = 'cpu'):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.channels = channels
        self.device = device
        self.loss_values = [] # loss_values attribute to save change of loss function over trainging steps

        self.encoder = Encoder(input_size, channels, latent_size)
        self.decoder = Decoder(latent_size, channels, input_size)


    def forward(self, x):
        z, _ = self.encoder(x) # since I reuse the Encoder which is supposes to be used in VAE, I just need one output hence just ignore the log_var
        x = self.decoder(z)

        return x


    def loss_function(self, x_hat, x, reconstruct_type = "mse"):
        """
            Loss function for AE is reconstruction loss which can be calculated by binarycrossentropy (for binary input) or MSEm, l1.
            Note that, binary_cross_entroy only works for binary input, so we need to use sigmoid to convert the output of decoder to probability;
                about the input, it shoulde be binary already after using ToTensor() in torchvision.transforms.

            reconstruct_type should be either "mse", "l1" or "binarycrossentropy".
            All the input should be flatten to 1D tensor.
        """
        assert reconstruct_type in ["binarycrossentropy", "mse", 'l1'], "reconstruct_type should be either binarycrossentropy or mse, l1"
        if reconstruct_type == "mse":
            reconstruction_Loss = F.mse_loss(x_hat, x, reduction='sum')
        elif reconstruct_type == "l1":
            reconstruction_Loss = F.l1_loss(x_hat, x, reduction='sum')
        elif reconstruct_type == "binarycrossentropy":
            reconstruction_Loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    
        return reconstruction_Loss


    def train(self, reconstruct_type, train_loader, optimizer, num_epochs, input_size = 28, channels = 1, latent_size = 8):
        """
            Train the AE models.
            reconstruct_type should be either "mse", "l1" or "binarycrossentropy".
        """
        for epoch in range(num_epochs):
            for i, (images, label) in enumerate(train_loader):
                x = images.to(self.device)
                optimizer.zero_grad()

                x_hat = self.forward(x)
                x = x.view(-1, channels* input_size*input_size)

                loss = self.loss_function(x_hat, x, reconstruct_type = reconstruct_type)
                loss.backward()
                optimizer.step()
                self.loss_values.append(loss.item())

                if (i+1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss(reconstruction {}): {:.4f}, Time: {}".format(epoch+1, num_epochs, i+1, len(train_loader), reconstruct_type, loss.item(), datetime.datetime.now()))
