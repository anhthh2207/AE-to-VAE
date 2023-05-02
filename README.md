# VAE and AE with PyTorch
This repository contains the code for the Variational Autoencoder and Autoencoder with PyTorch. </p>
It can be used to train on the MNIST, FashionMNIST or Cifar10 dataset. </p>
# Usage

To train model:

```bash
python train.py -m <name_of_model> -i <input_size> -c <channels> l <latent_size> -b <batch_size> -d <data_set> -ne <number_of_epochs> -lr <learning_rate> -rt <reconstruct_type>
```

To load and use model for image generation:
```bash
python img_generate.py -m <name_of_model>
```

To visualize latent space:
```bash
python latent_space.py -m <name_of_model>
```

# References
[1] [Variational AutoEncoders (VAE) with PyTorch](https://avandekleut.github.io/vae/)