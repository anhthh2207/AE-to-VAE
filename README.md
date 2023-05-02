# VAE and AE with PyTorch
Variational Autoencoder and Autoencoder with PyTorch. </p>
Models can be trained on the MNIST, FashionMNIST or Cifar10 dataset. </p>
# Usage

To train model:

```bash
python train.py -m <model> -i <input_size> -c <channels> -l <latent_size> -b <batch_size> -d <data_set> -ne <number_of_epochs> -lr <learning_rate> -rt <reconstruct_type>
```

To load and use model for image generation:
```bash
python img_generate.py -m <name_of_trained_model>
```

To visualize latent space:
```bash
python latent_space.py -m <name_of_trained_model>
```

# References
[1] [Variational AutoEncoders (VAE) with PyTorch](https://avandekleut.github.io/vae/) </p>
[2] [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

# Further Readings
[1] [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) </p>
[2] [Beta-Variational Autoencoder as an Entanglement Classifier](https://arxiv.org/abs/2004.14420)</p>
[3] [Learning Structured Output Representation using Deep Conditional Generative Models](https://arxiv.org/abs/2201.09874) </p>
[4] [Recurrent World Models Facilitate Policy Evolution](https://papers.nips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html)
