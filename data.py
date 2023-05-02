import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size, data_set = "mnist"):
    """
        This function returns the train and test data loaders for the specified data set. 
        For now, this function can work with MNIST, CIFAR-10, and Fashion-MNIST data sets.
    """

    assert data_set in ["mnist", "cifar10", "fashionmnist"], "data_set should be one of the following: mnist, cifar10, fashionmnist."
    if data_set == "mnist":
        train_set = torchvision.datasets.MNIST(root='./data', 
                                                train=True, 
                                                download=True, 
                                                transform=transforms.ToTensor())
        test_set = torchvision.datasets.MNIST(root='./data', 
                                                train=False, 
                                                download=True, 
                                                transform=transforms.ToTensor())
    elif data_set == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root='./data', 
                                                train=True, 
                                                download=True, 
                                                transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(root='./data', 
                                                train=False, 
                                                download=True, 
                                                transform=transforms.ToTensor())
    elif data_set == "fashionmnist":
        train_set = torchvision.datasets.FashionMNIST(root='./data', 
                                                train=True, 
                                                download=True, 
                                                transform=transforms.ToTensor())
        test_set = torchvision.datasets.FashionMNIST(root='./data', 
                                                train=False, 
                                                download=True, 
                                                transform=transforms.ToTensor())

                                    
    train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                num_workers=2)

    return train_loader, test_loader