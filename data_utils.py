import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Fix for repeated random augmentation issue
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Dataset:
    def __init__(self, augment):
        self.augment = augment
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_loaders(self, batch_size, proper_init, num_workers=2):
        wif = None
        if proper_init:
            wif = worker_init_fn
            
        trainloader = DataLoader(self.trainset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 worker_init_fn=wif,
                                 shuffle=True)
        testloader = DataLoader(self.testset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                worker_init_fn=wif,
                                shuffle=False)
        return (trainloader, testloader)


class CIFAR(Dataset):
    def __init__(self, augment=False):
        super(CIFAR, self).__init__(augment)

        if augment:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(.25,.25,.25),
                transforms.RandomRotation(2),
                transforms.ToTensor(),
            ])
        else:
            self.train_transform = self.test_transform
        

        self.trainset = CIFAR10(root='./data', train=True,
                                download=True,
                                transform=self.train_transform)
        self.testset = CIFAR10(root='./data', train=False,
                               download=True,
                               transform=self.test_transform)
