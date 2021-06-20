from model_utils import train_model
from data_utils import CIFAR
from models import VGG19
import torch.nn as nn
import os
import numpy as np
import torch as ch
import random
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def set_randomness(seed):
    np.random.seed(seed)
    random.seed(seed)
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)
    ch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train model for')
    parser.add_argument('--bs', type=int, default=1500, help='batch size')
    parser.add_argument('--save', required=True, help='where to save model')
    parser.add_argument('--seed', type=int, default=0, help='seed for deterministic experiments')
    parser.add_argument('--augment', action="store_true",
                        help='use data augmentations when training models?')
    parser.add_argument('--adv', action="store_true",
                        help='use adversarial training?')
    parser.add_argument('--proper_init', action="store_true",
                        help='proper worker-function initialization')
    args = parser.parse_args()

    # Set controllable randomness
    set_randomness(args.seed)

    # Define network
    net = VGG19(num_classes=10).cuda()
    net = nn.DataParallel(net)

    # Get data
    ds = CIFAR(augment=args.augment)
    loaders = ds.get_loaders(args.bs, args.proper_init)

    # Train model
    metrics = train_model(net, loaders, epochs=args.epochs, adv=args.adv)

    # Save training metrics (to plot later)
    np.save(os.path.join("./metrics", args.save), metrics)

    # Save model (for later evaluation)
    state_dict = net.module.state_dict()
    ch.save(state_dict, os.path.join("./models", args.save))
