import torch as ch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


def epoch(net, loader, optim=None, adv=False):
    criterion = nn.CrossEntropyLoss()
    is_train = (optim is not None)
    
    if is_train:
        net.train()
        prefix = "[Train]"
    else:
        net.eval()
        prefix = "[Val]"

    n_samples, running_loss, correct = 0, 0.0, 0.0
    running_loss_, correct_ = 0.0, 0
    iterator = tqdm(loader)
    with ch.set_grad_enabled(is_train):
        for data in iterator:

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            if adv:
                # Replace clean example with adversarial example for adversarial training
                eps = 8/255
                nb_iters = 40
                eps_iter = 2.5 * eps/ nb_iters
                with ch.set_grad_enabled(True):
                    # Adversarial example needs gradient computation
                    inputs_adv = projected_gradient_descent(
                        net, inputs, eps, eps_iter, nb_iters, np.inf)

            if is_train:
                # zero the parameter gradients
                optim.zero_grad()

            # forward + backward + optimize
            if adv:
                outputs = net(inputs_adv)
            else:
                outputs = net(inputs)

            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optim.step()
            
            # print statistics
            with ch.no_grad():
                _, predicted = ch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                bs = labels.shape[0]
                running_loss += loss.item() * bs
                n_samples += bs
                if adv and not is_train:
                    outputs = net(inputs)
                    loss_clean = criterion(outputs, labels)

                    running_loss_ += loss_clean.item() * bs
                    _, predicted = ch.max(outputs.data, 1)
                    correct_ += (predicted == labels).sum().item()

                    iterator.set_description("%s [Clean] Loss: %.3f Acc: %.3f | [Adv] Loss: %.3f Acc: %.3f" % (
                        prefix, running_loss_ / n_samples, correct_ / n_samples, running_loss / n_samples, correct / n_samples))
                else:
                    iterator.set_description("%s Loss: %.3f Acc: %.3f" % (
                        prefix, running_loss / n_samples, correct / n_samples))
                        
    return (running_loss_ / n_samples, correct_ / n_samples), (running_loss / n_samples, correct / n_samples)


def train_model(net, loaders, epochs, adv):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    metrics = []

    for e in range(epochs):
        print("Epoch %d/%d" % (e+1, epochs))
        # Train
        train_metrics = epoch(net, loaders[0], optimizer, adv=adv)

        # Val
        eval_metrics = epoch(net, loaders[1], adv=adv)
        
        metrics.append([train_metrics, eval_metrics])
        print()
    
    return np.array(metrics)
