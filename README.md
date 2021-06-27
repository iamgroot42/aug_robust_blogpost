# Experiments for blog post 'Reassessing adversarial training with fixed data augmentation'

## Setup

Install fork of robustness package:

```bash
git clone https://github.com/iamgroot42/robustness.git
cd robustness
python setup.py install
```

## Training models

1. Model without data-augmentation

    ```bash
    python -m robustness.main --dataset cifar --data data/ --arch resnet50 --out-dir models/ --exp-name standard_noaug --data-aug 0 --adv-train 0
    ```

2. Model with data-augmentation (faulty worker-init)

    ```bash
    python -m robustness.main --dataset cifar --data data/ --arch resnet50 --out-dir models/ --exp-name standard_aug --data-aug 1 --adv-train 0
    ```

3. Model with data-augmentation (fixed worker-init)

    ```bash
    python -m robustness.main --dataset cifar --data data/ --arch resnet50 --out-dir models/ --exp-name standard_fixed_aug --data-aug 1 --adv-train 0 --better_init 1
    ```

## Evaluating models

```bash
    python -m robustness.main --dataset cifar --data data/ --arch resnet50 --out-dir evals/ --attack-steps 20 --constraint inf --eps 8/255 --attack-lr 20/5100 --resume models/robust_fixed_aug/checkpoint.pt.best --eval-only 1 --adv-eval 1
```
