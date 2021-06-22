# aug_robust_blogpost

## Setup

Install fork of robustness package:

```bash
git clone https://github.com/iamgroot42/robustness.git
cd robustness
python setup.py install
```

## Experiments

1. Model without data-augmentation

    `python -m robustness.main --dataset cifar --data data/ --arch resnet50 --out-dir models/ --exp-name standard_noaug --data-aug 0 --adv-train 0`

2. Model with data-augmentation (faulty worker-init)

    `python -m robustness.main --dataset cifar --data data/ --arch resnet50 --out-dir models/ --exp-name standard_aug --data-aug 1 --adv-train 0`

3. Model with data-augmentation (fixed worker-init)

    `python -m robustness.main --dataset cifar --data data/ --arch resnet50 --out-dir models/ --exp-name standard_fixed_aug --data-aug 1 --adv-train 0 --better_init 1`
