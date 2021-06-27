# CIFAR10 Experiments

## CIFAR10 Linf-robust accuracy

Robust model trained with ε = 8/255

| Model | Standard Accuracy (%) | Robust Accuracy (%), ε = 8/255 | Robust Accuracy (%), ε = 16/255 |
| ----------- | ----------- | ----------- | ----------- |
| Standard | 89.140 | 0.000 | 0.000 |
| Standard (with augmentation) | 94.720 | 0.000 | 0.000 |
| Standard (with fixed augmentation) | 94.620 | 0.000 | 0.000 |
| Robust | 79.520 | 44.370 | 15.680 |
| Robust (with augmentation) | 86.320 | 51.400 | 17.480 |
| Robust (with fixed augmentation) | 86.730 | 51.890 | 17.570 |

## CIFAR10 L2-robust accuracy

Robust model trained with ε = 1

| Model | Standard Accuracy (%) | Robust Accuracy (%), ε = 0.5 | Robust Accuracy (%), ε = 1 |
| ----------- | ----------- | ----------- | ----------- |
| Standard | 89.140 | 0.040 | 0.000 |
| Standard (with augmentation) | 94.720 | 0.240 | 0.000 |
| Standard (with fixed augmentation) | 94.620 | 0.220 | 0.000 |
| Robust | 78.190 | 61.740 | 42.830 |
| Robust (with augmentation) | 80.560 | 67.200 | 51.140 |
| Robust (with fixed augmentation) | 81.070 | 67.620 | 51.220 |

## ImageNet L2-robust accuracy

Robust model trained with ε = 3

| Model | Standard Accuracy (%) | Robust Accuracy (%), ε = 0.5 | Robust Accuracy (%), ε = 1 | Robust Accuracy (%), ε = 2 | Robust Accuracy (%), ε = 3 |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Standard (with augmentation) | 76.13 | 3.35 | 0.44 | 0.16 | 0.13 |
| Standard (with fixed augmentation) | ? | ? | ? | ? | ? |
| Robust (with augmentation) | 57.90 | 54.42 | 50.67 | 43.04 | 35.16 |
| Robust (with fixed augmentation) | ? | ? | ? | ? | ? |
