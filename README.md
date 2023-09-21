# ResNeXt_Cifar_10

This is the practice for the implementation of ResNeXt.  Models are trained with Cifar 10.

## Prerequisites
- Pytorch 2.0.1
- Python 3.11.4
- Window 11
- conda 23.7.4

## Training
```
# GPU training
python train.py -m Resnect29-8x64d -e 300 -lr 0.01 -b 128 -s 32 -d outputs
```

## Testing
```
python test.py -m Resnet20 -e 200 -lr 0.01 -b 128 -s 32 -d outputs
```

## Result (Accuracy)

Pretrained model should be downloaded if you click the name of Model.

| Model             | Acc.        | Param.        |
| ----------------- | ----------- |----------- |
| [ResNet20](https://drive.google.com/file/d/1BIklR-0qXeWw9zhEscAPZZQrk6Q98zFQ/view?usp=drive_link)          | 91.48%      |  0.27M     |
| [ResNet32](https://drive.google.com/file/d/1ekH2JjeBiaUtZ2cUxP63PWg0DQlKm8vj/view?usp=drive_link)          | 92.64%      | 0.46M      |
| [ResNet44](https://drive.google.com/file/d/1TqbykyFFvf2QxZbwv-k3G0L9iJ90Hd1e/view?usp=drive_link)         | 92.67%      | 0.66M      |
| [ResNet56](https://drive.google.com/file/d/1u_k_acCgvQYCjbQdWcJ43mzvka6llX3e/view?usp=drive_link)          | 92.47%      | 0.85M     |
| [Plain20](https://drive.google.com/file/d/1YUWiG6LII_UPZmgooB3IhYLnGqaBQRsE/view?usp=drive_link)          | 90.38%      | 0.27M      |
| [Plain32](https://drive.google.com/file/d/17XWnRGJIDa2MuvGJIGproa9Xjnx6glj1/view?usp=drive_link)         | 89.65%      | 0.46M     |

## Plot
Plots are in the plots folder.
