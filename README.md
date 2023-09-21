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
python train.py -m Resnext29-8x64d -e 300 -lr 0.01 -b 128 -s 32 -d outputs
```

## Testing
```
python test.py -m Resnext29-8x64d -e 300 -lr 0.01 -b 128 -s 32 -d outputs
```

## Result (Accuracy)

Pretrained model should be downloaded if you click the name of Model.

| Model             | Acc.        | Param.        |
| ----------------- | ----------- |----------- |
| [ResNet29](https://drive.google.com/file/d/1BIklR-0qXeWw9zhEscAPZZQrk6Q98zFQ/view?usp=drive_link)          | 91.48%      |  0.27M     |
| [ResNeXt29-8x64d](https://drive.google.com/file/d/1ekH2JjeBiaUtZ2cUxP63PWg0DQlKm8vj/view?usp=drive_link)          | 92.64%      | 0.46M      |
| [ResNeXt29-16x4d](https://drive.google.com/file/d/1TqbykyFFvf2QxZbwv-k3G0L9iJ90Hd1e/view?usp=drive_link)         | 92.67%      | 0.66M      |


## Plot
Plots are in the plots folder.
