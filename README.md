# AlexNet-Implementation

## Architecture

![AlexNet Architecture](images/architecture.png)

"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton. 

Paper: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

## GPU POOR !!!

Currently, GPU Poor. So didn't train the model. But the model is as in the paper. Dive in and check.

## Info

For information about the model, following script is helpful.

```sh
python info.py
```

```sh
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
AlexNet                                  [1, 1000]                 --
├─ConvBlock: 1-1                         [1, 96, 27, 27]           --
│    └─Sequential: 2-1                   [1, 96, 27, 27]           --
│    │    └─Conv2d: 3-1                  [1, 96, 55, 55]           34,944
│    │    └─ReLU: 3-2                    [1, 96, 55, 55]           --
│    │    └─LocalResponseNorm: 3-3       [1, 96, 55, 55]           --
│    │    └─MaxPool2d: 3-4               [1, 96, 27, 27]           --
├─ConvBlock: 1-2                         [1, 256, 13, 13]          --
│    └─Sequential: 2-2                   [1, 256, 13, 13]          --
│    │    └─Conv2d: 3-5                  [1, 256, 27, 27]          614,656
│    │    └─ReLU: 3-6                    [1, 256, 27, 27]          --
│    │    └─LocalResponseNorm: 3-7       [1, 256, 27, 27]          --
│    │    └─MaxPool2d: 3-8               [1, 256, 13, 13]          --
├─ConvBlock: 1-3                         [1, 384, 13, 13]          --
│    └─Sequential: 2-3                   [1, 384, 13, 13]          --
│    │    └─Conv2d: 3-9                  [1, 384, 13, 13]          885,120
│    │    └─ReLU: 3-10                   [1, 384, 13, 13]          --
├─ConvBlock: 1-4                         [1, 384, 13, 13]          --
│    └─Sequential: 2-4                   [1, 384, 13, 13]          --
│    │    └─Conv2d: 3-11                 [1, 384, 13, 13]          1,327,488
│    │    └─ReLU: 3-12                   [1, 384, 13, 13]          --
├─ConvBlock: 1-5                         [1, 256, 6, 6]            --
│    └─Sequential: 2-5                   [1, 256, 6, 6]            --
│    │    └─Conv2d: 3-13                 [1, 256, 13, 13]          884,992
│    │    └─ReLU: 3-14                   [1, 256, 13, 13]          --
│    │    └─LocalResponseNorm: 3-15      [1, 256, 13, 13]          --
│    │    └─MaxPool2d: 3-16              [1, 256, 6, 6]            --
├─LinearBlock: 1-6                       [1, 4096]                 --
│    └─Sequential: 2-6                   [1, 4096]                 --
│    │    └─Dropout: 3-17                [1, 9216]                 --
│    │    └─Linear: 3-18                 [1, 4096]                 37,752,832
│    │    └─ReLU: 3-19                   [1, 4096]                 --
├─LinearBlock: 1-7                       [1, 4096]                 --
│    └─Sequential: 2-7                   [1, 4096]                 --
│    │    └─Dropout: 3-20                [1, 4096]                 --
│    │    └─Linear: 3-21                 [1, 4096]                 16,781,312
│    │    └─ReLU: 3-22                   [1, 4096]                 --
├─LinearBlock: 1-8                       [1, 1000]                 --
│    └─Sequential: 2-8                   [1, 1000]                 --
│    │    └─Linear: 3-23                 [1, 1000]                 4,097,000
│    │    └─ReLU: 3-24                   [1, 1000]                 --
==========================================================================================
Total params: 62,378,344
Trainable params: 62,378,344
Non-trainable params: 0
Total mult-adds (G): 1.14
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 5.27
Params size (MB): 249.51
Estimated Total Size (MB): 255.39
==========================================================================================
torch.Size([1, 1000])
```

## Usage

Before running the script, place your data directory location for both train and test data in `root_dir="{DIR}"` here at [dataloader.py](data_preprocessing/dataloader.py)

```sh
python train.py --epochs 90 --in_channels 3 --num_classes 1000
```
