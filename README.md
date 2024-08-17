# AlexNet-Implementation

"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton. 

Paper: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

## Architecture

![AlexNet Architecture](images/architecture.png)

## GPU POOR !!!

Currently, GPU Poor. So didn't train the model. But the model is as in the paper. Dive in and check.

## Info

For information about the model, following script is helpful.

```sh
python info.py
```

![AlexNet Information](images/info.png)

## Usage

Before running the script, place your data directory location for both train and test data in `root_dir="{DIR}"` here at [dataloader.py](./dataloader/dataloader.py)

```sh
python train.py --epochs 90 --in_channels 3 --num_classes 1000
```

## Citation

```
@inproceedings{10.5555/2999134.2999257,
author = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E.},
title = {ImageNet classification with deep convolutional neural networks},
year = {2012},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
abstract = {We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5\% and 17.0\% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overriding in the fully-connected layers we employed a recently-developed regularization method called "dropout" that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3\%, compared to 26.2\% achieved by the second-best entry.},
booktitle = {Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1},
pages = {1097–1105},
numpages = {9},
location = {Lake Tahoe, Nevada},
series = {NIPS'12}
}
```
