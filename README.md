# Semantic segmentation using unet

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

The goal of this project is to create a system for analyzing aerial and satellite imagery using deep learning that works on a variety of tasks and datasets. At the moment, we have implemented approaches for semantic segmentation, and are working on tagging/recognition. In the future we may add support for object detection. There is code for building Docker containers, running experiments on AWS EC2 using [AWS Batch](https://aws.amazon.com/batch/), loading and processing data, and constructing, training and evaluating models
using the [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) libraries.

### Semantic Segmentation
The goal of semantic segmentation is to infer a meaningful label such as "road" or "building" for each pixel in an image. Here is an example of an aerial image segmented using a model learned by our system.

![Example segmentation](results/unet/img/good1.png)

More details on this feature can be found in this [blog post](https://www.azavea.com/blog/2017/05/30/deep-learning-on-aerial-imagery/).

The following datasets and model architectures are implemented.

#### Datasets
* [ISPRS Potsdam 2D dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)
* [ISPRS Vaihingen 2D dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html)

#### Model architectures
* [FCN](https://arxiv.org/abs/1411.4038) (Fully Convolutional Networks) using [ResNets](https://arxiv.org/abs/1512.03385)
* [U-Net](https://arxiv.org/abs/1505.04597)
* [Fully Convolutional DenseNets](https://arxiv.org/abs/1611.09326) (aka the 100 Layer Tiramisu)

#### How to run
* Run file main.py, change the path to the data to fit the device
