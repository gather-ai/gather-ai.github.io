---
title: "Federated Learning on IoT Devices Tutorials (Part 2): From Zero to Hero"
date: 2022-10-19
categories: 
  - Tutorials
tags: 
  - ai research
  - federated learning
  - iot
header: 
  image: "/assets/images/federated-learning-iot/flower-cover.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my page, this is part 2 of my tutorial series on deploying Federated Learning on IoT devices. In the last article, we discussed what FL is and built a network of IoT devices as well as environments for starting work. Today, I will guide you step by step to train a simple CNN model on the CIFAR10 dataset in real IoT devices by using [Flower](https://flower.dev/). Let's get started. 
{: style="text-align: justify;"}

## 1. Preparing Dataset

### CIFAR10 Dataset
The CIFAR10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Here are the classes in the dataset, as well as 10 random images from each: 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/federated-learning-iot/cifar10.jpg">
  <figcaption>Figure 1. CIFAR10 Dataset. Mount from [1]</figcaption>
</figure>

### Data Partition

## 2. A Simple CNN Model


## 3. Client Site


## 4. Server Site


## 5. Running


## References
[[1] CIFAR10 and CIFAR100 Datasets](https://www.cs.toronto.edu/~kriz/cifar.html)<br>
[[2] Flower: A Friendly Federated Learning Framework](https://flower.dev/)<br>