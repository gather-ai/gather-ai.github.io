---
title: "Federated Learning on IoT Devices Tutorials (Part 1): Introduction and Setup"
date: 2022-10-12
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

👋 Hi there. Recently, I have started working on the [Federated Learning](https://en.wikipedia.org/wiki/Federated_learning) (FL) field, Federated Learning deployment on IoT devices in specific. In this 2-part series of tutorials, I will use a powerful framework [Flower](https://flower.dev/) to implement a simple FL algorithm on a real network of IoT devices. Let's get started by summarizing what FL is. 
{: style="text-align: justify;"}

## 1. Background

### Motivation
Currently, there are nearly 7 billion connected Internet of Things (IoT) devices and 3 billion smartphones around the world. These devices generate data at the edge constantly. However, due to limits in data privacy regulations and communication bandwidth, it is usually infeasible to transmit and store all training data at a central location. Coupled with the rise of Machine Learning (ML), the wealth of data collected by end devices opens up countless possibilities for meaningful research and applications. 
{: style="text-align: justify;"}

From these observations, the topic of Federated Learning (FL) was introduced. FL is a distributed ML strategy that generates a global model by learning from multiple decentralized edge clients. FL enables on-device training, keeping the client's local data private, and further, updating the global model based on the local model updates. 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/federated-learning-iot/flower.jpg">
  <figcaption>Figure 1. Federated Learning Illustration. </figcaption>
</figure>

### Formulation
The federated learning problem involves learning a single, global model from data stored on tens to potentially millions of remote devices. In particular, the goal is typically to minimize the following objective function: 
{: style="text-align: justify;"}

$$\underset{\theta}{min}L(\theta)$$
{: style="text-align: justify;"}

where
{: style="text-align: justify;"}

$$L(\theta) := \sum_{m=1}^{M} p_mL_m(\theta)$$
{: style="text-align: justify;"}

Here $M$ is the total number of devices, $L_m$ is the local objective function for the $m^{th}$ device, and $p_m$ specifies the relative impact of each device with $p_m \geq 0$ and $\sum_{m=1}^{M}p_m = 1$. The local objective function $L_m$ is often defined as the empirical risk over local data. The relative impact of each device $p_m$ is user-defined. 
{: style="text-align: justify;"}

## 2. A Network of IoT Devices

### Network Ingredients
Most of the existing research on FL uses an FL setting simulation on a single machine. This does not make sense much because it does not introduce major issues of real FL like communication and system heterogeneity. In this tutorial, to introduce and handle these issues, I create and use a local network that consists of various types of edge devices. Specifically, I use 3 Raspberry Pi (RPi) 4 Model B, 2 NVIDIA Jetson Nano (Jetson) 2GB, and 5 NVIDIA Jetson Nano 4GB. Figure 2 shows the ingredients of the network where my laptop is used as a remote server and connects to all devices via local Wifi. 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/federated-learning-iot/network.jpg">
  <figcaption>Figure 2. A Network of IoT Devices. </figcaption>
</figure>

### Setup Environments
Setup and configuring environments for smoothly working on Raspberry Pi and NVIDIA Jetson Nano is a nightmare. Therefore, I recommend using [Docker](https://docs.docker.com/) for convenience and consistency. I have built Docker images for RPi and Jetson [here](https://hub.docker.com/repositories), you can pull and run them without additional installations. Make sure that you have booted Raspberry Pi OS (64-bit) for RPi and JetPack 4.6.1 for Jetson. 
{: style="text-align: justify;"}
* For RPi: 
```
$ docker pull lhkhiem28/rpi:1.4
$ docker run -it -d -w /root --network=host --name=rpi-container -v $(pwd):/usr/src/ lhkhiem28/rpi:1.4 /bin/bash
```
* For Jetson: 
```
$ docker pull lhkhiem28/jetson:1.0
$ docker run -it -d -w /root --runtime=nvidia --network=host --name=jetson-container -v $(pwd):/usr/src/ lhkhiem28/jetson:1.0 /bin/bash
```
{: style="text-align: justify;"}

Now we are ready to start deploying FL on IoT devices, in the next part, I will implement a common FL strategy FedAvg on the above network using Flower. 
{: style="text-align: justify;"}

## References
[[1] A Survey on Federated Learning for Resource-Constrained IoT Devices](https://ieeexplore.ieee.org/document/9475501)<br>
[[2] Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)<br>
