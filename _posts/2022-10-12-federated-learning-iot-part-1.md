---
title: "Federated Learning on IoT Devices Tutorials (Part 1): Introduction"
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

👋 Hi there. Recently, I have started working on the [Federated Learning](https://en.wikipedia.org/wiki/Federated_learning) (FL) field, Federated Learning deployment on IoT devices in specific. There are many powerful frameworks outside such as [Flower](https://flower.dev/), [FedML](https://doc.fedml.ai/) to help you learn and practice the topic easily. However, I believe that we cannot understand the real challenges and issues of applying FL to an IoT environment when using these frameworks. Therefore, this 2-part tutorial walks through implementing an FL setting on a network of IoT devices with pure Python. Let's get started by summarizing what is FL. 
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
Most of the existing research on FL uses an FL setting simulation on a single machine. This does not make sense much because it does not introduce major issues of real FL like communication and system heterogeneity. In this tutorial, to introduce and handle these issues, I create and use a local network that consists of various types of edge devices. Specifically, I use 2 Raspberry Pi 4 Model B, 2 NVIDIA Jetson Nano, and 1 NVIDIA Jetson Nano 2GB. Figure 2 shows the ingredients of the network where my laptop is used as a remote server and connects to all devices via local Wifi. 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/federated-learning-iot/network.jpg">
  <figcaption>Figure 2. A Network of IoT Devices. </figcaption>
</figure>

## References
[[1] A Survey on Federated Learning for Resource-Constrained IoT Devices](https://ieeexplore.ieee.org/document/9475501)<br>
[[2] Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)<br>