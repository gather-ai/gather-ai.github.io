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
  <figcaption>Figure 1. Federated Learning Illustration. Adapted from [1]. </figcaption>
</figure>

### Formulation


## 2. A Network of IoT Devices

## References
[[1] Flower: A Friendly Federated Learning Framework](https://flower.dev/)<br>
[[2] A Survey on Federated Learning for Resource-Constrained IoT Devices](https://ieeexplore.ieee.org/document/9475501)<br>