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

From these observations, the topic of Federated Learning (FL) was introduced. FL is a distributed machine learning strategy that generates a global model by learning from multiple decentralized edge clients. FL enables on-device training, keeping the client's local data private, and further, updating the global model based on the local model updates. 
{: style="text-align: justify;"}

<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/federated-learning-iot/flower.jpg">
  <figcaption>Figure 1. Federated Learning Illustration. </figcaption>
</figure>

### Formulation

## 2. A Network of IoT Devices