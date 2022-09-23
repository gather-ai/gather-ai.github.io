---
title: "Domain Generalization Tutorials (Part 2): Multi-task Learning and Flat Minima Seeking"
date: 2022-09-20
categories: 
  - Tutorials
tags: 
  - deep learning
header: 
  image: "/assets/images/domain-generalization/shift-cover.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my website, this is part 2 of my tutorial series about the topic of Domain Generalization (DG). In this article, I will introduce the first approach to the DG problem, which I call **conventional generalization**. 
{: style="text-align: justify;"}

You can find the source code of the whole series [here](https://github.com/lhkhiem28/DGECG). 
{: style="text-align: justify;"}
{: .notice--info}

## 1. Conventional Generalization
Conventional generalization methods such as data augmentation or weight decay aim to make ML models less overfit on training data, therefore these models after training are assumpted to generalize well on testing data regardless of its domain. This is a great starting point to approach the DG problem. Despite the popularity of data augmentation and weight decay, I will present two more advanced and effective methods, _multi-task learning_ and _flat minima seeking_. 
{: style="text-align: justify;"}

## 2. Multi-task Learning

### Motivation
The goal of Multi-task Learning (with deep neural networks) is to jointly learn one or more sub-tasks beside the main task using a shared model, therefore facilitating the model’s shared representations to be generic enough to deal with different tasks, eventually reducing overfitting on the main task. In general, sub-tasks for performing multi-task learning are defined based on specific data and problems. After that, jointly learning is established by minimizing a joint loss function. 
{: style="text-align: justify;"}

Multi-task Learning is popular in ML literature but rarely realized. For example, in Computer Vision, [Object Detection](https://paperswithcode.com/task/object-detection) aims to localize and classify objects simultaneously. In Natural Language Processing, [Intent Detection and Slot Filling](http://nlpprogress.com/english/intent_detection_slot_filling.html) aims to simultaneously identify the speaker's intent from a given utterance and extract from the utterance the correct argument value for the slots of the intent. 
{: style="text-align: justify;"}

### Method


## 3. Flat Minima Seeking

### Motivation


### Method

