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

### Multi-task Learning


### Flat Minima Seeking
