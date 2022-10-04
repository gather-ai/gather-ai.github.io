---
title: "Domain Generalization Tutorials (Part 5): Test-Time Adjustment"
date: 2022-10-04
categories: 
  - Tutorials
tags: 
  - ai research
  - deep learning
header: 
  image: "/assets/images/domain-generalization/shift-cover.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my page, this is part 5 of my tutorial series about the topic of Domain Generalization (DG). While all previous parts discussed DG methods that focus on the training phase, this article presents a new and unique approach that focuses on the test phase, namely **test-time adjustment**. 
{: style="text-align: justify;"}

You can find the source code of the whole series [here](https://github.com/lhkhiem28/DGECG). 
{: style="text-align: justify;"}
{: .notice--info}

## 1. Test-Time Adjustment
Test-time adjustment is a novel approach to DG problems where the trained model twists its parameters to correct its prediction by itself during the test time. Since no data about the target domain is available during training in a DG setup, the existing DG methods focus on how to use labeled data from multiple-source domains. However, at test time, the model always has access to test data from the target domain. Although the available data is constrained to be:
{: style="text-align: justify;"}
* unlabeled, 
* only available online (models can not know all test cases in advance), 
{: style="text-align: justify;"}
this data provides clues about the target distribution that is not available during training. It is natural to ask the question: How can we use the off-the-shelf unlabeled data available at test time to increase performance on the target domain?
{: style="text-align: justify;"}

## 2. Test-Time Template Adjuster

### Motivation
Test-time template adjuster (T3A) is a pioneer in this approach. The method is an optimization-free procedure that adjusts the linear classifier (the last layer of deep neural networks) at test time. This procedure makes the adjusted decision boundary avoid the high-data density region on the target domain and reduces the ambiguity (entropy) of predictions, which is known to be connected to classification error. One interesting property of T3A is that it does not alter the training phase, therefore it can be used together with any existing DG algorithms. Moreover, it can be used together with any classification model since it only adjusts the linear classifier on top of the representations. 
{: style="text-align: justify;"}

### Method
Firstly, what is "template" in the name of T3A? Let's say the linear classifier of a trained model is denoted as $g$ with the parameters $\theta_{g}$. $\theta_{g}$ has a shape of $dim_{z}\times C$, where $dim_{z}$ is the dimension of output from feature extractor $f$ and $C$ is the total number of classes. The template of representation for the class $k$ is defined as:
{: style="text-align: justify;"}

$$\omega^k = \theta_{g}[:, k]. $$
{: style="text-align: justify;"}

<!-- To be continued ... -->
{: style="text-align: justify;"}

## References
[[1] Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726)<br>
[[2] Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html)<br>