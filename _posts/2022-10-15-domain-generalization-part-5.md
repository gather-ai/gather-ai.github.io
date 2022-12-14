---
title: "Domain Generalization Tutorials - Part 5"
subtitle: "Test-Time Adjustment"
date: 2022-10-15
categories: 
  - Tutorials
tags: 
  - AI Research
  - Deep Learning
header: 
  image: "/assets/images/domain-generalization/shift-cover.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my page, this is part 5 of my tutorial series about the topic of Domain Generalization (DG). While all previous parts discussed DG methods that focus on the training phase, this article presents a new and unique approach that focuses on the test phase, namely **test-time adjustment**. 

<!-- You can find the source code of the whole series [here](https://github.com/lhkhiem28/DGECG). 
{: .notice--info} -->

## 1. Test-Time Adjustment
Test-time adjustment is a novel approach to DG problems where the trained model twists its parameters to correct its prediction by itself during the test time. Since no data about the target domain is available during training in a DG setup, the existing DG methods focus on how to use labeled data from multiple-source domains. However, at test time, the model always has access to test data from the target domain. Although the available data is constrained to be:
* unlabeled, 
* only available online (models can not know all test cases in advance), 
this data provides clues about the target distribution that is not available during training. It is natural to ask the question: How can we use the off-the-shelf unlabeled data available at test time to increase performance on the target domain?

## 2. Test-Time Template Adjuster

### Motivation
Test-time template adjuster (T3A) is a pioneer in this approach. The method is an optimization-free procedure that adjusts the linear classifier (the last layer of deep neural networks) at test time. This procedure makes the adjusted decision boundary avoid the high-data density region on the target domain and reduces the ambiguity (entropy) of predictions, which is known to be connected to classification error. One interesting property of T3A is that it does not alter the training phase, therefore it can be used together with any existing DG algorithms. Moreover, it can be used together with any classification model since it only adjusts the linear classifier on top of the representations. 

### Method
Firstly, what is "template" in the name of T3A? Let's say the linear classifier of a trained model is denoted as $g$ with the parameters $\theta_{g}$. $\theta_{g}$ has a shape of $dim_{z}\times C$, where $dim_{z}$ is the dimension of output from feature extractor $f$ and $C$ is the total number of classes. The template of representations for the class $k$ is defined as: 

$$\omega^k = \theta_{g}[:, k]$$

During test time, the model generates its logits by measuring the distance (dot product) between its templates and the representations $z$ of the input data $x$, then the prediction $\widehat{y}$ is made by final operations, e.g., softmax function for multi-class classification: 

$$logit^k = z\omega^k$$

Since these templates were trained in the source domain, there is no guarantee that they will be a good template in the target domain. 

Next, how does T3A adjust the model templates to make better predictions on the target domain? Assume we have (batch of) test data $x$ at time $t$, T3A introduces a _support set_ $\mathbb{S}_t^k$ for each class $k$: 

$$
\begin{align}
\mathbb{S}_t^k &= \begin{cases}
\mathbb{S}_{t-1}^k \cup \{ \frac{f(x)}{\left \| f(x) \right \|} \} & \text{if $\widehat{y}=k$} \\ \mathbb{S}_{t-1}^k & \text{else}
\end{cases}
\end{align}
$$

where $$\left \| \cdot \right \|$$ represents the L2 norm of a vector and $$\mathbb{S}_0^k = \{ \frac{\omega^k}{\left \| \omega^k \right \|} \}$$. If the input data contains multiple samples at the same time (e.g., a batch of data), the above procedure is repeated for each sample in the batch. 

Then, T3A uses centroids of these support sets as adjusted templates to make it new prediction: 

$$c^k = \frac{1}{\left | \mathbb{S}^k \right |}\sum_{s\in  \mathbb{S}^k}s$$

and

$$logit^k = zc^k$$

then the prediction $\widehat{y}$ is made by final operations, e.g., softmax function for multi-class classification, sigmoid and thresholding for multi-label classification. 

<head><style>hr.solid {border-top: 1px solid #bbb;}</style></head>
<body><hr class="solid"></body>

This is the final part of my tutorial series on Domain Generalization (DG). Actually, there is another interesting approach to DG which is based on [Meta-Learning](https://en.wikipedia.org/wiki/Meta_learning_(computer_science)). I might come back to this approach later. 

## References
[[1] Tent: Fully Test-time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726)<br>
[[2] Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html)<br>
{: style="font-size: 14px;"}