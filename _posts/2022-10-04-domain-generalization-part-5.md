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

{: style="text-align: justify;"}

### Method

{: style="text-align: justify;"}

## 3. Results
The table below shows the performance of the two presented methods in this article: 
{: style="text-align: justify;"}

|            |    Chapman |       CPSC | CPSC-Extra |      G12EC |     Ningbo |     PTB-XL |        Avg |
| :--------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Baseline &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4290 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.1643 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.2067 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3809 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3987 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3626 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3237 |
| AgeReg | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4222 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.1715 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.2136 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3923 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4024 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4021 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3340 |
| SWA | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4271 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.1759 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.2052 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3969 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4313 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4203 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3428 |
| Mixup | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4225 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.1759 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.2127 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3901 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4025 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3934 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3329 |
| MixStyle | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4253 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.1681 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.2027 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3927 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4117 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3853 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3310 |
| DAT | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4282 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.1712 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.1966 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3956 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4114 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3878 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **0.3318** |
| I-BN | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4252 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.1748 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.2045 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3817 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4193 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4161 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **0.3369** |
| DS I-BN | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4484 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.1805 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.2191 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4318 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.3916 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0.4242 | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **0.3493** |
{: style="text-align: justify;"}

<!-- To be continued ... -->
{: style="text-align: justify;"}

## References
[[1] Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html)<br>
[[2] Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913)<br>